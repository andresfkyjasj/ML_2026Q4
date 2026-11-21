"""Transformers para entender a los usuarios de un asistente virtual (Semana 7).

Caso: un asistente virtual recibe frases como «pon algo de Shakira en la sala» y debe
ejecutar la acción correcta al primer intento, o pasar la conversación a un humano
cuando no está seguro.

Datos: MASSIVE (Amazon Science), versión en español de SetFit en Hugging Face
(`SetFit/amazon_massive_intent_es-ES`): 11 514 frases de entrenamiento, 2 033 de
validación y 2 974 de prueba, con 60 intenciones. Licencia CC BY 4.0.

Todo el modelo está escrito a mano con PyTorch (sin `nn.TransformerEncoder`) para que
cada operación del forward pass se pueda señalar en el código:

    texto → tokens (WordPiece) → embeddings + posición → [atención → residual + norma →
    feed-forward → residual + norma] × N bloques → promedio → 60 puntajes → softmax
"""
from __future__ import annotations

import json
import math
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

URL_HF = "https://huggingface.co/datasets/SetFit/amazon_massive_intent_es-ES/resolve/main/{split}.jsonl"
SPLITS = {"entrenamiento": "train", "validación": "validation", "prueba": "test"}
ESPECIALES = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
PAD, UNK, BOS, EOS = 0, 1, 2, 3

# Supuestos de negocio (pesos colombianos), ajustables en la app
COSTO_ERROR = 20_000        # acción equivocada: el usuario reclama, se reprocesa, queda molesto
COSTO_HUMANO = 6_000        # pasar la conversación a un agente humano

# Frases que NO pertenecen a ningún dominio del asistente: sirven para ver la «alucinación»
FUERA_DE_DOMINIO = [
    "transfiere doscientos mil pesos a la cuenta de ahorros de mi mamá",
    "mi pedido llegó roto y quiero que me devuelvan el dinero",
    "¿me aprobaron el crédito de vivienda?",
    "quiero cancelar mi tarjeta de crédito",
    "agenda una cita con el dermatólogo de la eps",
    "asdf qwerty zxcv",
    "el gato azul baila con la luna en el tejado",
    "¿cuántos puntos tengo en mi programa de millas?",
]


# ----------------------------------------------------------------------------- datos
def normalizar(texto: str) -> str:
    """Minúsculas y forma Unicode NFC (las tildes se conservan)."""
    return unicodedata.normalize("NFC", str(texto).strip().lower())


def cargar_massive(carpeta: str | Path | None = None) -> tuple[dict[str, pd.DataFrame], str]:
    """Devuelve ({conjunto: DataFrame}, origen). Orden: carpeta local → Hugging Face.

    Cada DataFrame tiene: `texto`, `etiqueta` (0–59), `intencion` (en inglés) e
    `intencion_es` (traducción del dataset).
    """
    carpeta = Path(carpeta) if carpeta else None
    datos, origen = {}, ""
    for nombre, split in SPLITS.items():
        local = carpeta / f"{split}.jsonl" if carpeta else None
        if local and local.exists():
            df, origen = pd.read_json(local, lines=True), f"archivos locales en {carpeta.name}/"
        else:
            df = pd.read_json(URL_HF.format(split=split), lines=True)
            origen = "Hugging Face · SetFit/amazon_massive_intent_es-ES"
            if carpeta:
                carpeta.mkdir(parents=True, exist_ok=True)
                df.to_json(carpeta / f"{split}.jsonl", orient="records", lines=True, force_ascii=False)
        datos[nombre] = pd.DataFrame({"texto": df["text"].map(normalizar), "etiqueta": df["label"].astype(int),
                                      "intencion": df["label_text"], "intencion_es": df["label_text_es"]})
    return datos, origen


def catalogo_intenciones(datos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tabla etiqueta → intención → dominio (lo que va antes del primer «_»)."""
    t = (pd.concat(datos.values()).drop_duplicates("etiqueta").sort_values("etiqueta")
         [["etiqueta", "intencion", "intencion_es"]].reset_index(drop=True))
    t["dominio"] = t["intencion"].str.split("_").str[0]
    return t


# ----------------------------------------------------------------------------- tokenizador
class TokenizadorWordPiece:
    """Tokenizador de subpalabras (WordPiece, el mismo esquema que BERT y BETO).

    Se entrena con la librería `tokenizers` de Hugging Face y se aplica con el algoritmo
    «la pieza más larga primero», que es fácil de seguir a mano (y de programar en la app):
    para cada palabra se busca el prefijo más largo que esté en el vocabulario; lo que
    sobra se busca con el prefijo `##`, que indica «continuación de palabra».
    """

    def __init__(self, vocab: list[str]):
        self.vocab = list(vocab)
        self.id = {t: i for i, t in enumerate(self.vocab)}

    @classmethod
    def entrenar(cls, textos: list[str], tam_vocab: int = 2000) -> "TokenizadorWordPiece":
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
        tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tk.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Lowercase()])
        tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        trainer = trainers.WordPieceTrainer(vocab_size=tam_vocab, special_tokens=ESPECIALES, min_frequency=2)
        tk.train_from_iterator(textos, trainer)
        v = tk.get_vocab()
        return cls([t for t, _ in sorted(v.items(), key=lambda x: x[1])])

    @staticmethod
    def palabras(texto: str) -> list[str]:
        """Separa por espacios y deja cada signo de puntuación como palabra aparte."""
        out, actual = [], ""
        for ch in normalizar(texto):
            if ch.isspace():
                if actual:
                    out.append(actual); actual = ""
            elif unicodedata.category(ch).startswith("P"):
                if actual:
                    out.append(actual); actual = ""
                out.append(ch)
            else:
                actual += ch
        if actual:
            out.append(actual)
        return out

    def piezas(self, texto: str) -> list[str]:
        """Texto → lista de tokens (piezas)."""
        res = []
        for w in self.palabras(texto):
            i, partes = 0, []
            while i < len(w):
                j, pieza = len(w), None
                while j > i:
                    cand = w[i:j] if i == 0 else "##" + w[i:j]
                    if cand in self.id:
                        pieza = cand; break
                    j -= 1
                if pieza is None:
                    partes = ["[UNK]"]; break
                partes.append(pieza); i = j
            res.extend(partes)
        return res

    def codificar(self, texto: str) -> list[int]:
        """Texto → ids (las «direcciones» en la tabla de embeddings)."""
        return [self.id.get(p, UNK) for p in self.piezas(texto)]

    def decodificar(self, ids: list[int]) -> str:
        texto = ""
        for i in ids:
            t = self.vocab[i]
            if t in ESPECIALES:
                continue
            texto += t[2:] if t.startswith("##") else (" " + t if texto else t)
        return texto


# ----------------------------------------------------------------------------- modelo
def codificacion_posicional(n: int, d: int) -> torch.Tensor:
    """Huella de posición sinusoidal (Vaswani et al., 2017): PE[p, 2i] = sin(p / 10000^(2i/d)),
    PE[p, 2i+1] = cos(p / 10000^(2i/d))."""
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class BloqueTransformer(nn.Module):
    """Un bloque: atención multi-cabeza → residual + norma → feed-forward (ReLU) → residual + norma."""

    def __init__(self, d: int, cabezas: int, d_ff: int, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        assert d % cabezas == 0
        self.d, self.h, self.dk, self.causal = d, cabezas, d // cabezas, causal
        self.Wq, self.Wk, self.Wv, self.Wo = (nn.Linear(d, d) for _ in range(4))
        self.norma1, self.norma2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.W1, self.W2 = nn.Linear(d, d_ff), nn.Linear(d_ff, d)
        self.drop = nn.Dropout(dropout)

    def atencion(self, x: torch.Tensor, mascara: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, _ = x.shape
        div = lambda t: t.view(b, n, self.h, self.dk).transpose(1, 2)            # (b, h, n, dk)
        Q, K, V = div(self.Wq(x)), div(self.Wk(x)), div(self.Wv(x))
        S = Q @ K.transpose(-2, -1) / math.sqrt(self.dk)                          # puntajes (b, h, n, n)
        bloqueo = ~mascara[:, None, None, :]                                      # no mirar el relleno
        if self.causal:                                                           # GPT: no mirar el futuro
            bloqueo = bloqueo | torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), 1)
        A = F.softmax(S.masked_fill(bloqueo, -1e9), dim=-1)                       # cada fila suma 1
        mezcla = (self.drop(A) @ V).transpose(1, 2).reshape(b, n, self.d)         # concatena cabezas
        return self.Wo(mezcla), A

    def forward(self, x, mascara):
        a, A = self.atencion(x, mascara)
        x = self.norma1(x + self.drop(a))                                          # residual + norma
        f = self.W2(self.drop(F.relu(self.W1(x))))                                 # la red de la Semana 6
        return self.norma2(x + self.drop(f)), A


class MiniTransformer(nn.Module):
    """Codificador (tipo BERT) para clasificar, o decodificador (tipo GPT) si `causal=True`."""

    def __init__(self, tam_vocab: int, n_clases: int | None, d: int = 16, cabezas: int = 2, d_ff: int = 32,
                 bloques: int = 2, max_len: int = 32, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.config = dict(tam_vocab=tam_vocab, n_clases=n_clases, d=d, cabezas=cabezas, d_ff=d_ff,
                           bloques=bloques, max_len=max_len, dropout=dropout, causal=causal)
        self.E = nn.Embedding(tam_vocab, d, padding_idx=PAD)
        self.register_buffer("PE", codificacion_posicional(max_len, d))
        self.bloques = nn.ModuleList(BloqueTransformer(d, cabezas, d_ff, dropout, causal) for _ in range(bloques))
        self.salida = nn.Linear(d, n_clases if not causal else tam_vocab)
        self.drop = nn.Dropout(dropout)

    def forward(self, ids: torch.Tensor, devolver_atencion: bool = False):
        mascara = ids != PAD
        x = self.drop(self.E(ids) + self.PE[: ids.shape[1]])                       # embedding + posición
        atenciones = []
        for b in self.bloques:
            x, A = b(x, mascara)
            atenciones.append(A)
        if self.config["causal"]:
            out = self.salida(x)                                                   # puntaje de cada token siguiente
        else:
            m = mascara.unsqueeze(-1).float()
            out = self.salida((x * m).sum(1) / m.sum(1))                           # promedio de tokens → 60 puntajes
        return (out, atenciones) if devolver_atencion else out

    def n_parametros(self) -> pd.DataFrame:
        filas = [("Embeddings (vocabulario × d)", self.E.weight.numel())]
        for i, b in enumerate(self.bloques, 1):
            filas += [(f"Bloque {i} · atención (W_Q, W_K, W_V, W_O)", sum(p.numel() for n, p in b.named_parameters() if n[:2] in ("Wq", "Wk", "Wv", "Wo"))),
                      (f"Bloque {i} · feed-forward (W₁, W₂)", sum(p.numel() for n, p in b.named_parameters() if n[:2] in ("W1", "W2"))),
                      (f"Bloque {i} · normalizaciones", sum(p.numel() for n, p in b.named_parameters() if n.startswith("norma")))]
        filas.append(("Cabeza de salida", sum(p.numel() for p in self.salida.parameters())))
        t = pd.DataFrame(filas, columns=["capa", "parámetros"])
        return pd.concat([t, pd.DataFrame([{"capa": "Total", "parámetros": int(t["parámetros"].sum())}])], ignore_index=True)


def _lote(ids: list[list[int]], max_len: int) -> torch.Tensor:
    n = min(max(len(s) for s in ids), max_len)
    return torch.tensor([s[:n] + [PAD] * (n - len(s[:n])) for s in ids])


# ----------------------------------------------------------------------------- caso de negocio
class AsistenteIntenciones:
    """Entrena y evalúa el Transformer de intenciones y responde la pregunta de negocio.

    Flujo: `preparar()` → `entrenar()` → `evaluar()` → `elegir_umbral()` → `respuesta_negocio()`.
    """

    def __init__(self, d: int = 16, cabezas: int = 2, d_ff: int = 32, bloques: int = 2, tam_vocab: int = 2000,
                 epocas: int = 40, tasa_aprendizaje: float = 3e-3, semilla: int = 7,
                 costo_error: float = COSTO_ERROR, costo_humano: float = COSTO_HUMANO):
        self.hp = dict(d=d, cabezas=cabezas, d_ff=d_ff, bloques=bloques, tam_vocab=tam_vocab, epocas=epocas,
                       tasa_aprendizaje=tasa_aprendizaje, semilla=semilla)
        self.costo_error, self.costo_humano = float(costo_error), float(costo_humano)
        self.tok: TokenizadorWordPiece | None = None
        self.modelo: MiniTransformer | None = None
        self.umbral = 0.0
        self.historial: list[dict] = []

    def preparar(self, datos: dict[str, pd.DataFrame], tokenizador: TokenizadorWordPiece | None = None):
        self.datos = datos
        self.catalogo = catalogo_intenciones(datos)
        self.tok = tokenizador or TokenizadorWordPiece.entrenar(list(datos["entrenamiento"]["texto"]), self.hp["tam_vocab"])
        self.ids = {k: [self.tok.codificar(t) for t in df["texto"]] for k, df in datos.items()}
        return self

    def entrenar(self, verbose: bool = True) -> "AsistenteIntenciones":
        torch.manual_seed(self.hp["semilla"]); np.random.seed(self.hp["semilla"])
        h = self.hp
        self.modelo = MiniTransformer(len(self.tok.vocab), len(self.catalogo), h["d"], h["cabezas"], h["d_ff"], h["bloques"])
        opt = torch.optim.AdamW(self.modelo.parameters(), lr=h["tasa_aprendizaje"], weight_decay=1e-2)
        ids, y = self.ids["entrenamiento"], torch.tensor(self.datos["entrenamiento"]["etiqueta"].to_numpy())
        mejor, estado, t0 = -1.0, None, time.time()
        for ep in range(1, h["epocas"] + 1):
            self.modelo.train()
            orden = np.random.permutation(len(ids)); perdida = 0.0
            for i in range(0, len(ids), 64):
                idx = orden[i:i + 64]
                logits = self.modelo(_lote([ids[j] for j in idx], 32))
                loss = F.cross_entropy(logits, y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                perdida += loss.item() * len(idx)
            acc_val = (self.predecir_proba(ids=self.ids["validación"]).argmax(1) == self.datos["validación"]["etiqueta"].to_numpy()).mean()
            self.historial.append({"época": ep, "pérdida": perdida / len(ids), "exactitud validación": acc_val})
            if acc_val > mejor:
                mejor, estado = acc_val, {k: v.clone() for k, v in self.modelo.state_dict().items()}
            if verbose and (ep % 10 == 0 or ep == 1):
                print(f"época {ep:>3} · pérdida {perdida / len(ids):.3f} · exactitud en validación {acc_val:.3f}")
        self.modelo.load_state_dict(estado)                                        # el mejor en validación
        self.segundos = time.time() - t0
        self.elegir_umbral()
        return self

    # -- predicción
    @torch.no_grad()
    def predecir_proba(self, textos: list[str] | None = None, ids: list[list[int]] | None = None) -> np.ndarray:
        self.modelo.eval()
        ids = ids if ids is not None else [self.tok.codificar(t) for t in textos]
        out = [F.softmax(self.modelo(_lote(ids[i:i + 512], 32)), -1).numpy() for i in range(0, len(ids), 512)]
        return np.concatenate(out)

    @torch.no_grad()
    def explicar(self, texto: str) -> dict[str, Any]:
        """Una frase: piezas, ids, atención de cada bloque y las 5 intenciones más probables."""
        self.modelo.eval()
        ids = self.tok.codificar(texto)
        logits, A = self.modelo(torch.tensor([ids]), devolver_atencion=True)
        p = F.softmax(logits, -1)[0].numpy()
        top = np.argsort(-p)[:5]
        return {"piezas": self.tok.piezas(texto), "ids": ids,
                "atencion": [a[0].numpy() for a in A],                               # bloque → (cabezas, n, n)
                "top": [(self.catalogo.intencion[i], float(p[i])) for i in top],
                "confianza": float(p.max()), "decision": "Ejecutar" if p.max() >= self.umbral else "Pasar a un humano"}

    # -- evaluación y negocio
    def evaluar(self, conjunto: str = "prueba") -> dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score
        y = self.datos[conjunto]["etiqueta"].to_numpy()
        pred = self.predecir_proba(ids=self.ids[conjunto]).argmax(1)
        return {"exactitud": accuracy_score(y, pred), "F1 macro": f1_score(y, pred, average="macro")}

    def costos(self, conjunto: str, umbral: float, costo_error: float | None = None,
               costo_humano: float | None = None) -> dict[str, float]:
        """Si la confianza supera el umbral se ejecuta (y si se equivoca cuesta `costo_error`);
        si no, se pasa a un humano (cuesta `costo_humano` y se resuelve bien)."""
        ce = self.costo_error if costo_error is None else costo_error
        ch = self.costo_humano if costo_humano is None else costo_humano
        p = self.predecir_proba(ids=self.ids[conjunto]); y = self.datos[conjunto]["etiqueta"].to_numpy()
        conf, pred = p.max(1), p.argmax(1)
        ejecuta = conf >= umbral
        errores = int((ejecuta & (pred != y)).sum()); escaladas = int((~ejecuta).sum())
        n = len(y)
        return {"umbral": umbral, "automáticas": int(ejecuta.sum()), "aciertos automáticos": int((ejecuta & (pred == y)).sum()),
                "errores": errores, "a humano": escaladas, "% automáticas": ejecuta.mean(), "% errores": errores / n,
                "% a humano": escaladas / n, "costo": errores * ce + escaladas * ch,
                "costo todo humano": n * ch, "costo sin umbral": int((pred != y).sum()) * ce}

    def curva_costos(self, conjunto: str = "validación", **kw) -> pd.DataFrame:
        return pd.DataFrame([self.costos(conjunto, u, **kw) for u in np.round(np.linspace(0, 0.99, 100), 2)])

    def elegir_umbral(self, **kw) -> float:
        c = self.curva_costos("validación", **kw)
        self.umbral = float(c.loc[c["costo"].idxmin(), "umbral"])
        return self.umbral

    def respuesta_negocio(self, conjunto: str = "prueba") -> dict[str, Any]:
        c = self.costos(conjunto, self.umbral)
        c["ahorro vs todo humano"] = c["costo todo humano"] - c["costo"]
        c["ahorro %"] = c["ahorro vs todo humano"] / c["costo todo humano"]
        return c

    def fuera_de_dominio(self, frases: list[str] = FUERA_DE_DOMINIO) -> pd.DataFrame:
        """Frases que no pertenecen a ningún dominio: el modelo igual elige una intención."""
        p = self.predecir_proba(frases)
        return pd.DataFrame({"frase": frases, "intención elegida": self.catalogo.intencion[p.argmax(1)].to_numpy(),
                             "confianza": p.max(1).round(3),
                             "decisión": np.where(p.max(1) >= self.umbral, "Ejecutar (¡alucina!)", "Pasar a un humano")})

    # -- exportar para la animación y la app web
    def exportar_json(self, ruta: str | Path, decimales: int = 5) -> Path:
        sd = {k: np.round(v.numpy(), decimales).tolist() for k, v in self.modelo.state_dict().items() if k != "PE"}
        obj = {"config": self.modelo.config, "vocab": self.tok.vocab, "intenciones": list(self.catalogo.intencion),
               "intenciones_es": list(self.catalogo.intencion_es), "umbral": self.umbral,
               "costo_error": self.costo_error, "costo_humano": self.costo_humano, "pesos": sd}
        ruta = Path(ruta); ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return ruta

    def guardar(self, ruta: str | Path) -> Path:
        ruta = Path(ruta); ruta.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hp": self.hp, "vocab": self.tok.vocab, "catalogo": self.catalogo.to_dict("list"),
                    "estado": self.modelo.state_dict(), "config": self.modelo.config, "umbral": self.umbral,
                    "costos": (self.costo_error, self.costo_humano), "historial": self.historial}, ruta)
        return ruta

    @classmethod
    def cargar(cls, ruta: str | Path) -> "AsistenteIntenciones":
        g = torch.load(ruta, weights_only=False)
        obj = cls(**{k: v for k, v in g["hp"].items()}, costo_error=g["costos"][0], costo_humano=g["costos"][1])
        obj.tok = TokenizadorWordPiece(g["vocab"])
        obj.catalogo = pd.DataFrame(g["catalogo"])
        obj.modelo = MiniTransformer(**g["config"]); obj.modelo.load_state_dict(g["estado"]); obj.modelo.eval()
        obj.umbral = g["umbral"]
        obj.historial = g.get("historial", [])
        return obj


# ----------------------------------------------------------------------------- generar texto (tipo GPT)
class GeneradorFrases:
    """Decodificador pequeño (tipo GPT) que aprende a predecir el siguiente token de las frases."""

    def __init__(self, tok: TokenizadorWordPiece, d: int = 32, cabezas: int = 2, d_ff: int = 64, bloques: int = 2,
                 epocas: int = 25, semilla: int = 7):
        self.tok, self.hp = tok, dict(d=d, cabezas=cabezas, d_ff=d_ff, bloques=bloques, epocas=epocas, semilla=semilla)

    def entrenar(self, textos: list[str], verbose: bool = True) -> "GeneradorFrases":
        torch.manual_seed(self.hp["semilla"]); np.random.seed(self.hp["semilla"])
        h = self.hp
        self.modelo = MiniTransformer(len(self.tok.vocab), None, h["d"], h["cabezas"], h["d_ff"], h["bloques"], causal=True)
        opt = torch.optim.AdamW(self.modelo.parameters(), lr=3e-3, weight_decay=1e-2)
        seqs = [[BOS] + self.tok.codificar(t)[:30] + [EOS] for t in textos]
        for ep in range(1, h["epocas"] + 1):
            self.modelo.train(); orden = np.random.permutation(len(seqs)); tot = 0.0
            for i in range(0, len(seqs), 64):
                x = _lote([seqs[j] for j in orden[i:i + 64]], 32)
                logits = self.modelo(x[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), ignore_index=PAD)
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            if verbose and (ep % 5 == 0 or ep == 1):
                print(f"época {ep:>3} · pérdida {tot / math.ceil(len(seqs) / 64):.3f}")
        self.modelo.eval()
        return self

    def guardar(self, ruta: str | Path) -> Path:
        ruta = Path(ruta); ruta.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hp": self.hp, "vocab": self.tok.vocab, "config": self.modelo.config, "estado": self.modelo.state_dict()}, ruta)
        return ruta

    @classmethod
    def cargar(cls, ruta: str | Path) -> "GeneradorFrases":
        g = torch.load(ruta, weights_only=False)
        obj = cls(TokenizadorWordPiece(g["vocab"]), **g["hp"])
        obj.modelo = MiniTransformer(**g["config"]); obj.modelo.load_state_dict(g["estado"]); obj.modelo.eval()
        return obj

    @torch.no_grad()
    def siguiente(self, texto: str, k: int = 8, temperatura: float = 1.0) -> pd.DataFrame:
        """Probabilidad de cada posible token siguiente (las k más altas)."""
        ids = [BOS] + self.tok.codificar(texto)
        p = F.softmax(self.modelo(torch.tensor([ids]))[0, -1] / temperatura, -1).numpy()
        top = np.argsort(-p)[:k]
        return pd.DataFrame({"token": [self.tok.vocab[i] for i in top], "probabilidad": p[top].round(4)})

    @torch.no_grad()
    def generar(self, inicio: str, temperatura: float = 1.0, max_tokens: int = 15, semilla: int = 0) -> str:
        """Elige un token, lo pega y vuelve a predecir, hasta [EOS]."""
        g = torch.Generator().manual_seed(semilla)
        ids = [BOS] + self.tok.codificar(inicio)
        for _ in range(max_tokens):
            p = F.softmax(self.modelo(torch.tensor([ids]))[0, -1] / temperatura, -1)
            nxt = int(torch.multinomial(p, 1, generator=g)) if temperatura > 0 else int(p.argmax())
            if nxt == EOS:
                break
            ids.append(nxt)
        return self.tok.decodificar(ids)
