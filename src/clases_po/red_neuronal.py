"""Red neuronal (perceptrón multicapa) para mantenimiento predictivo industrial (Semana 7).

Caso: una planta metalmecánica con fresadoras CNC quiere anticipar las fallas de sus
máquinas para programar mantenimiento preventivo en lugar de sufrir paradas no planificadas.

Datos: *AI4I 2020 Predictive Maintenance Dataset* (UCI, 10 000 registros, 14 columnas).
Si no hay internet, `generar_datos_sinteticos()` produce un conjunto equivalente con las
mismas reglas físicas de falla que describe la documentación del dataset.

La clase no se guarda con pickle directamente: `guardar()` escribe un diccionario con objetos
de scikit-learn y pandas. Así el archivo se puede abrir tanto desde el paquete `clases_po`
(notebook) como desde la app de Streamlit, que importa el módulo suelto.
"""
from __future__ import annotations

import copy
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

URL_AI4I = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"

# Nombres originales del dataset → nombres en español usados en todo el curso
RENOMBRAR = {
    "UDI": "udi",
    "Product ID": "producto_id",
    "Type": "tipo",
    "Air temperature [K]": "temp_aire_K",
    "Process temperature [K]": "temp_proceso_K",
    "Rotational speed [rpm]": "velocidad_rpm",
    "Torque [Nm]": "torque_Nm",
    "Tool wear [min]": "desgaste_min",
    "Machine failure": "falla",
}
MODOS_FALLA = {
    "TWF": "Desgaste de herramienta",
    "HDF": "Disipación de calor",
    "PWF": "Potencia fuera de rango",
    "OSF": "Sobreesfuerzo",
    "RNF": "Aleatoria",
}
TIPOS = {"L": "Baja calidad (L)", "M": "Media calidad (M)", "H": "Alta calidad (H)"}
LIMITE_SOBREESFUERZO = {"L": 11_000, "M": 12_000, "H": 13_000}   # desgaste × torque [min·Nm]

VARIABLES_BASE = ["temp_aire_K", "temp_proceso_K", "velocidad_rpm", "torque_Nm", "desgaste_min"]
VARIABLES_DERIVADAS = ["dif_temp_K", "potencia_W", "esfuerzo_minNm"]
OBJETIVO = "falla"

# Rangos de operación observados en AI4I 2020 (sirven para los controles del frontend)
RANGOS = {
    "temp_aire_K": (295.0, 305.0, 300.0),
    "temp_proceso_K": (305.0, 314.0, 310.0),
    "velocidad_rpm": (1150, 2900, 1500),
    "torque_Nm": (3.0, 77.0, 40.0),
    "desgaste_min": (0, 255, 100),
}


# ----------------------------------------------------------------------------- datos
def agregar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables físicas que explican las fallas.

    - `dif_temp_K`: temperatura de proceso − aire (si es baja, la máquina no disipa calor).
    - `potencia_W`: torque × velocidad angular (ω = rpm · 2π / 60).
    - `esfuerzo_minNm`: desgaste × torque (sobreesfuerzo de la herramienta).
    """
    d = df.copy()
    d["dif_temp_K"] = d["temp_proceso_K"] - d["temp_aire_K"]
    d["potencia_W"] = d["torque_Nm"] * d["velocidad_rpm"] * 2 * np.pi / 60
    d["esfuerzo_minNm"] = d["desgaste_min"] * d["torque_Nm"]
    return d


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Acepta un CSV con los nombres originales de AI4I o con los nombres en español."""
    d = df.rename(columns={k: v for k, v in RENOMBRAR.items() if k in df.columns})
    d.columns = [c.strip() for c in d.columns]
    faltan = [c for c in ["tipo", *VARIABLES_BASE] if c not in d.columns]
    if faltan:
        raise ValueError(f"Faltan columnas: {faltan}. Se esperan: tipo, {', '.join(VARIABLES_BASE)}")
    d["tipo"] = d["tipo"].astype(str).str.strip().str.upper()
    tipos_raros = sorted(set(d["tipo"]) - set(TIPOS))
    if tipos_raros:
        raise ValueError(f"Tipo de producto desconocido: {tipos_raros}. Use L, M o H.")
    return d


def generar_datos_sinteticos(n: int = 10_000, semilla: int = 42) -> pd.DataFrame:
    """Simula datos con las reglas documentadas de AI4I 2020 (respaldo sin internet)."""
    rng = np.random.default_rng(semilla)
    tipo = rng.choice(["L", "M", "H"], size=n, p=[0.5, 0.3, 0.2])
    aire = 300 + np.cumsum(rng.normal(0, 0.05, n))                 # paseo aleatorio suave
    aire = 300 + (aire - aire.mean()) / aire.std() * 2
    proceso = aire + 10 + rng.normal(0, 1, n)
    torque = np.clip(rng.normal(40, 10, n), 3, None)
    # a más torque, menos velocidad (como en AI4I: correlación negativa fuerte)
    rpm = np.clip(1540 - (torque - 40) * 18 + rng.normal(0, 110, n), 1150, 2900)
    desgaste = rng.integers(0, 254, n)                            # minutos de uso de la herramienta

    df = pd.DataFrame({
        "udi": np.arange(1, n + 1),
        "producto_id": [f"{t}{rng.integers(10000, 99999)}" for t in tipo],
        "tipo": tipo,
        "temp_aire_K": aire.round(1),
        "temp_proceso_K": proceso.round(1),
        "velocidad_rpm": rpm.round(0).astype(int),
        "torque_Nm": torque.round(1),
        "desgaste_min": desgaste.astype(int),
    })
    d = agregar_variables(df)
    df["TWF"] = ((d["desgaste_min"] >= 200) & (d["desgaste_min"] <= 240) & (rng.random(n) < 0.03)).astype(int)
    df["HDF"] = ((d["dif_temp_K"] < 8.6) & (d["velocidad_rpm"] < 1380)).astype(int)
    df["PWF"] = ((d["potencia_W"] < 3500) | (d["potencia_W"] > 9000)).astype(int)
    df["OSF"] = (d["esfuerzo_minNm"] > d["tipo"].map(LIMITE_SOBREESFUERZO)).astype(int)
    df["RNF"] = (rng.random(n) < 0.001).astype(int)
    df["falla"] = df[list(MODOS_FALLA)].max(axis=1)
    return df[["udi", "producto_id", "tipo", *VARIABLES_BASE, "falla", *MODOS_FALLA]]


def cargar_datos_ai4i(ruta: str | Path | None = None, descargar: bool = True) -> tuple[pd.DataFrame, str]:
    """Devuelve (datos, origen). Orden: archivo local → descarga de UCI → simulación."""
    ruta = Path(ruta) if ruta else None
    if ruta and ruta.exists():
        return normalizar_columnas(pd.read_csv(ruta, encoding="utf-8-sig")), f"archivo local {ruta.name}"
    if descargar:
        try:
            df = pd.read_csv(URL_AI4I, encoding="utf-8-sig")
            if ruta:
                ruta.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(ruta, index=False)
            return normalizar_columnas(df), "UCI Machine Learning Repository (AI4I 2020)"
        except Exception as e:  # sin internet, firewall, etc.
            print(f"No se pudo descargar AI4I 2020 ({type(e).__name__}); se usan datos simulados.")
    return generar_datos_sinteticos(), "simulación con las reglas de AI4I 2020"


# ----------------------------------------------------------------------------- modelo
class RedNeuronalMantenimiento:
    """Perceptrón multicapa que estima la probabilidad de falla de una máquina.

    Flujo: `preparar()` → `entrenar()` → `optimizar_umbral()` → `evaluar()` / `predecir_maquina()`.
    La decisión de negocio (parar o no la máquina) no usa el umbral 0.5, sino el que minimiza
    el costo esperado con los costos que define la planta.
    """

    def __init__(
        self,
        capas_ocultas: tuple[int, ...] = (32, 16),
        activacion: str = "relu",
        alpha: float = 1e-3,
        tasa_aprendizaje: float = 1e-3,
        max_iter: int = 300,
        variables_derivadas: bool = True,
        balancear: bool = True,
        costo_falla: float = 10_000,
        costo_mantenimiento: float = 600,
        random_state: int = 42,
    ) -> None:
        if costo_falla <= 0 or costo_mantenimiento <= 0:
            raise ValueError("Los costos deben ser positivos.")
        self.capas_ocultas = tuple(capas_ocultas)
        self.activacion = activacion
        self.alpha = alpha
        self.tasa_aprendizaje = tasa_aprendizaje
        self.max_iter = max_iter
        self.variables_derivadas = variables_derivadas
        self.balancear = balancear
        self.costo_falla = float(costo_falla)
        self.costo_mantenimiento = float(costo_mantenimiento)
        self.random_state = random_state

        self.pipeline: Pipeline | None = None
        self.umbral: float = 0.5
        self.origen_datos: str = ""
        self.entrenado_en: str = ""
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.modos_test: pd.DataFrame | None = None

    # -- utilidades
    @property
    def variables(self) -> list[str]:
        return ["tipo", *VARIABLES_BASE, *(VARIABLES_DERIVADAS if self.variables_derivadas else [])]

    @property
    def red(self) -> MLPClassifier:
        self._verificar()
        return self.pipeline.named_steps["red"]

    def _verificar(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Primero entrena el modelo con entrenar() o cárgalo con cargar().")

    def _matriz(self, df: pd.DataFrame) -> pd.DataFrame:
        return agregar_variables(normalizar_columnas(df))[self.variables]

    def _crear_pipeline(self, red: Any) -> Pipeline:
        numericas = [v for v in self.variables if v != "tipo"]
        prep = ColumnTransformer([
            ("num", StandardScaler(), numericas),                         # las redes necesitan escalas parecidas
            ("tipo", OneHotEncoder(categories=[list(TIPOS)], handle_unknown="ignore"), ["tipo"]),
        ])
        return Pipeline([("prep", prep), ("red", red)])

    def crear_red(self) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=self.capas_ocultas,
            activation=self.activacion,
            solver="adam",
            alpha=self.alpha,
            learning_rate_init=self.tasa_aprendizaje,
            batch_size=128,
            max_iter=self.max_iter,
            early_stopping=False,
            n_iter_no_change=25,
            random_state=self.random_state,
        )

    # -- preparación
    def preparar(self, df: pd.DataFrame, origen: str = "", tam_val: float = 0.15,
                 tam_test: float = 0.15) -> dict[str, int]:
        """Divide en entrenamiento / validación / prueba de forma estratificada.

        - Entrenamiento: la red aprende los pesos.
        - Validación: se elige el umbral de decisión (sin tocar la prueba).
        - Prueba: la evaluación final, como si fueran máquinas nuevas.
        """
        d = normalizar_columnas(df)
        if OBJETIVO not in d.columns:
            raise ValueError("Para entrenar se necesita la columna 'falla' (o 'Machine failure').")
        X = agregar_variables(d)[self.variables]
        y = d[OBJETIVO].astype(int)
        modos = d[[m for m in MODOS_FALLA if m in d.columns]]
        X_tmp, self.X_test, y_tmp, self.y_test, _, self.modos_test = train_test_split(
            X, y, modos, test_size=tam_test, stratify=y, random_state=self.random_state)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_tmp, y_tmp, test_size=tam_val / (1 - tam_test), stratify=y_tmp, random_state=self.random_state)
        self.origen_datos = origen
        return {"entrenamiento": len(self.X_train), "validación": len(self.X_val), "prueba": len(self.X_test),
                "fallas_entrenamiento": int(self.y_train.sum()), "fallas_validación": int(self.y_val.sum()),
                "fallas_prueba": int(self.y_test.sum())}

    def _sobremuestrear(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Repite al azar las fallas del entrenamiento hasta que sean ~1/3 de los datos.

        Con 3 % de fallas, una red que diga siempre "no falla" ya acierta el 97 %: el
        sobremuestreo obliga a la red a prestar atención a la clase minoritaria.
        """
        rng = np.random.default_rng(self.random_state)
        idx_falla = np.flatnonzero(y.to_numpy() == 1)
        n_extra = max(0, len(y) // 2 - len(idx_falla))
        extra = rng.choice(idx_falla, size=n_extra, replace=True)
        orden = rng.permutation(np.r_[np.arange(len(y)), extra])
        return X.iloc[orden].reset_index(drop=True), y.iloc[orden].reset_index(drop=True)

    # -- entrenamiento
    def entrenar(self) -> "RedNeuronalMantenimiento":
        if self.X_train is None:
            raise RuntimeError("Primero ejecuta preparar(df).")
        X, y = (self._sobremuestrear(self.X_train, self.y_train) if self.balancear
                else (self.X_train, self.y_train))
        self.pipeline = self._crear_pipeline(self.crear_red())
        with warnings.catch_warnings():               # max_iter es el presupuesto de épocas: se usa completo
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.pipeline.fit(X, y)
        self.entrenado_en = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.optimizar_umbral()
        return self

    def entrenar_referencia(self) -> Pipeline:
        """Regresión logística con las mismas variables: la línea base con la que se compara la red."""
        X, y = (self._sobremuestrear(self.X_train, self.y_train) if self.balancear
                else (self.X_train, self.y_train))
        base = self._crear_pipeline(LogisticRegression(max_iter=2000))
        return base.fit(X, y)

    def curva_perdida(self) -> pd.DataFrame:
        return pd.DataFrame({"época": np.arange(1, len(self.red.loss_curve_) + 1),
                             "pérdida": self.red.loss_curve_})

    def arquitectura(self) -> pd.DataFrame:
        """Neuronas y parámetros (pesos + sesgos) de cada capa."""
        red = self.red
        tam = [red.coefs_[0].shape[0], *self.capas_ocultas, 1]
        nombres = ["Entrada", *[f"Oculta {i}" for i in range(1, len(self.capas_ocultas) + 1)], "Salida"]
        act = ["—", *[self.activacion] * len(self.capas_ocultas), "logistic (sigmoide)"]
        params = [0, *[w.size + b.size for w, b in zip(red.coefs_, red.intercepts_)]]
        return pd.DataFrame({"capa": nombres, "neuronas": tam, "activación": act, "parámetros": params})

    # -- predicción
    def predecir_proba(self, df: pd.DataFrame | None = None, X: pd.DataFrame | None = None) -> np.ndarray:
        self._verificar()
        X = self._matriz(df) if X is None else X
        return self.pipeline.predict_proba(X)[:, 1]

    def _conjunto(self, conjunto: str) -> tuple[pd.DataFrame, pd.Series]:
        return {"entrenamiento": (self.X_train, self.y_train), "validación": (self.X_val, self.y_val),
                "prueba": (self.X_test, self.y_test)}[conjunto]

    # -- costos y umbral
    def costo(self, y: np.ndarray, prob: np.ndarray, umbral: float,
              costo_falla: float | None = None, costo_mantenimiento: float | None = None) -> dict[str, float]:
        """Costo de operar con el modelo frente a no tener modelo.

        - Se para la máquina (predicción = falla): se paga un mantenimiento preventivo.
        - No se para y falla: se paga la parada no planificada (reparación + producción perdida).
        - Supuesto: el mantenimiento preventivo evita la falla.
        """
        cf = self.costo_falla if costo_falla is None else costo_falla
        cm = self.costo_mantenimiento if costo_mantenimiento is None else costo_mantenimiento
        y = np.asarray(y)
        pred = (prob >= umbral).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        con_modelo = float(fn * cf + (tp + fp) * cm)
        sin_modelo = float(y.sum() * cf)
        return {"umbral": float(umbral), "VP": int(tp), "FP": int(fp), "FN": int(fn), "VN": int(tn),
                "paradas": int(tp + fp),
                "costo_con_modelo": con_modelo, "costo_sin_modelo": sin_modelo,
                "ahorro": sin_modelo - con_modelo,
                "ahorro_pct": (sin_modelo - con_modelo) / sin_modelo if sin_modelo else 0.0}

    def curva_costos(self, conjunto: str = "validación", costo_falla: float | None = None,
                     costo_mantenimiento: float | None = None) -> pd.DataFrame:
        X, y = self._conjunto(conjunto)
        prob = self.predecir_proba(X=X)
        umbrales = np.round(np.linspace(0.01, 0.99, 99), 2)
        return pd.DataFrame([self.costo(y, prob, u, costo_falla, costo_mantenimiento) for u in umbrales])

    def umbral_optimo(self, costo_falla: float | None = None, costo_mantenimiento: float | None = None) -> float:
        """Umbral que minimiza el costo en VALIDACIÓN (la prueba queda intacta para evaluar)."""
        curva = self.curva_costos("validación", costo_falla, costo_mantenimiento)
        minimo = curva["costo_con_modelo"].min()
        empatados = curva.loc[curva["costo_con_modelo"] == minimo, "umbral"]
        return float(empatados.median())                                   # el centro de la zona óptima

    def optimizar_umbral(self) -> float:
        self.umbral = self.umbral_optimo()
        return self.umbral

    def con_costos(self, costo_falla: float, costo_mantenimiento: float,
                   umbral: float | None = None) -> "RedNeuronalMantenimiento":
        """Copia que comparte la red entrenada pero usa otros costos (y su umbral óptimo)."""
        nuevo = copy.copy(self)
        nuevo.costo_falla, nuevo.costo_mantenimiento = float(costo_falla), float(costo_mantenimiento)
        nuevo.umbral = nuevo.umbral_optimo() if umbral is None else float(umbral)
        return nuevo

    # -- evaluación
    def metricas(self, y: np.ndarray, prob: np.ndarray, umbral: float | None = None) -> dict[str, float]:
        u = self.umbral if umbral is None else umbral
        pred = (prob >= u).astype(int)
        return {"umbral": u,
                "exactitud": float((pred == np.asarray(y)).mean()),
                "precisión": float(precision_score(y, pred, zero_division=0)),
                "sensibilidad (recall)": float(recall_score(y, pred, zero_division=0)),
                "F1": float(f1_score(y, pred, zero_division=0)),
                "ROC-AUC": float(roc_auc_score(y, prob)),
                "PR-AUC": float(average_precision_score(y, prob))}

    def evaluar(self, conjunto: str = "prueba", umbral: float | None = None) -> dict[str, float]:
        X, y = self._conjunto(conjunto)
        prob = self.predecir_proba(X=X)
        u = self.umbral if umbral is None else umbral
        return {**self.metricas(y, prob, u), **{k: v for k, v in self.costo(y, prob, u).items()
                                                if k in ("costo_con_modelo", "costo_sin_modelo", "ahorro", "ahorro_pct")}}

    def matriz_confusion(self, conjunto: str = "prueba", umbral: float | None = None) -> pd.DataFrame:
        X, y = self._conjunto(conjunto)
        pred = (self.predecir_proba(X=X) >= (self.umbral if umbral is None else umbral)).astype(int)
        m = confusion_matrix(y, pred, labels=[0, 1])
        return pd.DataFrame(m, index=["Real: sin falla", "Real: falla"],
                            columns=["Predicho: sin falla", "Predicho: falla"])

    def curvas_roc_pr(self, conjunto: str = "prueba") -> tuple[pd.DataFrame, pd.DataFrame]:
        X, y = self._conjunto(conjunto)
        prob = self.predecir_proba(X=X)
        fpr, tpr, _ = roc_curve(y, prob)
        prec, rec, _ = precision_recall_curve(y, prob)
        return pd.DataFrame({"fpr": fpr, "tpr": tpr}), pd.DataFrame({"recall": rec, "precision": prec})

    def deteccion_por_modo(self, umbral: float | None = None) -> pd.DataFrame:
        """¿Qué tipo de falla detecta mejor la red? (conjunto de prueba)."""
        if self.modos_test is None or self.modos_test.empty:
            return pd.DataFrame()
        pred = self.predecir_proba(X=self.X_test) >= (self.umbral if umbral is None else umbral)
        filas = []
        for m, nombre in MODOS_FALLA.items():
            if m in self.modos_test:
                mask = self.modos_test[m].to_numpy() == 1
                if mask.sum():
                    filas.append({"modo": m, "descripción": nombre, "casos": int(mask.sum()),
                                  "detectados": int(pred[mask].sum()),
                                  "% detectado": round(100 * pred[mask].mean(), 1)})
        return pd.DataFrame(filas)

    # -- uso en planta
    @staticmethod
    def alertas_fisicas(fila: pd.Series | dict) -> list[str]:
        """Reglas de ingeniería que explican el riesgo (no son la red: ayudan a leer su resultado)."""
        f = agregar_variables(pd.DataFrame([dict(fila)])).iloc[0]
        alertas = []
        if f["desgaste_min"] >= 200:
            alertas.append(f"Herramienta muy desgastada ({f['desgaste_min']:.0f} min ≥ 200): programar cambio.")
        if f["dif_temp_K"] < 8.6 and f["velocidad_rpm"] < 1380:
            alertas.append(f"Disipación de calor insuficiente (ΔT {f['dif_temp_K']:.1f} K < 8.6 y "
                           f"{f['velocidad_rpm']:.0f} rpm < 1380).")
        if f["potencia_W"] < 3500 or f["potencia_W"] > 9000:
            alertas.append(f"Potencia fuera de rango ({f['potencia_W']:,.0f} W; normal 3 500–9 000 W).")
        lim = LIMITE_SOBREESFUERZO.get(str(f["tipo"]).upper(), 11_000)
        if f["esfuerzo_minNm"] > lim:
            alertas.append(f"Sobreesfuerzo ({f['esfuerzo_minNm']:,.0f} min·Nm > {lim:,} para tipo {f['tipo']}).")
        return alertas

    def nivel_riesgo(self, prob: float, umbral: float | None = None) -> str:
        u = self.umbral if umbral is None else umbral
        if prob >= max(u, 0.5):
            return "Crítico"
        if prob >= u:
            return "Alto"
        if prob >= u / 2:
            return "Medio"
        return "Bajo"

    def predecir_maquina(self, tipo: str, temp_aire_K: float, temp_proceso_K: float,
                         velocidad_rpm: float, torque_Nm: float, desgaste_min: float,
                         umbral: float | None = None) -> dict[str, Any]:
        """Diagnóstico de UNA máquina: probabilidad, decisión y explicación."""
        fila = {"tipo": tipo, "temp_aire_K": temp_aire_K, "temp_proceso_K": temp_proceso_K,
                "velocidad_rpm": velocidad_rpm, "torque_Nm": torque_Nm, "desgaste_min": desgaste_min}
        u = self.umbral if umbral is None else umbral
        p = float(self.predecir_proba(pd.DataFrame([fila]))[0])
        parar = p >= u
        return {"probabilidad": p, "umbral": u, "nivel": self.nivel_riesgo(p, u), "parar": parar,
                "decisión": ("Detener y programar mantenimiento preventivo" if parar
                             else "Seguir operando y monitorear"),
                "costo_esperado_seguir": p * self.costo_falla,
                "costo_parar": self.costo_mantenimiento,
                "alertas": self.alertas_fisicas(fila),
                "variables": agregar_variables(pd.DataFrame([fila])).iloc[0].to_dict()}

    def priorizar(self, df: pd.DataFrame, umbral: float | None = None) -> pd.DataFrame:
        """Lista de máquinas ordenada por riesgo: la orden de trabajo del turno."""
        u = self.umbral if umbral is None else umbral
        d = normalizar_columnas(df).reset_index(drop=True)
        prob = self.predecir_proba(d)
        out = d.copy()
        out["prob_falla"] = prob.round(4)
        out["riesgo"] = [self.nivel_riesgo(p, u) for p in prob]
        out["acción"] = np.where(prob >= u, "Mantenimiento preventivo", "Operar")
        out["costo_esperado_si_opera"] = (prob * self.costo_falla).round(0)
        return out.sort_values("prob_falla", ascending=False).reset_index(drop=True)

    def respuesta_negocio(self) -> dict[str, Any]:
        """Responde la pregunta de negocio con las cifras del conjunto de prueba."""
        ev = self.evaluar("prueba")
        c = self.costo(self.y_test, self.predecir_proba(X=self.X_test), self.umbral)
        n = len(self.y_test)
        return {
            "máquinas evaluadas": n,
            "fallas reales": int(self.y_test.sum()),
            "fallas anticipadas": int(c["VP"]),
            "fallas no detectadas": int(c["FN"]),
            "paradas preventivas": int(c["paradas"]),
            "paradas innecesarias": int(c["FP"]),
            "% de fallas anticipadas": ev["sensibilidad (recall)"],
            "% de máquinas a intervenir": c["paradas"] / n,
            "costo sin modelo": c["costo_sin_modelo"],
            "costo con modelo": c["costo_con_modelo"],
            "ahorro": c["ahorro"],
            "ahorro %": c["ahorro_pct"],
            "umbral": self.umbral,
        }

    # -- persistencia
    def guardar(self, ruta: str | Path) -> Path:
        self._verificar()
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        estado = {k: v for k, v in self.__dict__.items()}
        estado["sklearn_version"] = sklearn.__version__
        with open(ruta, "wb") as f:
            pickle.dump(estado, f)
        return ruta

    @classmethod
    def cargar(cls, ruta: str | Path) -> "RedNeuronalMantenimiento":
        with open(ruta, "rb") as f:
            estado = pickle.load(f)
        version = estado.pop("sklearn_version", None)
        if version and version != sklearn.__version__:
            raise RuntimeError(f"Modelo guardado con scikit-learn {version}; aquí hay {sklearn.__version__}.")
        obj = cls.__new__(cls)
        obj.__dict__.update(estado)
        return obj

    @classmethod
    def entrenar_desde_cero(cls, ruta_datos: str | Path | None = None, **kwargs) -> "RedNeuronalMantenimiento":
        """Atajo: carga los datos, prepara y entrena (lo usa la app si no encuentra el modelo)."""
        df, origen = cargar_datos_ai4i(ruta_datos)
        modelo = cls(**kwargs)
        modelo.preparar(df, origen)
        return modelo.entrenar()
