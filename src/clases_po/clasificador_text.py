"""Flujo reutilizable para entrenar y evaluar un clasificador de texto.

Cada línea del código mínimo se convierte en un método:

    load_dataset   -> cargar_datos
    make_pipeline  -> construir_modelo
    fit            -> entrenar
    score          -> evaluar
    predict        -> predecir

y la clase agrega lo que el código mínimo no tenía: preparar y validar los
datos, dividir de forma estratificada, guardar el modelo con fecha y ejecutar
todo el flujo en orden.
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class ClasificadorTexto:
    """Clasifica textos (por ejemplo, correos en normal, spam o tóxico).

    Args:
        modelo: clasificador de scikit-learn sin entrenar. Por defecto,
            ``MultinomialNB()``. Se copia con ``clone`` para que el mismo
            objeto pueda reutilizarse en varios experimentos.
        columna_texto: columna con el texto (X). Si no existe pero el
            DataFrame trae ``asunto`` y ``mensaje``, se crea uniéndolas.
        columna_objetivo: columna con la etiqueta (y).
        test_size: proporción reservada para prueba, entre 0 y 1.
        random_state: semilla para repetir la misma partición.
        ngram_range: ``(1, 1)`` usa palabras sueltas; ``(1, 2)`` también pares.
        max_features: tamaño máximo del vocabulario (``None`` = sin límite).
        directorio_modelos: carpeta donde se guardan los ``.pkl``.
    """

    def __init__(
        self,
        modelo: Any | None = None,
        columna_texto: str = "texto",
        columna_objetivo: str = "etiqueta",
        test_size: float = 0.2,
        random_state: int = 42,
        ngram_range: tuple[int, int] = (1, 1),
        max_features: int | None = None,
        directorio_modelos: str | Path = "modelos",
    ) -> None:
        if not 0 < test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1, sin incluirlos.")
        if columna_texto == columna_objetivo:
            raise ValueError("columna_texto y columna_objetivo deben ser diferentes.")
        self.modelo = modelo if modelo is not None else MultinomialNB()
        self.columna_texto = columna_texto
        self.columna_objetivo = columna_objetivo
        self.test_size = test_size
        self.random_state = random_state
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.directorio_modelos = Path(directorio_modelos)

        self.pipeline: Pipeline | None = None
        self.ruta_modelo: Path | None = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.predicciones: np.ndarray | None = None
        self.metricas: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # 1. cargar
    # ------------------------------------------------------------------ #
    @staticmethod
    def cargar_datos(fuente: str | Path | pd.DataFrame) -> pd.DataFrame:
        """Devuelve un DataFrame desde un DataFrame, un .csv, un .json o Hugging Face.

        - ``pd.DataFrame``: se devuelve una copia.
        - ``"archivo.csv"`` / ``"archivo.json"``: se lee del disco
          (por ejemplo, la salida de ``FormateadorEnron``).
        - Cualquier otro texto se trata como nombre de un dataset de
          Hugging Face (por ejemplo, ``"SetFit/enron_spam"``); se unen todas
          sus particiones. Requiere ``pip install datasets``.
        """
        if isinstance(fuente, pd.DataFrame):
            return fuente.copy()
        ruta = str(fuente)
        if ruta.lower().endswith(".csv"):
            return pd.read_csv(ruta, keep_default_na=False)
        if ruta.lower().endswith(".json"):
            return pd.read_json(ruta)
        from datasets import load_dataset  # import local: solo si se usa

        ds = load_dataset(ruta)
        return pd.concat([ds[p].to_pandas() for p in ds], ignore_index=True)

    # ------------------------------------------------------------------ #
    # preparar y dividir (lo que agrega la clase)
    # ------------------------------------------------------------------ #
    def preparar_datos(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Valida el DataFrame y devuelve X (textos) e y (etiquetas).

        - Si falta ``columna_texto`` y existen ``asunto`` y ``mensaje``,
          crea ``texto = asunto + " " + mensaje``.
        - Convierte textos y etiquetas a ``str`` y elimina filas con texto
          o etiqueta vacíos.
        - Exige al menos 2 clases y 2 ejemplos por clase (para estratificar).
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un pandas.DataFrame.")
        datos = df.copy()
        if self.columna_texto not in datos.columns and {"asunto", "mensaje"} <= set(datos.columns):
            datos[self.columna_texto] = (
                datos["asunto"].fillna("").astype(str) + " " + datos["mensaje"].fillna("").astype(str)
            )
        faltantes = {self.columna_texto, self.columna_objetivo} - set(datos.columns)
        if faltantes:
            raise KeyError(f"Faltan columnas requeridas: {sorted(faltantes)}.")

        datos = datos[[self.columna_texto, self.columna_objetivo]]
        datos = datos.fillna("").astype(str)
        datos[self.columna_texto] = datos[self.columna_texto].str.strip()
        datos[self.columna_objetivo] = datos[self.columna_objetivo].str.strip()
        filas_iniciales = len(datos)
        datos = datos[(datos[self.columna_texto] != "") & (datos[self.columna_objetivo] != "")]

        conteo = datos[self.columna_objetivo].value_counts()
        if len(conteo) < 2:
            raise ValueError("Se necesitan al menos dos clases distintas.")
        if (conteo < 2).any():
            raise ValueError(f"Cada clase necesita al menos 2 ejemplos: {conteo.to_dict()}.")

        X = datos[self.columna_texto].reset_index(drop=True)
        y = datos[self.columna_objetivo].reset_index(drop=True)
        print(
            f"[preparar_datos] {len(X)} textos válidos; "
            f"{filas_iniciales - len(X)} filas eliminadas; clases: {conteo.to_dict()}."
        )
        return X, y

    def dividir_datos(self, X: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Divide en entrenamiento y prueba conservando la proporción de cada clase."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        print(
            f"[dividir_datos] stratify=y; entrenamiento={len(X_train)}; prueba={len(X_test)}; "
            f"random_state={self.random_state}."
        )
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------ #
    # 2. construir
    # ------------------------------------------------------------------ #
    def construir_modelo(self) -> Pipeline:
        """Encadena TF-IDF (texto -> números) y el clasificador (números -> clase)."""
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=self.ngram_range, max_features=self.max_features)),
                ("modelo", clone(self.modelo)),
            ]
        )
        print(f"[construir_modelo] TfidfVectorizer{self.ngram_range} -> {type(self.modelo).__name__}")
        return self.pipeline

    # ------------------------------------------------------------------ #
    # 3. entrenar
    # ------------------------------------------------------------------ #
    def entrenar(self, X_train: pd.Series, y_train: pd.Series) -> dict[str, Any]:
        """Ajusta el pipeline y lo guarda como .pkl con fecha y hora.

        TF-IDF aprende el vocabulario solo con X_train (fit); a los datos de
        prueba solo los transformará al predecir: no se espía el examen.
        """
        if self.pipeline is None:
            self.construir_modelo()
        self.pipeline.fit(X_train, y_train)

        self.directorio_modelos.mkdir(parents=True, exist_ok=True)
        fecha = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        nombre = type(self.modelo).__name__
        self.ruta_modelo = self.directorio_modelos / f"{fecha}_{nombre}.pkl"
        with self.ruta_modelo.open("wb") as archivo:
            pickle.dump(self.pipeline, archivo)
        print(f"[entrenar] {len(X_train)} textos; modelo guardado en {self.ruta_modelo}")
        return {"pipeline": self.pipeline, "ruta_modelo": self.ruta_modelo, "fecha_modelo": fecha}

    def cargar_modelo(self, ruta_modelo: str | Path) -> Pipeline:
        """Carga un .pkl guardado por ``entrenar`` para volver a usarlo."""
        ruta = Path(ruta_modelo)
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el modelo: {ruta}.")
        with ruta.open("rb") as archivo:
            self.pipeline = pickle.load(archivo)
        self.ruta_modelo = ruta
        return self.pipeline

    # ------------------------------------------------------------------ #
    # 5. predecir
    # ------------------------------------------------------------------ #
    @staticmethod
    def _como_lista(textos: str | Iterable[str]) -> list[str]:
        if isinstance(textos, str):
            return [textos]
        return [str(t) for t in textos]

    def _modelo_listo(self) -> Pipeline:
        if self.pipeline is None or not hasattr(self.pipeline, "classes_"):
            raise RuntimeError("Primero entrena el modelo (entrenar o ejecutar) o cárgalo (cargar_modelo).")
        return self.pipeline

    def predecir_proba(self, textos: str | Iterable[str]) -> pd.DataFrame:
        """Probabilidad de cada clase por texto (una columna por clase)."""
        pipe = self._modelo_listo()
        if not hasattr(pipe, "predict_proba"):
            raise TypeError(f"{type(self.modelo).__name__} no entrega probabilidades.")
        lista = self._como_lista(textos)
        return pd.DataFrame(pipe.predict_proba(lista), columns=pipe.classes_)

    def predecir(
        self,
        textos: str | Iterable[str],
        umbral: float | None = None,
        clase_umbral: str = "toxico",
    ) -> np.ndarray:
        """Devuelve la clase de cada texto.

        Sin ``umbral``: gana la clase más probable.
        Con ``umbral``: se predice ``clase_umbral`` si su probabilidad es
        ``>= umbral``; si no, gana la más probable entre las demás clases.
        """
        pipe = self._modelo_listo()
        lista = self._como_lista(textos)
        if umbral is None:
            return pipe.predict(lista)
        if not 0 <= umbral <= 1:
            raise ValueError("umbral debe estar entre 0 y 1.")
        proba = self.predecir_proba(lista)
        if clase_umbral not in proba.columns:
            raise ValueError(f"'{clase_umbral}' no es una clase del modelo: {list(proba.columns)}.")
        otras = proba.drop(columns=[clase_umbral])
        return np.where(proba[clase_umbral] >= umbral, clase_umbral, otras.idxmax(axis=1)).astype(object)

    # ------------------------------------------------------------------ #
    # 4. evaluar
    # ------------------------------------------------------------------ #
    def evaluar(
        self,
        X_test: pd.Series,
        y_test: pd.Series,
        umbral: float | None = None,
        clase_umbral: str = "toxico",
    ) -> dict[str, Any]:
        """Calcula exactitud, precisión, recall y F1 por clase, macro y ponderado, y la matriz de confusión."""
        pred = self.predecir(X_test, umbral=umbral, clase_umbral=clase_umbral)
        self.predicciones = pred
        clases = list(self._modelo_listo().classes_)
        p, r, f, n = precision_recall_fscore_support(y_test, pred, labels=clases, zero_division=0)
        por_clase = pd.DataFrame({"precision": p, "recall": r, "f1": f, "casos": n}, index=clases)
        matriz = pd.DataFrame(
            confusion_matrix(y_test, pred, labels=clases),
            index=[f"real {c}" for c in clases],
            columns=[f"pred {c}" for c in clases],
        )
        self.metricas = {
            "exactitud": float(accuracy_score(y_test, pred)),
            "f1_macro": float(por_clase["f1"].mean()),
            "f1_ponderado": float(np.average(por_clase["f1"], weights=por_clase["casos"])),
            "por_clase": por_clase,
            "matriz": matriz,
            "umbral": umbral,
        }
        print(f"[evaluar] exactitud={self.metricas['exactitud']:.4f}; "
              f"F1 macro={self.metricas['f1_macro']:.4f}; F1 ponderado={self.metricas['f1_ponderado']:.4f}")
        print(classification_report(y_test, pred, labels=clases, zero_division=0, digits=4))
        print(matriz, "\n")
        return self.metricas

    # ------------------------------------------------------------------ #
    # todo el flujo
    # ------------------------------------------------------------------ #
    def ejecutar(self, fuente: str | Path | pd.DataFrame) -> dict[str, Any]:
        """Ejecuta en orden: cargar, preparar, dividir, construir, entrenar y evaluar."""
        print(f"[ejecutar][1/6] cargar_datos ({type(self.modelo).__name__})")
        df = self.cargar_datos(fuente)
        print("[ejecutar][2/6] preparar_datos")
        X, y = self.preparar_datos(df)
        print("[ejecutar][3/6] dividir_datos")
        self.X_train, self.X_test, self.y_train, self.y_test = self.dividir_datos(X, y)
        print("[ejecutar][4/6] construir_modelo")
        self.construir_modelo()
        print("[ejecutar][5/6] entrenar")
        entrenamiento = self.entrenar(self.X_train, self.y_train)
        print("[ejecutar][6/6] evaluar")
        metricas = self.evaluar(self.X_test, self.y_test)
        print("[ejecutar] flujo completo terminado.")
        return {
            "ruta_modelo": entrenamiento["ruta_modelo"],
            "metricas": metricas,
            "X_test": self.X_test,
            "y_test": self.y_test,
            "predicciones": self.predicciones,
        }