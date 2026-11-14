"""App web: mantenimiento predictivo con redes neuronales (Semana 7).

Ejecutar en local (desde la raíz del repositorio):
    streamlit run src/clases_po/app_mantenimiento.py

En Streamlit Community Cloud: Main file path = src/clases_po/app_mantenimiento.py
"""
import sys
from pathlib import Path

import streamlit as st

AQUI = Path(__file__).resolve().parent                    # src/clases_po
SRC = AQUI.parent                                         # src/
sys.path.insert(0, str(AQUI))                             # importa los módulos sin cargar todo el paquete

from red_neuronal import RedNeuronalMantenimiento         # noqa: E402
from frontend_red_neuronal import FrontendRedNeuronal     # noqa: E402

RUTA_MODELO = SRC / "sincronicas" / "modelos" / "red_neuronal_mantenimiento.pkl"
RUTA_DATOS = SRC / "sincronicas" / "datos" / "ai4i2020.csv"


@st.cache_resource(show_spinner="Cargando la red neuronal…")
def cargar_modelo() -> RedNeuronalMantenimiento:
    try:
        return RedNeuronalMantenimiento.cargar(RUTA_MODELO)
    except Exception as e:                                # no existe o es de otra versión de scikit-learn
        print(f"No se usó el modelo guardado ({e}). Se entrena de nuevo.")
        return RedNeuronalMantenimiento.entrenar_desde_cero(RUTA_DATOS)


FrontendRedNeuronal(cargar_modelo()).render()
