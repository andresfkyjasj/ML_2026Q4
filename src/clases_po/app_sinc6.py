"""App web: mantenimiento predictivo con redes neuronales (Semana 7).

Ejecutar en local (desde la raíz del repositorio):
    streamlit run src/clases_po/app_sinc6.py
    (o también: python src/clases_po/app_sinc6.py, que relanza con Streamlit)

En Streamlit Community Cloud: Main file path = src/clases_po/app_sinc6.py
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


if __name__ == "__main__":
    from streamlit import runtime

    if runtime.exists():                                  # ya se está ejecutando con `streamlit run`
        FrontendRedNeuronal(cargar_modelo()).render()
    else:                                                 # se lanzó con `python app_sinc6.py`: relanzar con Streamlit
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
        sys.exit(stcli.main())
