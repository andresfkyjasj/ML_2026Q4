"""App web: asistente virtual con Transformers (Semana 7).

Ejecutar en local (desde la raíz del repositorio):
    streamlit run src/clases_po/app_sinc7.py

En Streamlit Community Cloud: Main file path = src/clases_po/app_sinc7.py
"""
import sys
from pathlib import Path

import streamlit as st

AQUI = Path(__file__).resolve().parent                    # src/clases_po
SRC = AQUI.parent                                         # src/
sys.path.insert(0, str(AQUI))                             # importa los módulos sueltos

from transformer_intenciones import AsistenteIntenciones, GeneradorFrases, cargar_massive  # noqa: E402
from frontend_transformer import FrontendTransformer                                        # noqa: E402

MODELOS = SRC / "sincronicas" / "modelos"
DATOS = SRC / "sincronicas" / "datos" / "massive_es"


@st.cache_resource(show_spinner="Cargando el Transformer…")
def cargar():
    datos, _ = cargar_massive(DATOS)
    try:
        a = AsistenteIntenciones.cargar(MODELOS / "transformer_intenciones.pt")
        a.datos = datos
        a.ids = {k: [a.tok.codificar(t) for t in df["texto"]] for k, df in datos.items()}
    except Exception as e:                                # no existe o es de otra versión
        print(f"No se usó el modelo guardado ({e}). Se entrena de nuevo.")
        a = AsistenteIntenciones().preparar(datos).entrenar(verbose=False)
    try:
        g = GeneradorFrases.cargar(MODELOS / "generador_frases.pt")
    except Exception:
        g = None
    return a, g


if __name__ == "__main__":
    from streamlit import runtime

    if runtime.exists():
        FrontendTransformer(*cargar()).render()
    else:                                                 # python app_sinc7.py → relanza con Streamlit
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
        sys.exit(stcli.main())
