"""Frontend en Streamlit para probar y validar la red neuronal de mantenimiento predictivo (Semana 7).

Los métodos `figura_*()` devuelven figuras de Plotly: se usan igual en el notebook (vista previa)
y en la app web (`render()`). Streamlit y Plotly se importan solo cuando se necesitan, así el
paquete `clases_po` se puede importar aunque no estén instalados.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:                                    # dentro del paquete clases_po (notebook)
    from .red_neuronal import (MODOS_FALLA, RANGOS, TIPOS, VARIABLES_BASE,
                               RedNeuronalMantenimiento, agregar_variables)
except ImportError:                     # como módulo suelto (app de Streamlit)
    from red_neuronal import (MODOS_FALLA, RANGOS, TIPOS, VARIABLES_BASE,  # type: ignore
                              RedNeuronalMantenimiento, agregar_variables)

COLOR = {"ok": "#2a9d8f", "falla": "#e63946", "neutro": "#8d99ae", "acento": "#eb6834",
         "texto": "#1f1f1e", "medio": "#f4a261"}
COLOR_RIESGO = {"Bajo": "#2a9d8f", "Medio": "#e9c46a", "Alto": "#f4a261", "Crítico": "#e63946"}

ETIQUETAS = {
    "temp_aire_K": "Temperatura del aire [K]",
    "temp_proceso_K": "Temperatura del proceso [K]",
    "velocidad_rpm": "Velocidad de rotación [rpm]",
    "torque_Nm": "Torque [Nm]",
    "desgaste_min": "Desgaste de la herramienta [min]",
    "dif_temp_K": "Diferencia de temperatura [K]",
    "potencia_W": "Potencia [W]",
    "esfuerzo_minNm": "Esfuerzo (desgaste × torque) [min·Nm]",
}

# Escenarios para probar la red con un clic (cada uno activa un modo de falla distinto)
ESCENARIOS = {
    "Operación normal": dict(tipo="M", temp_aire_K=300.0, temp_proceso_K=310.0, velocidad_rpm=1500,
                             torque_Nm=40.0, desgaste_min=60),
    "Herramienta gastada + torque alto (OSF)": dict(tipo="L", temp_aire_K=300.0, temp_proceso_K=310.0,
                                                    velocidad_rpm=1350, torque_Nm=58.0, desgaste_min=215),
    "Mala disipación de calor (HDF)": dict(tipo="M", temp_aire_K=303.0, temp_proceso_K=310.5,
                                           velocidad_rpm=1330, torque_Nm=50.0, desgaste_min=90),
    "Potencia baja (PWF)": dict(tipo="L", temp_aire_K=299.0, temp_proceso_K=309.0, velocidad_rpm=2750,
                                torque_Nm=9.0, desgaste_min=100),
    "Potencia excesiva (PWF)": dict(tipo="M", temp_aire_K=299.0, temp_proceso_K=309.0, velocidad_rpm=1250,
                                    torque_Nm=72.0, desgaste_min=40),
}

PREGUNTA = ("¿Qué máquinas debemos detener para mantenimiento preventivo en el próximo turno, "
            "de modo que anticipemos la mayor cantidad de fallas al menor costo total?")


def _go():
    import plotly.graph_objects as go
    return go


def _layout(fig, titulo: str, alto: int = 380):
    fig.update_layout(title=dict(text=titulo, x=0, font=dict(size=16)), height=alto,
                      margin=dict(l=10, r=10, t=50, b=10), template="plotly_white",
                      font=dict(color=COLOR["texto"]), legend=dict(orientation="h", y=-0.18))
    return fig


def _dinero(x: float) -> str:
    return f"US$ {x:,.0f}"


class FrontendRedNeuronal:
    """Arma la app web que prueba, valida y explica la red neuronal."""

    def __init__(self, modelo: RedNeuronalMantenimiento, titulo: str = "Mantenimiento predictivo con redes neuronales"):
        if modelo.pipeline is None:
            raise ValueError("El modelo debe estar entrenado.")
        self.modelo = modelo
        self.titulo = titulo

    # ================================================================ datos de apoyo
    def datos_prueba(self) -> pd.DataFrame:
        """El conjunto de prueba en formato 'crudo' (como lo enviaría la planta)."""
        m = self.modelo
        d = m.X_test[["tipo", *VARIABLES_BASE]].copy()
        d.insert(0, "maquina_id", [f"CNC-{i:04d}" for i in m.X_test.index])
        d["falla"] = m.y_test.to_numpy()
        return d.reset_index(drop=True)

    def plantilla_csv(self, n: int = 20) -> bytes:
        return self.datos_prueba().drop(columns="falla").head(n).to_csv(index=False).encode("utf-8")

    # ================================================================ figuras: datos
    @staticmethod
    def figura_balance(df: pd.DataFrame):
        go = _go()
        c = df["falla"].value_counts().reindex([0, 1], fill_value=0)
        fig = go.Figure(go.Bar(x=["Sin falla", "Falla"], y=c.values, marker_color=[COLOR["ok"], COLOR["falla"]],
                               text=[f"{v:,} ({v / c.sum():.1%})" for v in c.values], textposition="outside"))
        fig.update_yaxes(title="Registros")
        return _layout(fig, "Clases muy desbalanceadas: las fallas son pocas")

    @staticmethod
    def figura_modos(df: pd.DataFrame):
        go = _go()
        modos = [m for m in MODOS_FALLA if m in df.columns]
        c = df[modos].sum().sort_values()
        fig = go.Figure(go.Bar(y=[f"{MODOS_FALLA[m]} ({m})" for m in c.index], x=c.values, orientation="h",
                               marker_color=COLOR["falla"], text=c.values, textposition="outside"))
        fig.update_xaxes(title="Casos")
        return _layout(fig, "¿Por qué fallan las máquinas? (modos de falla)", 320)

    @staticmethod
    def figura_dispersion(df: pd.DataFrame, x: str = "velocidad_rpm", y: str = "torque_Nm"):
        go = _go()
        fig = go.Figure()
        for v, nombre, color, op in [(0, "Sin falla", COLOR["neutro"], 0.25), (1, "Falla", COLOR["falla"], 0.9)]:
            s = df[df["falla"] == v]
            fig.add_trace(go.Scattergl(x=s[x], y=s[y], mode="markers", name=nombre,
                                       marker=dict(color=color, size=5 if v == 0 else 7, opacity=op)))
        fig.update_xaxes(title=ETIQUETAS.get(x, x))
        fig.update_yaxes(title=ETIQUETAS.get(y, y))
        return _layout(fig, "Las fallas aparecen en los extremos: la frontera NO es una línea recta", 430)

    # ================================================================ figuras: la red
    @staticmethod
    def figura_activaciones():
        go = _go()
        z = np.linspace(-5, 5, 200)
        fig = go.Figure()
        for nombre, val, color in [("Sigmoide 1/(1+e^-z)", 1 / (1 + np.exp(-z)), COLOR["acento"]),
                                   ("Tanh", np.tanh(z), COLOR["neutro"]),
                                   ("ReLU max(0, z)", np.maximum(0, z), COLOR["ok"])]:
            fig.add_trace(go.Scatter(x=z, y=val, name=nombre, line=dict(width=3, color=color)))
        fig.update_xaxes(title="z = w·x + b (entrada de la neurona)", zeroline=True)
        fig.update_yaxes(title="salida de la neurona", range=[-1.2, 3])
        return _layout(fig, "Funciones de activación: lo que hace NO lineal a la red")

    def figura_arquitectura(self, max_neuronas: int = 10):
        """Dibujo de la red: nodos por capa y conexiones coloreadas por el signo del peso."""
        go = _go()
        red = self.modelo.red
        tam = [red.coefs_[0].shape[0], *self.modelo.capas_ocultas, 1]
        nombres = ["Entrada", *[f"Oculta {i}" for i in range(1, len(tam) - 1)], "Salida"]
        pos = []
        for c, n in enumerate(tam):
            k = min(n, max_neuronas)
            ys = np.linspace(-1, 1, k) if k > 1 else np.array([0.0])
            pos.append([(c, y) for y in ys])
        fig = go.Figure()
        for c in range(len(tam) - 1):
            W = red.coefs_[c]
            for i, (x0, y0) in enumerate(pos[c]):
                for j, (x1, y1) in enumerate(pos[c + 1]):
                    w = W[i, j]
                    fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip",
                                             showlegend=False,
                                             line=dict(width=0.4 + 2 * min(abs(w), 1.5),
                                                       color=COLOR["ok"] if w > 0 else COLOR["falla"]),
                                             opacity=0.35))
        for c, capa in enumerate(pos):
            fig.add_trace(go.Scatter(x=[p[0] for p in capa], y=[p[1] for p in capa], mode="markers",
                                     marker=dict(size=16, color="white", line=dict(width=2, color=COLOR["texto"])),
                                     showlegend=False, hovertext=f"{nombres[c]}: {tam[c]} neuronas", hoverinfo="text"))
            extra = f"<br>(se dibujan {max_neuronas})" if tam[c] > max_neuronas else ""
            fig.add_annotation(x=c, y=1.25, text=f"<b>{nombres[c]}</b><br>{tam[c]} neuronas{extra}",
                               showarrow=False, font=dict(size=12))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, range=[-1.2, 1.5])
        return _layout(fig, "Arquitectura: verde = peso positivo, rojo = peso negativo", 430)

    def figura_curva_perdida(self):
        go = _go()
        c = self.modelo.curva_perdida()
        fig = go.Figure(go.Scatter(x=c["época"], y=c["pérdida"], line=dict(color=COLOR["acento"], width=3)))
        fig.update_xaxes(title="Época (pasadas completas por los datos)")
        fig.update_yaxes(title="Pérdida (log-loss)")
        return _layout(fig, "Curva de aprendizaje: la pérdida baja mientras la red ajusta sus pesos")

    # ================================================================ figuras: evaluación
    def figura_costos(self, modelo: RedNeuronalMantenimiento | None = None):
        go = _go()
        m = modelo or self.modelo
        c = m.curva_costos("validación")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c["umbral"], y=c["costo_con_modelo"], name="Costo con la red",
                                 line=dict(color=COLOR["acento"], width=3)))
        fig.add_trace(go.Scatter(x=c["umbral"], y=c["costo_sin_modelo"], name="Costo sin modelo (todas fallan)",
                                 line=dict(color=COLOR["neutro"], dash="dash")))
        fig.add_vline(x=m.umbral, line=dict(color=COLOR["texto"], dash="dot"),
                      annotation_text=f"umbral elegido = {m.umbral:.2f}", annotation_position="top")
        fig.add_vline(x=0.5, line=dict(color=COLOR["neutro"], dash="dot", width=1),
                      annotation_text="0.5 (por defecto)", annotation_position="bottom right")
        fig.update_xaxes(title="Umbral de probabilidad para detener la máquina")
        fig.update_yaxes(title="Costo total (US$)", tickformat=",.0f")
        return _layout(fig, "¿Desde qué probabilidad conviene parar? (validación)")

    def figura_matriz(self, modelo: RedNeuronalMantenimiento | None = None, matriz: pd.DataFrame | None = None,
                      titulo: str = "Matriz de confusión (prueba)"):
        go = _go()
        m = modelo or self.modelo
        mc = m.matriz_confusion("prueba") if matriz is None else matriz
        etiquetas = [["Sin falla y operó ✓", "Parada innecesaria (FP)"],
                     ["Falla NO anticipada (FN)", "Falla anticipada ✓ (VP)"]]
        texto = [[f"{etiquetas[i][j]}<br><b>{mc.iat[i, j]:,}</b>" for j in range(2)] for i in range(2)]
        fig = go.Figure(go.Heatmap(z=[[0, 1], [2, 0]], x=["Predicho: sin falla", "Predicho: falla"],
                                   y=["Real: sin falla", "Real: falla"], text=texto, texttemplate="%{text}",
                                   colorscale=[[0, "#d8f3dc"], [0.5, "#ffe8d6"], [1, "#ffccd5"]],
                                   showscale=False, hoverinfo="skip"))
        fig.update_yaxes(autorange="reversed")
        return _layout(fig, titulo, 360)

    def figura_roc_pr(self):
        go = _go()
        from plotly.subplots import make_subplots
        roc, pr = self.modelo.curvas_roc_pr("prueba")
        ev = self.modelo.evaluar("prueba")
        fig = make_subplots(cols=2, subplot_titles=[f"ROC (AUC = {ev['ROC-AUC']:.3f})",
                                                    f"Precisión-Recall (AP = {ev['PR-AUC']:.3f})"])
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], line=dict(color=COLOR["acento"], width=3),
                                 name="Red neuronal"), 1, 1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=COLOR["neutro"], dash="dash"),
                                 name="Azar"), 1, 1)
        fig.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], line=dict(color=COLOR["ok"], width=3),
                                 name="Red neuronal (PR)"), 1, 2)
        base = self.modelo.y_test.mean()
        fig.add_trace(go.Scatter(x=[0, 1], y=[base, base], line=dict(color=COLOR["neutro"], dash="dash"),
                                 name=f"Azar ({base:.1%})"), 1, 2)
        fig.update_xaxes(title_text="Falsos positivos", row=1, col=1)
        fig.update_yaxes(title_text="Fallas detectadas", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precisión", row=1, col=2)
        return _layout(fig, "", 380)

    def figura_deteccion_modos(self, modelo: RedNeuronalMantenimiento | None = None):
        go = _go()
        d = (modelo or self.modelo).deteccion_por_modo()
        if d.empty:
            return None
        fig = go.Figure(go.Bar(x=d["descripción"], y=d["% detectado"], marker_color=COLOR["acento"],
                               text=[f"{p:.0f}% ({a}/{c})" for p, a, c in zip(d["% detectado"], d["detectados"], d["casos"])],
                               textposition="outside"))
        fig.update_yaxes(title="% de fallas anticipadas", range=[0, 115])
        return _layout(fig, "¿Qué tipo de falla anticipa la red? (prueba)", 360)

    @staticmethod
    def figura_medidor(prob: float, umbral: float):
        go = _go()
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number=dict(suffix=" %", valueformat=".1f"),
            gauge=dict(axis=dict(range=[0, 100]),
                       bar=dict(color=COLOR["falla"] if prob >= umbral else COLOR["ok"]),
                       steps=[dict(range=[0, umbral * 50], color="#e9f5f2"),
                              dict(range=[umbral * 50, umbral * 100], color="#fdf3dc"),
                              dict(range=[umbral * 100, 100], color="#fde2e4")],
                       threshold=dict(line=dict(color=COLOR["texto"], width=4), value=umbral * 100))))
        return _layout(fig, "Probabilidad de falla (la línea negra es el umbral)", 280)

    @staticmethod
    def figura_ranking(tabla: pd.DataFrame, top: int = 15):
        go = _go()
        t = tabla.head(top).iloc[::-1]
        etiqueta = t["maquina_id"] if "maquina_id" in t else t.index.astype(str)
        fig = go.Figure(go.Bar(y=etiqueta, x=t["prob_falla"] * 100, orientation="h",
                               marker_color=[COLOR_RIESGO[r] for r in t["riesgo"]],
                               text=[f"{p:.0%} · {r}" for p, r in zip(t["prob_falla"], t["riesgo"])],
                               textposition="outside"))
        fig.update_xaxes(title="Probabilidad de falla (%)", range=[0, 115])
        return _layout(fig, f"Top {min(top, len(tabla))} máquinas por riesgo", 60 + 28 * min(top, len(tabla)) + 80)

    # ================================================================ texto de negocio
    @staticmethod
    def texto_respuesta(r: dict[str, Any]) -> str:
        return (f"Con el umbral **{r['umbral']:.2f}**, la red anticipa **{r['fallas anticipadas']} de "
                f"{r['fallas reales']}** fallas (**{r['% de fallas anticipadas']:.0%}**) interviniendo solo el "
                f"**{r['% de máquinas a intervenir']:.1%}** de las máquinas ({r['paradas preventivas']} paradas, "
                f"{r['paradas innecesarias']} de ellas innecesarias). El costo pasa de **{_dinero(r['costo sin modelo'])}** "
                f"a **{_dinero(r['costo con modelo'])}**: un ahorro de **{_dinero(r['ahorro'])} "
                f"({r['ahorro %']:.0%})**. **Decisión:** cada turno, detener las máquinas cuya probabilidad "
                f"supere el umbral, empezando por las de riesgo *Crítico*.")

    # ================================================================ app
    def render(self) -> None:
        import streamlit as st

        st.set_page_config(page_title="Mantenimiento predictivo · Red neuronal", page_icon="🛠️", layout="wide")
        m0 = self.modelo

        # ---------- barra lateral: costos del negocio
        with st.sidebar:
            st.header("⚙️ Supuestos de negocio")
            cf = st.number_input("Costo de una falla no anticipada (US$)", 1_000, 200_000,
                                 int(m0.costo_falla), step=1_000,
                                 help="Reparación correctiva + producción perdida + piezas dañadas.")
            cm = st.number_input("Costo de un mantenimiento preventivo (US$)", 50, 20_000,
                                 int(m0.costo_mantenimiento), step=50,
                                 help="Técnico + repuestos + tiempo de máquina detenida planificado.")
            modo = st.radio("Umbral de decisión", ["Óptimo según costos", "Manual"], horizontal=True)
            umbral_manual = None
            if modo == "Manual":
                umbral_manual = st.slider("Umbral", 0.01, 0.99, float(m0.umbral), 0.01)
            m = m0.con_costos(cf, cm, umbral_manual)
            st.metric("Umbral en uso", f"{m.umbral:.2f}")
            st.caption(f"Relación de costos falla : mantenimiento = **{cf / cm:.0f} : 1**. "
                       "Mientras más cara la falla, más bajo el umbral (se para antes).")
            st.divider()
            st.caption(f"Datos: {m0.origen_datos}  \nEntrenado: {m0.entrenado_en}  \n"
                       f"Red: {m0.activacion}, capas {m0.capas_ocultas}")

        st.title(f"🛠️ {self.titulo}")
        st.markdown(f"**Pregunta de negocio:** {PREGUNTA}")

        r = m.respuesta_negocio()
        k = st.columns(5)
        k[0].metric("Fallas anticipadas", f"{r['% de fallas anticipadas']:.0%}",
                    f"{r['fallas anticipadas']} de {r['fallas reales']}")
        k[1].metric("Máquinas a intervenir", f"{r['% de máquinas a intervenir']:.1%}",
                    f"{r['paradas preventivas']} paradas", delta_color="off")
        k[2].metric("Costo sin modelo", _dinero(r["costo sin modelo"]))
        k[3].metric("Costo con la red", _dinero(r["costo con modelo"]))
        k[4].metric("Ahorro", _dinero(r["ahorro"]), f"{r['ahorro %']:.0%}")

        tabs = st.tabs(["🎯 Respuesta de negocio", "🔧 Probar una máquina", "📋 Validar un lote",
                        "📈 Desempeño del modelo", "🧠 Cómo funciona la red"])

        # ---------- 1. respuesta
        with tabs[0]:
            st.subheader("Respuesta")
            st.markdown(self.texto_respuesta(r))
            c1, c2 = st.columns(2)
            c1.plotly_chart(self.figura_costos(m), width="stretch")
            c2.plotly_chart(self.figura_matriz(m), width="stretch")
            st.info("**Cómo leerlo:** con 3 % de fallas, un modelo que diga siempre *'no falla'* tendría 97 % de "
                    "exactitud y no serviría para nada. Por eso se mide cuántas fallas se **anticipan** (recall) y "
                    "cuánto **dinero** se ahorra, y el umbral se elige por costo, no en 0.5.")
            fig = self.figura_deteccion_modos(m)
            if fig is not None:
                st.plotly_chart(fig, width="stretch")

        # ---------- 2. probar una máquina
        with tabs[1]:
            st.subheader("Diagnóstico de una máquina")
            esc = st.selectbox("Cargar un escenario de ejemplo", list(ESCENARIOS), key="escenario")
            base = ESCENARIOS[esc]
            with st.form("maquina"):
                c1, c2, c3 = st.columns(3)
                tipo = c1.selectbox("Tipo de producto", list(TIPOS), index=list(TIPOS).index(base["tipo"]),
                                    format_func=TIPOS.get, key=f"tipo_{esc}")
                valores = {}
                for i, var in enumerate(VARIABLES_BASE):
                    lo, hi, _ = RANGOS[var]
                    col = [c1, c2, c3][(i + 1) % 3]
                    entero = var in ("velocidad_rpm", "desgaste_min")
                    valores[var] = col.slider(ETIQUETAS[var], int(lo) if entero else float(lo),
                                              int(hi) if entero else float(hi),
                                              int(base[var]) if entero else float(base[var]),
                                              1 if entero else 0.1, key=f"{var}_{esc}")
                st.form_submit_button("Diagnosticar", type="primary", width="stretch")

            d = m.predecir_maquina(tipo=tipo, **valores)        # se recalcula en cada envío del formulario
            c1, c2 = st.columns([1, 1])
            c1.plotly_chart(self.figura_medidor(d["probabilidad"], d["umbral"]), width="stretch")
            with c2:
                color = COLOR_RIESGO[d["nivel"]]
                st.markdown(f"### Riesgo: <span style='color:{color}'>{d['nivel']}</span>", unsafe_allow_html=True)
                (st.error if d["parar"] else st.success)(f"**{d['decisión']}**")
                st.markdown(f"- Costo esperado si sigue operando: **{_dinero(d['costo_esperado_seguir'])}** "
                            f"(probabilidad × costo de falla)  \n"
                            f"- Costo de parar ahora: **{_dinero(d['costo_parar'])}**")
                v = d["variables"]
                st.caption(f"Variables que calcula la clase: ΔT = {v['dif_temp_K']:.1f} K · "
                           f"potencia = {v['potencia_W']:,.0f} W · esfuerzo = {v['esfuerzo_minNm']:,.0f} min·Nm")
            st.markdown("**Señales de ingeniería** (reglas físicas, ayudan a explicar el resultado de la red):")
            if d["alertas"]:
                for a in d["alertas"]:
                    st.warning(a)
            else:
                st.write("✅ Ninguna variable está fuera de los rangos de riesgo conocidos.")

        # ---------- 3. validar un lote
        with tabs[2]:
            st.subheader("Validar la red con un lote de máquinas")
            st.markdown("Sube un CSV con las columnas `tipo`, " + ", ".join(f"`{v}`" for v in VARIABLES_BASE) +
                        " (o con los nombres originales de AI4I 2020). Si incluye `falla` (0/1), la app "
                        "compara la predicción con la realidad.")
            c1, c2 = st.columns([2, 1])
            archivo = c1.file_uploader("CSV de máquinas", type="csv")
            c2.download_button("⬇️ Descargar plantilla", self.plantilla_csv(), "plantilla_maquinas.csv", "text/csv")
            try:
                lote = pd.read_csv(archivo) if archivo is not None else self.datos_prueba()
                fuente = archivo.name if archivo is not None else "conjunto de prueba (máquinas que la red nunca vio)"
                tabla = m.priorizar(lote)
            except Exception as e:
                st.error(f"No se pudo procesar el archivo: {e}")
                tabla = None
            if tabla is not None:
                st.caption(f"Lote: {fuente} · {len(tabla):,} máquinas")
                n_parar = int((tabla["acción"] == "Mantenimiento preventivo").sum())
                a, b, c = st.columns(3)
                a.metric("Máquinas a intervenir", f"{n_parar:,}", f"{n_parar / len(tabla):.1%} del lote",
                         delta_color="off")
                b.metric("Riesgo crítico", f"{int((tabla['riesgo'] == 'Crítico').sum()):,}")
                c.metric("Costo esperado si nadie interviene", _dinero(tabla["costo_esperado_si_opera"].sum()))
                if "falla" in tabla.columns:
                    y, p = tabla["falla"].astype(int).to_numpy(), tabla["prob_falla"].to_numpy()
                    if 0 < y.sum() < len(y):
                        met = m.metricas(y, p)
                        cos = m.costo(y, p, m.umbral)
                        st.markdown("**Validación contra la realidad del lote**")
                        x = st.columns(4)
                        x[0].metric("Fallas anticipadas (recall)", f"{met['sensibilidad (recall)']:.0%}")
                        x[1].metric("Precisión de las alertas", f"{met['precisión']:.0%}")
                        x[2].metric("ROC-AUC", f"{met['ROC-AUC']:.3f}")
                        x[3].metric("Ahorro en el lote", _dinero(cos["ahorro"]), f"{cos['ahorro_pct']:.0%}")
                        mc = pd.DataFrame([[cos["VN"], cos["FP"]], [cos["FN"], cos["VP"]]])
                        st.plotly_chart(self.figura_matriz(m, mc, "Matriz de confusión del lote"),
                                        width="stretch")
                st.plotly_chart(self.figura_ranking(tabla), width="stretch")
                solo_parar = st.checkbox("Mostrar solo las máquinas a intervenir", value=True)
                vista = tabla[tabla["acción"] == "Mantenimiento preventivo"] if solo_parar else tabla
                st.dataframe(vista, width="stretch", hide_index=True,
                             column_config={"prob_falla": st.column_config.ProgressColumn(
                                 "Prob. falla", min_value=0.0, max_value=1.0, format="%.2f")})
                st.download_button("⬇️ Descargar orden de trabajo (CSV)", tabla.to_csv(index=False).encode("utf-8"),
                                   "orden_trabajo_mantenimiento.csv", "text/csv")

        # ---------- 4. desempeño
        with tabs[3]:
            st.subheader("¿Qué tan buena es la red?")
            filas = {c: m.evaluar(c) for c in ("entrenamiento", "validación", "prueba")}
            met = pd.DataFrame(filas).T[["exactitud", "precisión", "sensibilidad (recall)", "F1", "ROC-AUC", "PR-AUC"]]
            st.dataframe(met.round(3), width="stretch")
            st.caption("Si entrenamiento es mucho mejor que prueba, la red memorizó (sobreajuste).")
            st.plotly_chart(self.figura_roc_pr(), width="stretch")
            st.plotly_chart(self.figura_curva_perdida(), width="stretch")

        # ---------- 5. cómo funciona
        with tabs[4]:
            st.subheader("La red neuronal por dentro")
            st.markdown(
                "1. **Entrada:** las 8 variables numéricas se **estandarizan** y el tipo de producto se codifica "
                "*one-hot* (3 columnas).\n"
                "2. **Capas ocultas:** cada neurona calcula `z = w·x + b` y aplica **ReLU**. Combinando muchas "
                "neuronas la red dibuja fronteras curvas, que una sola recta no puede.\n"
                "3. **Salida:** una neurona con **sigmoide** convierte el resultado en una probabilidad entre 0 y 1.\n"
                "4. **Aprendizaje:** el optimizador **Adam** ajusta los pesos por *backpropagation* para "
                "reducir la pérdida (log-loss), época tras época.\n"
                "5. **Decisión:** la probabilidad se compara con el **umbral de costo** de la barra lateral.")
            st.plotly_chart(self.figura_arquitectura(), width="stretch")
            c1, c2 = st.columns([1, 1])
            c1.dataframe(m.arquitectura(), width="stretch", hide_index=True)
            c2.plotly_chart(self.figura_activaciones(), width="stretch")
