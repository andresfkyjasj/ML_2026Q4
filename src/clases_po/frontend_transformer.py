"""Frontend en Streamlit para el asistente de intenciones con Transformers (Semana 7).

Las figuras (Plotly) se usan igual en el notebook y en la app. El frontend no recalcula
nada por su cuenta: todo se lo pide a `AsistenteIntenciones` y a `GeneradorFrases`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:                                    # dentro del paquete clases_po (notebook)
    from .transformer_intenciones import AsistenteIntenciones, GeneradorFrases, FUERA_DE_DOMINIO
except ImportError:                     # como módulo suelto (app de Streamlit)
    from transformer_intenciones import AsistenteIntenciones, GeneradorFrases, FUERA_DE_DOMINIO  # type: ignore

PREGUNTA = ("¿Qué pide cada usuario, para ejecutar la acción correcta al primer intento, "
            "y cuándo conviene pasarlo a un humano?")
COLOR = {"ok": "#2a9d8f", "mal": "#e63946", "humano": "#f4a261", "acento": "#6c3fd1", "neutro": "#8d99ae"}


def _go():
    import plotly.graph_objects as go
    return go


def _layout(fig, titulo: str, alto: int = 380):
    fig.update_layout(title=dict(text=titulo, x=0, font=dict(size=16)), height=alto, template="plotly_white",
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
    return fig


def pesos(x: float) -> str:
    return f"COP {x:,.0f}".replace(",", ".")


def pesos_corto(x: float) -> str:
    """Formato compacto para las tarjetas de indicadores."""
    return f"COP {x / 1e6:,.2f} M".replace(".", ",") if abs(x) >= 1e6 else pesos(x)


class FrontendTransformer:
    def __init__(self, asistente: AsistenteIntenciones, generador: GeneradorFrases | None = None,
                 titulo: str = "Asistente virtual con Transformers"):
        self.a, self.g, self.titulo = asistente, generador, titulo

    # ============================================================ figuras
    @staticmethod
    def figura_tokens(piezas: list[str], ids: list[int]):
        go = _go()
        colores = [COLOR["humano"] if p.startswith("##") else (COLOR["mal"] if p == "[UNK]" else COLOR["acento"]) for p in piezas]
        fig = go.Figure(go.Bar(x=list(range(len(piezas))), y=[1] * len(piezas), marker_color=colores,
                               text=[f"{p}<br><b>{i}</b>" for p, i in zip(piezas, ids)], textposition="inside",
                               hovertemplate="%{text}<extra></extra>"))
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        return _layout(fig, "Tokens (morado = inicio de palabra, naranja = continuación ##, rojo = desconocido) e ids", 200)

    @staticmethod
    def figura_atencion(A: np.ndarray, piezas: list[str], titulo: str):
        go = _go()
        et = [f"{p} ·{i}" for i, p in enumerate(piezas)]          # etiquetas únicas (puede haber tokens repetidos)
        fig = go.Figure(go.Heatmap(z=A, x=et, y=et, colorscale="Purples", zmin=0, zmax=1,
                                   text=np.round(A, 2), texttemplate="%{text}", showscale=False))
        fig.update_yaxes(autorange="reversed", title="token que pregunta (Q)")
        fig.update_xaxes(title="token que responde (K)")
        return _layout(fig, titulo, 360)

    @staticmethod
    def figura_top(top: list[tuple[str, float]], umbral: float):
        go = _go()
        fig = go.Figure(go.Bar(y=[t for t, _ in top][::-1], x=[p for _, p in top][::-1], orientation="h",
                               marker_color=[COLOR["ok"] if p >= umbral else COLOR["neutro"] for _, p in top][::-1],
                               text=[f"{p:.1%}" for _, p in top][::-1], textposition="outside"))
        fig.add_vline(x=umbral, line=dict(dash="dot"), annotation_text=f"umbral {umbral:.2f}")
        fig.update_xaxes(range=[0, 1.1], title="probabilidad (softmax)")
        return _layout(fig, "Las 5 intenciones más probables", 300)

    def figura_costos(self, ce: float, ch: float):
        go = _go()
        c = self.a.curva_costos("validación", costo_error=ce, costo_humano=ch)
        u = float(c.loc[c["costo"].idxmin(), "umbral"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c["umbral"], y=c["costo"], name="Costo con el asistente", line=dict(color=COLOR["acento"], width=3)))
        fig.add_trace(go.Scatter(x=c["umbral"], y=c["costo todo humano"], name="Todo con humanos", line=dict(color=COLOR["neutro"], dash="dash")))
        fig.add_vline(x=u, line=dict(dash="dot"), annotation_text=f"umbral = {u:.2f}")
        fig.update_xaxes(title="Umbral de confianza para ejecutar sin humano")
        fig.update_yaxes(title="Costo total (COP)", tickformat=",.0f")
        return _layout(fig, "¿Desde qué confianza conviene ejecutar? (validación)")

    def figura_reparto(self, r: dict):
        go = _go()
        vals = [r["aciertos automáticos"], r["errores"], r["a humano"]]
        fig = go.Figure(go.Pie(labels=["Ejecuta y acierta", "Ejecuta y se equivoca", "Pasa a un humano"], values=vals, hole=.55,
                               marker_colors=[COLOR["ok"], COLOR["mal"], COLOR["humano"]], sort=False))
        return _layout(fig, "¿Qué pasa con cada frase? (prueba)", 340)

    def figura_aprendizaje(self):
        go = _go()
        h = pd.DataFrame(self.a.historial)
        if h.empty:
            return None
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h["época"], y=h["pérdida"], name="pérdida (entrenamiento)", line=dict(color=COLOR["acento"], width=3)))
        fig.add_trace(go.Scatter(x=h["época"], y=h["exactitud validación"], name="exactitud (validación)", yaxis="y2", line=dict(color=COLOR["ok"], width=3)))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 1]))
        return _layout(fig, "Curva de aprendizaje")

    # ============================================================ app
    def render(self) -> None:
        import streamlit as st
        st.set_page_config(page_title="Asistente con Transformers", page_icon="🤖", layout="wide")
        a = self.a
        with st.sidebar:
            st.header("⚙️ Supuestos de negocio (COP)")
            ce = st.number_input("Costo de una acción equivocada", 1_000, 500_000, int(a.costo_error), step=1_000,
                                 help="El usuario reclama, se reprocesa y queda molesto.")
            ch = st.number_input("Costo de pasar a un humano", 500, 100_000, int(a.costo_humano), step=500,
                                 help="Tiempo de un agente del centro de contacto.")
            modo = st.radio("Umbral", ["Óptimo según costos", "Manual"], horizontal=True)
            if modo == "Manual":
                umbral = st.slider("Umbral de confianza", 0.0, 0.99, float(a.umbral), 0.01)
            else:
                c = a.curva_costos("validación", costo_error=ce, costo_humano=ch)
                umbral = float(c.loc[c["costo"].idxmin(), "umbral"])
            st.metric("Umbral en uso", f"{umbral:.2f}")
            st.caption(f"Modelo: Transformer de {a.modelo.config['bloques']} bloques, d = {a.modelo.config['d']}, "
                       f"{a.modelo.config['cabezas']} cabezas · vocabulario de {len(a.tok.vocab):,} tokens")

        st.title(f"🤖 {self.titulo}")
        st.markdown(f"**Pregunta de negocio:** {PREGUNTA}")
        r = a.costos("prueba", umbral, costo_error=ce, costo_humano=ch)
        r["ahorro"] = r["costo todo humano"] - r["costo"]
        k = st.columns(5)
        k[0].metric("Resueltas solas", f"{r['% automáticas']:.0%}", f"{r['automáticas']:,} frases", delta_color="off")
        k[1].metric("Pasan a un humano", f"{r['% a humano']:.0%}", f"{r['a humano']:,} frases", delta_color="off")
        k[2].metric("Errores al ejecutar", f"{r['% errores']:.1%}", f"{r['errores']} frases", delta_color="inverse")
        k[3].metric("Costo con el asistente", pesos_corto(r["costo"]), f"todo humano: {pesos_corto(r['costo todo humano'])}", delta_color="off")
        k[4].metric("Ahorro", pesos_corto(r["ahorro"]), f"{r['ahorro'] / r['costo todo humano']:.0%}")

        tabs = st.tabs(["🎯 Respuesta de negocio", "🔬 Dentro del Transformer", "📋 Validar un lote",
                        "⚠️ Alucinación", "✍️ Siguiente palabra", "📈 Desempeño"])
        with tabs[0]:
            c1, c2 = st.columns(2)
            c1.plotly_chart(self.figura_costos(ce, ch), width="stretch")
            c2.plotly_chart(self.figura_reparto(r), width="stretch")
            st.info(f"**Respuesta:** con el umbral {umbral:.2f}, el asistente resuelve solo el **{r['% automáticas']:.0%}** "
                    f"de las frases, pasa a un humano el **{r['% a humano']:.0%}** y se equivoca en el **{r['% errores']:.1%}**. "
                    f"Cuesta **{pesos(r['costo'])}** frente a **{pesos(r['costo todo humano'])}** si todo lo atendiera un humano.")

        with tabs[1]:
            texto = st.text_input("Escribe lo que le dirías al asistente", "pon algo de shakira", key="frase")
            if texto.strip():
                e = a.explicar(texto)
                st.plotly_chart(self.figura_tokens(e["piezas"], e["ids"]), width="stretch")
                b = st.radio("Bloque", list(range(1, len(e["atencion"]) + 1)), horizontal=True, key="bloque")
                cols = st.columns(e["atencion"][0].shape[0])
                for h, col in enumerate(cols):
                    col.plotly_chart(self.figura_atencion(e["atencion"][b - 1][h], e["piezas"], f"Bloque {b} · cabeza {h + 1}"), width="stretch")
                c1, c2 = st.columns([2, 1])
                c1.plotly_chart(self.figura_top(e["top"], umbral), width="stretch")
                with c2:
                    (st.success if e["confianza"] >= umbral else st.warning)(
                        f"**{'Ejecutar' if e['confianza'] >= umbral else 'Pasar a un humano'}** · confianza {e['confianza']:.1%}")
                    st.caption("Cada fila del mapa de atención suma 1: dice cuánto «mira» cada token a los demás.")

        with tabs[2]:
            st.markdown("Sube un CSV con la columna `texto` (y opcionalmente `intencion`, para comparar con la realidad).")
            archivo = st.file_uploader("CSV de frases", type="csv")
            lote = pd.read_csv(archivo) if archivo is not None else pd.DataFrame({"texto": list(a.datos["prueba"]["texto"][:200]),
                                                                                  "intencion": list(a.datos["prueba"]["intencion"][:200])}) if hasattr(a, "datos") else None
            if lote is not None and "texto" in lote:
                p = a.predecir_proba(list(lote["texto"].astype(str)))
                lote = lote.assign(prediccion=a.catalogo.intencion.to_numpy()[p.argmax(1)], confianza=p.max(1).round(3),
                                   decision=np.where(p.max(1) >= umbral, "Ejecutar", "Pasar a un humano"))
                if "intencion" in lote:
                    lote["¿acierta?"] = lote["prediccion"] == lote["intencion"]
                    st.metric("Exactitud en el lote", f"{lote['¿acierta?'].mean():.1%}")
                st.dataframe(lote, width="stretch", hide_index=True)
                st.download_button("⬇️ Descargar resultados", lote.to_csv(index=False).encode("utf-8"), "resultados.csv", "text/csv")

        with tabs[3]:
            st.markdown("El clasificador **siempre** elige una de las 60 intenciones, aunque la frase no tenga nada que ver. "
                        "El umbral es lo que evita que esa respuesta inventada se ejecute.")
            st.dataframe(a.fuera_de_dominio(), width="stretch", hide_index=True)

        with tabs[4]:
            if self.g is None:
                st.info("No hay generador cargado.")
            else:
                ini = st.text_input("Inicio de la frase", "pon una alarma a las", key="inicio")
                temp = st.slider("Temperatura", 0.1, 2.0, 1.0, 0.1, help="Baja: elige lo más probable. Alta: se arriesga.")
                c1, c2 = st.columns(2)
                c1.dataframe(self.g.siguiente(ini, 8, temp), width="stretch", hide_index=True)
                c2.write("**Tres frases generadas** (un token, otro y otro más):")
                for s in range(3):
                    c2.write(f"- {self.g.generar(ini, temperatura=temp, semilla=s)}")
                st.caption("Elige el token más plausible, no el verdadero: así nace una alucinación.")

        with tabs[5]:
            st.dataframe(a.modelo.n_parametros(), width="stretch", hide_index=True)
            fig = self.figura_aprendizaje()
            if fig is not None:
                st.plotly_chart(fig, width="stretch")
