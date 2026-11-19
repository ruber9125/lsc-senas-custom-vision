"""Gráficos interactivos (Plotly) con la paleta institucional."""
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from .estilos import AZUL, BORDE, COLORES, TEXTO, TEXTO_SUAVE

CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"],
          "toImageButtonOptions": {"format": "png", "scale": 2}}

_ESCALA_AZUL = [[0.0, "#F3F6F9"], [0.35, "#A9C2D6"], [0.7, "#2C6FA6"], [1.0, AZUL]]

# Fuente del sistema (no la fuente web de Streamlit): Plotly mide el texto para calcular márgenes,
# leyenda y etiquetas. Si la fuente aún no ha cargado, mide con otra y el texto queda cortado o encimado.
FUENTE = 'Arial, "Helvetica Neue", Helvetica, sans-serif'
# Marcas de eje invisibles: solo sirven para separar las etiquetas del gráfico.
_SEPARACION = dict(ticks="outside", ticklen=8, tickcolor="rgba(0,0,0,0)", automargin=True)
# Misma altura para las dos gráficas: así los dos recuadros del tablero quedan simétricos.
ALTO = 540


def _estilo_base(fig, titulo):
    fig.update_layout(
        title=dict(text=titulo, x=0.01, xanchor="left", y=0.98, yanchor="top",
                   font=dict(family=FUENTE, size=15, color=AZUL)),
        font=dict(family=FUENTE, size=13, color=TEXTO),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        # Márgenes fijos suficientes para etiquetas y títulos de los ejes; automargin solo agrega si falta.
        margin=dict(l=80, r=20, t=50, b=90), height=ALTO, autosize=True,
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=BORDE, font=dict(family=FUENTE, color=TEXTO)),
    )
    return fig


def matriz_confusion(cm, clases, titulo):
    cm = np.asarray(cm)
    tope = max(int(cm.max()), 1)
    fila_total = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, fila_total, out=np.zeros_like(cm, dtype=float), where=fila_total > 0) * 100
    fig = go.Figure(go.Heatmap(
        z=cm, x=clases, y=clases, zmin=0, zmax=tope, colorscale=_ESCALA_AZUL, showscale=False,
        customdata=pct, xgap=3, ygap=3,
        hovertemplate="Seña real: <b>%{y}</b><br>Predicción: <b>%{x}</b><br>"
                      "Videos: <b>%{z}</b> (%{customdata:.0f} % de la fila)<extra></extra>"))
    for i in range(len(clases)):
        for j in range(len(clases)):
            fig.add_annotation(x=clases[j], y=clases[i], text=f"<b>{int(cm[i, j])}</b>", showarrow=False,
                               font=dict(size=20, color="#FFFFFF" if cm[i, j] > tope * 0.5 else TEXTO))
    _estilo_base(fig, titulo)
    ejes = dict(showgrid=False, zeroline=False, tickfont=dict(size=14, color=TEXTO), type="category",
                tickmode="array", tickvals=list(clases), ticktext=list(clases), **_SEPARACION)
    fig.update_xaxes(title=dict(text="Predicción del modelo", standoff=10), side="bottom", **ejes)
    fig.update_yaxes(title=dict(text="Seña real", standoff=10), autorange="reversed", **ejes)
    return fig


def curvas_roc(reales, probs, clases):
    """Curvas ROC una contra el resto. Devuelve (figura, {clase: AUC}) o (None, {}) si aún no se puede."""
    y = label_binarize(reales, classes=clases)
    probs = np.asarray(probs)
    fig = go.Figure()
    aucs = {}
    for k, c in enumerate(clases):
        if 0 < y[:, k].sum() < len(y):            # la curva ROC necesita positivos y negativos de esa seña
            fpr, tpr, _ = roc_curve(y[:, k], probs[:, k])
            aucs[c] = auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{c} · AUC {aucs[c]:.3f}",
                line=dict(color=COLORES[c], width=3, shape="hv"),
                hovertemplate=f"<b>{c}</b><br>Falsos positivos: %{{x:.2f}}<br>"
                              "Verdaderos positivos: %{y:.2f}<extra></extra>"))
    if not aucs:
        return None, {}
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar · AUC 0.5",
                             line=dict(color=TEXTO_SUAVE, width=1.5, dash="dash"), hoverinfo="skip"))
    _estilo_base(fig, "Curvas ROC (una contra el resto)")
    ejes = dict(range=[-0.02, 1.02], tick0=0, dtick=0.2, tickformat=".1f", gridcolor="#EEF2F6",
                zeroline=False, tickfont=dict(size=12, color=TEXTO), **_SEPARACION)
    fig.update_xaxes(title=dict(text="Tasa de falsos positivos", standoff=8), **ejes)
    fig.update_yaxes(title=dict(text="Tasa de verdaderos positivos", standoff=8), **ejes)
    # Leyenda debajo del gráfico, una entrada por línea: así cabe completa a cualquier ancho
    # (en horizontal las entradas se encimaban y la última quedaba cortada en el borde derecho).
    fig.update_layout(height=ALTO, margin=dict(l=80, r=20, t=50, b=165),
                      legend=dict(orientation="v", x=0, xanchor="left", y=-0.17, yanchor="top",
                                  tracegroupgap=2, bgcolor="rgba(255,255,255,0)",
                                  font=dict(family=FUENTE, size=12)))
    return fig, aucs
