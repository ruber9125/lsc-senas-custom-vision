"""Componentes visuales en HTML/CSS (sin imágenes externas: la única imagen es el logo institucional)."""
import base64
import io
from html import escape
from pathlib import Path

import streamlit as st

from .estilos import COLORES, CSS

LOGO = Path(__file__).resolve().parent.parent / "assets" / "logo_lasalle.png"
FLUJO = Path(__file__).resolve().parent.parent / "assets" / "flujo_lsc.png"
PASOS = ["Video", "Puntos clave", "Trayectoria", "Modelo en la nube", "Resultado"]


def aplicar_estilos():
    st.html(CSS)


@st.cache_resource(show_spinner=False)
def _logo_base64(ancho: int = 720) -> str:
    """Logo reducido (el original mide 3840 px) y codificado para incrustarlo en el encabezado."""
    from PIL import Image
    with Image.open(LOGO) as im:
        alto = round(im.height * ancho / im.width)
        buf = io.BytesIO()
        im.resize((ancho, alto), Image.LANCZOS).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encabezado():
    logo = f'<img src="data:image/png;base64,{_logo_base64()}" alt="Universidad de La Salle">' if LOGO.exists() else ""
    st.html(f"""
<div class="lsc-encabezado">
  {logo}
  <div class="lsc-sep"></div>
  <div>
    <h1>Reconocimiento de Lengua de Señas Colombiana</h1>
    <p class="lsc-sub">Visión por computador y un modelo en la nube que identifica señas a partir de un video.</p>
    <div class="lsc-linea"></div>
    <p class="lsc-contexto">Señas reconocidas: <b>HOLA</b> · <b>BIEN</b> · <b>MAL</b></p>
  </div>
</div>""")


def indicador_pasos(contenedor, actual: int):
    """actual = índice (0-4) del paso en curso; 5 = todos completos."""
    items = []
    for i, nombre in enumerate(PASOS):
        clase = "hecho" if i < actual else "actual" if i == actual else ""
        marca = "✓" if i < actual else str(i + 1)
        items.append(f'<div class="lsc-paso {clase}"><div class="lsc-circulo">{marca}</div>'
                     f'<div class="lsc-nombre">{nombre}</div></div>')
    contenedor.html(f'<div class="lsc-pasos">{"".join(items)}</div>')


def seccion(titulo: str, nota: str = ""):
    st.html(f'<div class="lsc-seccion">{titulo}</div>' + (f'<p class="lsc-nota">{nota}</p>' if nota else ""))


def etiqueta(texto: str):
    st.html(f'<div class="lsc-etiqueta">{texto}</div>')


def tarjeta_instrucciones():
    st.html("""
<div class="lsc-tarjeta suave">
  <h4>Cómo grabar el video</h4>
  <ul>
    <li>Duración de <b>2 a 4 segundos</b> con <b>una sola seña</b>.</li>
    <li>De frente, sentado, de la cintura hacia arriba y con buena luz.</li>
    <li>Empieza y termina con las manos abajo.</li>
    <li>Formatos: <b>.mp4</b>, <b>.mov</b>, <b>.avi</b> o puntos clave <b>.npz</b>.</li>
  </ul>
</div>""")


def _nivel_confianza(p: float):
    if p >= 0.8:
        return "alta", "Alta"
    if p >= 0.5:
        return "media", "Media"
    return "baja", "Baja"


def tarjeta_resultado(ganadora: str, probs: dict, clases: list, descripcion: str):
    color = COLORES.get(ganadora, "#003151")
    p = probs.get(ganadora, 0.0)
    nivel, texto_nivel = _nivel_confianza(p)
    barras = []
    for c in sorted(clases, key=lambda c: -probs.get(c, 0.0)):
        v = probs.get(c, 0.0) * 100
        atenuada = "" if c == ganadora else " atenuada"
        barras.append(
            f'<div class="lsc-barra{atenuada}"><span class="lsc-clase">{c}</span>'
            f'<div class="lsc-pista"><div class="lsc-relleno" style="width:{v:.1f}%;background:{COLORES.get(c, "#888")}"></div></div>'
            f'<span class="lsc-valor">{v:.1f} %</span></div>')
    # El marco de la tarjeta lo pone el contenedor (estilos: st-key-tarjeta_resultado), que se estira
    # hasta la altura de las tarjetas vecinas (video e imagen de trayectoria).
    with st.container(key="tarjeta_resultado", height="stretch"):
        st.html(f"""
<div class="lsc-resultado">
  <span class="lsc-chip">Seña reconocida</span>
  <div class="lsc-sena" style="color:{color}">{escape(ganadora)}</div>
  <div class="lsc-confianza">Confianza <b>{p * 100:.1f} %</b><span class="lsc-nivel {nivel}">{texto_nivel}</span></div>
  <p class="lsc-desc">{escape(descripcion)}</p>
  <div class="lsc-etiqueta">Probabilidad por seña</div>
  {"".join(barras)}
</div>""")


@st.cache_resource(show_spinner=False)
def _degradado_turbo() -> str:
    """Mismo mapa de color (turbo) que usa lsc/trayectoria.py para indicar el tiempo."""
    from matplotlib import colormaps
    from matplotlib.colors import to_hex
    cmap = colormaps["turbo"]
    return ", ".join(to_hex(cmap(i / 8)) for i in range(9))


def leyenda_trayectoria(frames: int):
    st.html(f"""
<div class="lsc-leyenda" style="flex-direction:column;gap:.25rem;max-width:224px">
  <div style="height:8px;border-radius:999px;background:linear-gradient(90deg, {_degradado_turbo()})"></div>
  <div style="display:flex;justify-content:space-between"><span>inicio</span><span><b>{frames}</b> fotogramas</span><span>final</span></div>
</div>""")


def veredicto(real: str, prediccion: str):
    acierto = real == prediccion
    clase, icono, titulo = ("acierto", "✓", "Acierto") if acierto else ("error", "✕", "Error")
    st.html(f"""
<div class="lsc-veredicto {clase}">
  <div class="lsc-icono">{icono}</div>
  <div>
    <div class="lsc-titulo">{titulo}</div>
    <div class="lsc-detalle">Seña real: <b>{escape(real)}</b> · El modelo reconoció: <b>{escape(prediccion)}</b></div>
  </div>
</div>""")


def kpis(items):
    """items: lista de (nombre, valor, ayuda, destacado)."""
    tarjetas = "".join(
        f'<div class="lsc-kpi{" destacado" if dest else ""}"><div class="lsc-kpi-nombre">{n}</div>'
        f'<div class="lsc-kpi-valor">{v}</div><div class="lsc-kpi-ayuda">{a}</div></div>'
        for n, v, a, dest in items)
    st.html(f'<div class="lsc-kpis">{tarjetas}</div>')


def estado_vacio(titulo: str, texto: str):
    st.html(f'<div class="lsc-vacio"><b>{titulo}</b>{texto}</div>')


def estado_conexion(conectado: bool):
    if conectado:
        st.html('<div class="lsc-estado ok"><span class="lsc-led"></span>Conectado a Azure Custom Vision</div>')
    else:
        st.html('<div class="lsc-estado mal"><span class="lsc-led"></span>Azure sin configurar</div>')


def titulo_lateral(texto: str):
    st.html(f'<div class="lsc-lateral-titulo">{texto}</div>')


def contadores(videos: int, aciertos: int):
    st.html(f'<div class="lsc-contadores">'
            f'<div class="lsc-contador"><b>{videos}</b><span>Videos confirmados</span></div>'
            f'<div class="lsc-contador"><b>{aciertos}</b><span>Aciertos</span></div></div>')


def tarjeta_proyecto():
    st.html("""
<div class="lsc-proyecto">
  <div class="lsc-proyecto-etiqueta">Proyecto</div>
  <div class="lsc-proyecto-nombre">Reconocimiento LSC</div>
  <div class="lsc-proyecto-sep"></div>
  <div class="lsc-proyecto-autor">Ing. Rubert Eduardo Quintero Orozco</div>
  <div class="lsc-proyecto-inst">Maestría en IA · U. de La Salle</div>
</div>""")


def banner_flujo():
    """Ilustración general del proyecto (aportada por el autor) al inicio de "Cómo funciona"."""
    if not FLUJO.exists():
        return
    with st.container(key="banner_flujo"):
        st.image(str(FLUJO), width="stretch")
        st.html('<p class="lsc-banner-pie">Visión general del proyecto: del video a la seña reconocida.</p>')


def tarjeta_privacidad():
    st.html("""
<div class="lsc-privacidad">
  <div class="lsc-privacidad-titulo"><span class="lsc-candado">✓</span>Datos protegidos</div>
  <p>A Azure solo se envía la <b>imagen de trayectoria</b>, nunca el video.</p>
  <p>Se usa la opción <b>nostore</b>: Azure no guarda ninguna copia.</p>
</div>""")


def pie_pagina():
    st.html("""
<div class="lsc-pie">
  <div class="lsc-linea"></div>
  <b>Ing. Rubert Eduardo Quintero Orozco</b><br>
  <b>Maestría en Inteligencia Artificial · Universidad de La Salle</b><br>
  Visión por Computador · Actividad 6 – Proyecto final
</div>""")
