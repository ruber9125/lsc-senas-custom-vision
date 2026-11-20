"""Grabador de video con la cámara del dispositivo (componente web propio, sin paquetes nuevos).

El componente vive en componentes_web/grabador/index.html y devuelve a Python un diccionario con el
video en base64, su tipo (mimeType), la duración real en segundos y un id único de la grabación.
"""
import base64
import binascii
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_RUTA = Path(__file__).resolve().parent.parent / "componentes_web" / "grabador"
_componente = components.declare_component("grabador", path=str(_RUTA))

SEGUNDOS = 3                      # duración de cada grabación
MAX_BYTES = 40 * 1024 * 1024      # límite de seguridad: 3 s de video pesan 1-2 MB


def grabar_con_camara():
    """Muestra el grabador. Devuelve None o {"id", "contenido" (bytes), "nombre", "duracion_s"}."""
    valor = _componente(segundos=SEGUNDOS, cuenta=3, key="grabador_camara", default=None)
    if not isinstance(valor, dict) or not valor.get("id") or not valor.get("video_b64"):
        return None

    # El id de la grabación evita decodificar (y reprocesar) el mismo video en cada recarga de la página.
    guardada = st.session_state.get("_grabacion")
    if guardada and guardada["id"] == valor["id"]:
        return guardada

    try:
        contenido = base64.b64decode(valor["video_b64"], validate=True)
    except (binascii.Error, ValueError):
        st.error("La grabación llegó dañada. Intenta grabar de nuevo.")
        return None
    if not contenido or len(contenido) > MAX_BYTES:
        st.error("La grabación está vacía o es demasiado grande. Intenta grabar de nuevo.")
        return None

    extension = ".mp4" if "mp4" in str(valor.get("mimeType", "")).lower() else ".webm"
    try:
        duracion = float(valor.get("duracion_s") or 0)
    except (TypeError, ValueError):
        duracion = 0.0
    identificador = "".join(c for c in str(valor["id"]) if c.isalnum() or c == "-")[:40]
    grabacion = {"id": valor["id"], "contenido": contenido,
                 "nombre": f"camara_{identificador[:8]}{extension}",
                 "duracion_s": duracion if 0 < duracion <= 60 else None}
    st.session_state["_grabacion"] = grabacion
    return grabacion
