"""
Reconocimiento de señas de la Lengua de Señas Colombiana (LSC) — HOLA, BIEN, MAL.
Actividad 6 · Proyecto Final · Visión por Computador · Maestría en IA, Universidad de La Salle.

Flujo:  video -> MediaPipe Holistic -> imagen de trayectoria (224x224)
        -> modelo publicado en Azure Custom Vision (endpoint REST) -> seña reconocida.

Cada video se sube UNA vez. El modelo reconoce la seña sin pistas; DESPUÉS la persona confirma cuál era
realmente, y con eso la app actualiza la precisión, la matriz de confusión y las curvas ROC de la sesión.
"""
import hashlib
import hmac
import io
import os
import tempfile
import time

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from lsc.azure_cv import clasificar
from lsc.trayectoria import desde_npz, dibujar_trayectoria
from interfaz import componentes as ui
from interfaz.graficos import CONFIG as CONFIG_GRAFICOS, curvas_roc, matriz_confusion

CLASES = ["HOLA", "BIEN", "MAL"]
DESCRIPCION = {"HOLA": "Saludo: la mano abierta sube a la frente y sale hacia afuera.",
               "BIEN": "Pulgar hacia arriba frente al pecho.",
               "MAL": "Pulgar hacia abajo frente al pecho."}

st.set_page_config(page_title="Reconocimiento LSC · Universidad de La Salle",
                   page_icon=":material/sign_language:", layout="wide")
ui.aplicar_estilos()


# ---------- Acceso (opcional): si en los secrets existe APP_PASSWORD, se pide antes de mostrar nada ----------
def acceso_permitido():
    try:
        esperada = str(st.secrets.get("APP_PASSWORD", "") or "")
    except Exception:
        esperada = ""
    if not esperada or st.session_state.get("acceso_ok"):
        return True
    ui.encabezado()
    _, centro, _ = st.columns([1, 1.2, 1])
    with centro:
        with st.form("acceso", border=True):
            ui.etiqueta("Acceso restringido")
            escrita = st.text_input("Contraseña de acceso", type="password")
            entrar = st.form_submit_button("Entrar", type="primary", width="stretch")
        if entrar:
            if hmac.compare_digest(escrita.encode("utf-8"), esperada.encode("utf-8")):
                st.session_state["acceso_ok"] = True
                st.rerun()
            st.error("Contraseña incorrecta.")
    ui.pie_pagina()
    return False


if not acceso_permitido():
    st.stop()


# ---------- Configuración (la clave nunca va en el código) ----------
def credenciales():
    try:
        return st.secrets["AZURE_PREDICTION_URL"], st.secrets["AZURE_PREDICTION_KEY"]
    except Exception:
        return None, None


URL, CLAVE = credenciales()
if "historial" not in st.session_state:
    st.session_state["historial"] = []          # un registro por video subido en esta sesión
if "analizados" not in st.session_state:
    st.session_state["analizados"] = set()      # huellas ya procesadas en esta sesión (solo para la interfaz)


# ---------- Procesamiento (con caché: el mismo video nunca se consulta dos veces en Azure) ----------
@st.cache_data(show_spinner=False, max_entries=200)
def procesar(contenido: bytes, nombre: str):
    """Archivo -> puntos -> imagen de trayectoria -> predicción de Azure."""
    ext = os.path.splitext(nombre)[1].lower()
    if ext == ".npz":
        datos = desde_npz(io.BytesIO(contenido))
        datos["frames"] = len(datos["pose"])
    else:
        from lsc.extraccion import extraer_de_video   # MediaPipe se carga solo si hace falta
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(contenido)
            ruta = tmp.name
        try:
            datos = extraer_de_video(ruta)
        finally:
            os.remove(ruta)
    png = dibujar_trayectoria(datos["pose"], datos["mano_izq"], datos["mano_der"], datos["ancho"], datos["alto"])
    probs = clasificar(png, URL, CLAVE)
    return png, probs, datos["frames"]


# ---------- Tablero de métricas ----------
def bloque_metricas(reales, predichas, probs, titulo_cm, prefijo):
    """Exactitud, precisión, recall y F1 (macro y por seña), matriz de confusión y curvas ROC."""
    rep = classification_report(reales, predichas, labels=CLASES, output_dict=True, zero_division=0)
    exactitud = float(np.mean(np.array(reales) == np.array(predichas)))
    # Promedio macro solo sobre las señas que tienen videos (si no, una seña aún no probada contaría como 0)
    presentes = [c for c in CLASES if rep[c]["support"] > 0]
    macro = {k: float(np.mean([rep[c][k] for c in presentes])) for k in ("precision", "recall", "f1-score")}
    aciertos = int(np.sum(np.array(reales) == np.array(predichas)))
    ui.kpis([
        ("Videos evaluados", f"{len(reales)}", f"{aciertos} aciertos", False),
        ("Exactitud", f"{exactitud * 100:.1f} %", "accuracy", True),
        ("Precisión", f"{macro['precision'] * 100:.1f} %", "promedio macro", False),
        ("Recall", f"{macro['recall'] * 100:.1f} %", "promedio macro", False),
        ("F1", f"{macro['f1-score'] * 100:.1f} %", "promedio macro", False),
    ])

    fig_roc, aucs = curvas_roc(reales, probs, CLASES)
    tabla = pd.DataFrame({c: rep[c] for c in CLASES}).T[["precision", "recall", "f1-score", "support"]]
    tabla.columns = ["Precisión", "Recall", "F1", "N.º de videos"]
    tabla["AUC"] = [aucs.get(c, np.nan) for c in CLASES]
    tabla.index.name = "Seña"
    barra = lambda nombre: st.column_config.ProgressColumn(nombre, format="%.3f", min_value=0.0, max_value=1.0)
    ui.etiqueta("Resultados por seña")
    st.dataframe(tabla.reset_index(), hide_index=True, width="stretch", key=f"{prefijo}_tabla",
                 column_config={"Seña": st.column_config.TextColumn("Seña", width="small"),
                                "Precisión": barra("Precisión"), "Recall": barra("Recall"), "F1": barra("F1"),
                                "N.º de videos": st.column_config.NumberColumn("N.º de videos", format="%d"),
                                "AUC": st.column_config.NumberColumn("AUC", format="%.3f")})

    g1, g2 = st.columns(2, gap="medium")
    with g1:
        with st.container(key=f"panel_{prefijo}_cm"):
            st.plotly_chart(matriz_confusion(confusion_matrix(reales, predichas, labels=CLASES), CLASES, titulo_cm),
                            width="stretch", theme=None, config=CONFIG_GRAFICOS, key=f"{prefijo}_cm")
    with g2:
        with st.container(key=f"panel_{prefijo}_roc"):
            if fig_roc is not None:
                st.plotly_chart(fig_roc, width="stretch", theme=None, config=CONFIG_GRAFICOS, key=f"{prefijo}_roc")
            else:
                ui.estado_vacio("Curvas ROC", "Aparecen cuando haya videos de al menos dos señas distintas.")


# ---------- Barra lateral ----------
hist_lateral = st.session_state["historial"]
with st.sidebar:
    ui.tarjeta_proyecto()
    ui.titulo_lateral("Servicio")
    ui.estado_conexion(bool(URL and CLAVE))
    if not (URL and CLAVE):
        st.caption("Configura AZURE_PREDICTION_URL y AZURE_PREDICTION_KEY en los *secrets* de Streamlit.")
    st.html('<div class="lsc-ficha" style="margin-top:.6rem">Modelo: <b>lsc-senas-v1</b><br>'
            'Plataforma: <b>Azure Custom Vision</b></div>')
    ui.titulo_lateral("Esta sesión")
    ui.contadores(len(hist_lateral), sum(h["real"] == h["prediccion"] for h in hist_lateral))
    if st.button("Reiniciar sesión", icon=":material/restart_alt:", width="stretch",
                 disabled=not hist_lateral, help="Borra los videos confirmados y las métricas de esta sesión."):
        st.session_state["historial"] = []
        st.rerun()
    ui.titulo_lateral("Privacidad")
    ui.tarjeta_privacidad()

# ---------- Encabezado y pestañas ----------
ui.encabezado()
tab1, tab2 = st.tabs([":material/videocam: Reconocer una seña",
                      ":material/info: Cómo funciona"])

# ---------- Pestaña 1: reconocer + métricas de la sesión ----------
with tab1:
    pasos = st.empty()
    ui.indicador_pasos(pasos, 0)

    c_info, c_subir = st.columns([1, 1.15], gap="large")
    with c_info:
        ui.tarjeta_instrucciones()
    with c_subir:
        archivo = st.file_uploader("Sube el video de la seña", type=["mp4", "mov", "avi", "npz"],
                                   help="El modelo reconoce la seña sin saber cuál es; después tú confirmas.")

    if archivo is not None:
        if not (URL and CLAVE):
            st.error("La app no está conectada a Azure. Revisa la configuración de los *secrets*.")
            st.stop()
        contenido = archivo.getvalue()
        huella = hashlib.sha1(contenido).hexdigest()[:16]
        ui.indicador_pasos(pasos, 1)

        # Una sola llamada a procesar() (con caché). El panel de estado solo describe lo que ocurre dentro.
        if huella not in st.session_state["analizados"]:
            with st.status("Analizando la seña...", expanded=True) as estado:
                etapas = st.empty()
                etapas.markdown(
                    ":material/progress_activity: Detectando cuerpo y manos con **MediaPipe Holistic**  \n"
                    ":material/progress_activity: Dibujando la **imagen de trayectoria** (224×224)  \n"
                    ":material/progress_activity: Consultando el **modelo en Azure Custom Vision**")
                inicio = time.perf_counter()
                try:
                    png, probs, frames = procesar(contenido, archivo.name)
                except Exception as e:
                    estado.update(label="No se pudo procesar el video", state="error", expanded=True)
                    st.error(f"No se pudo procesar el video: {e}")
                    st.stop()
                duracion = time.perf_counter() - inicio
                etapas.markdown(
                    f":material/check_circle: Puntos clave detectados · **{frames} fotogramas**  \n"
                    ":material/check_circle: Imagen de trayectoria generada  \n"
                    ":material/check_circle: Predicción recibida del modelo en la nube")
                estado.update(label=f"Seña analizada en {duracion:.1f} s", state="complete", expanded=False)
            st.session_state["analizados"].add(huella)
        else:
            try:
                png, probs, frames = procesar(contenido, archivo.name)
            except Exception as e:
                st.error(f"No se pudo procesar el video: {e}")
                st.stop()
            st.caption(":material/bolt: Resultado recuperado de la caché: este video no se volvió a enviar a Azure.")

        ganadora = next(iter(probs))
        registro = next((h for h in st.session_state["historial"] if h["huella"] == huella), None)
        ui.indicador_pasos(pasos, 5 if registro else 4)

        col_video, col_img, col_res = st.columns([1.25, 0.95, 1.35], gap="medium")
        with col_video:
            with st.container(key="panel_video"):
                ui.etiqueta("Video")
                if archivo.name.lower().endswith(".npz"):
                    ui.estado_vacio("Archivo de puntos clave (.npz)", "No hay video para reproducir.")
                else:
                    st.video(contenido)
        with col_img:
            with st.container(key="panel_trayectoria"):
                ui.etiqueta("Imagen de trayectoria")
                st.image(png, width=224)
                ui.leyenda_trayectoria(frames)
        with col_res:
            ui.tarjeta_resultado(ganadora, probs, CLASES, DESCRIPCION.get(ganadora, ""))

        # ----- Confirmación: la persona indica DESPUÉS cuál seña era realmente -----
        # El modelo decide primero, sin pistas; la respuesta real se pide después para no influir en nada.
        ui.seccion("¿Qué seña hiciste realmente?",
                   "El modelo ya decidió sin pistas. Tu respuesta solo se usa para medir su desempeño.")
        if registro is None:
            botones = st.columns(len(CLASES), gap="medium")
            for col, clase in zip(botones, CLASES):
                if col.button(f"Era {clase}", key=f"real_{clase}_{huella}", width="stretch"):
                    st.session_state["historial"].append({
                        "huella": huella, "video": archivo.name, "real": clase, "prediccion": ganadora,
                        **{f"p_{c}": probs.get(c, 0.0) for c in CLASES}, "frames": frames})
                    st.rerun()
        else:
            ui.veredicto(registro["real"], registro["prediccion"])

    # ----- Métricas de la sesión: se actualizan con cada video -----
    hist = pd.DataFrame(st.session_state["historial"])
    if len(hist):
        ui.seccion("Desempeño del modelo en esta sesión",
                   f"Calculado con {len(hist)} video(s) confirmados. Se actualiza cada vez que confirmas "
                   "la seña real de un video nuevo.")
        bloque_metricas(hist["real"].tolist(), hist["prediccion"].tolist(),
                        hist[[f"p_{c}" for c in CLASES]].to_numpy(), "Matriz de confusión (sesión)", "sesion")
        with st.expander("Historial de videos de la sesión", icon=":material/history:"):
            vista = hist.drop(columns=["huella"])
            st.dataframe(vista, hide_index=True, width="stretch", key="sesion_historial",
                         column_config={"video": "Video", "real": "Seña real", "prediccion": "Predicción",
                                        "frames": st.column_config.NumberColumn("Fotogramas", format="%d"),
                                        **{f"p_{c}": st.column_config.NumberColumn(f"P({c})", format="%.3f")
                                           for c in CLASES}})
            st.download_button("Descargar historial (CSV)", vista.to_csv(index=False).encode("utf-8"),
                               "historial_sesion.csv", "text/csv", icon=":material/download:")
    elif archivo is None:
        ui.estado_vacio("Aún no hay videos en esta sesión",
                        "Sube un video para ver la predicción y empezar a medir el desempeño del modelo.")

# ---------- Pestaña 2: explicación ----------
with tab2:
    ui.banner_flujo()
    ui.seccion("¿Cómo funciona?", "Del video a la seña reconocida en cinco pasos.")
    st.html("""
<div class="lsc-flujo">
  <div class="lsc-etapa"><div class="lsc-num">1</div><h5>Video</h5>
    <p>Una persona hace <b>una</b> seña frente a la cámara durante 2 a 4 segundos.</p></div>
  <div class="lsc-etapa"><div class="lsc-num">2</div><h5>Puntos clave</h5>
    <p><i>MediaPipe Holistic</i> detecta en cada fotograma 33 puntos del cuerpo y 21 de cada mano.</p></div>
  <div class="lsc-etapa"><div class="lsc-num">3</div><h5>Imagen de trayectoria</h5>
    <p>Todos los fotogramas se dibujan en una imagen de 224×224, centrada en los hombros y coloreada
       según el tiempo (azul = inicio, rojo = final).</p></div>
  <div class="lsc-etapa"><div class="lsc-num">4</div><h5>Modelo en la nube</h5>
    <p>La imagen se envía al modelo publicado en <b>Azure Custom Vision</b>, que responde con la
       probabilidad de cada seña.</p></div>
  <div class="lsc-etapa"><div class="lsc-num">5</div><h5>Resultado y métricas</h5>
    <p>La persona confirma cuál seña hizo; la app compara y acumula exactitud, precisión, recall, F1 y ROC.</p></div>
</div>""")
    ui.seccion("Datos de entrenamiento")
    st.html("""
<div class="lsc-cifras">
  <div class="lsc-cifra"><b>3</b><span>señas: HOLA, BIEN, MAL</span></div>
  <div class="lsc-cifra"><b>2</b><span>señantes (S00 y S01)</span></div>
  <div class="lsc-cifra"><b>60</b><span>videos · 10 repeticiones por seña</span></div>
  <div class="lsc-cifra"><b>180</b><span>imágenes con aumento de datos</span></div>
</div>
<p class="lsc-nota">Cada video genera 3 imágenes (original + 2 variantes leves).</p>
<p class="lsc-nota"><b>Evaluación:</b> la persona que usa la aplicación (P01) confirma cuál seña realizó en cada
video; con esas confirmaciones se calculan las métricas del modelo.</p>""")
    c_priv, c_lim = st.columns(2, gap="large")
    with c_priv:
        ui.seccion("Privacidad")
        st.html('<div class="lsc-tarjeta suave">Las imágenes de trayectoria se envían al modelo con la opción '
                '<b>nostore</b>: Azure no guarda ninguna copia. A Azure solo llega la imagen de trayectoria, '
                'nunca el video.</div>')
    with c_lim:
        ui.seccion("Limitaciones")
        st.html("""
<div class="lsc-aviso"><ul>
  <li>Reconoce <b>solo tres señas</b> y <b>una seña por video</b>.</li>
  <li>Entrenado con dos señantes: el desempeño puede variar con otras personas, cámaras o encuadres.</li>
  <li>Los videos de cámara frontal de celular suelen estar en <b>espejo</b>; las grabaciones de entrenamiento no lo están.</li>
</ul></div>""")

ui.pie_pagina()
