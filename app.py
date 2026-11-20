"""
LSC Atiende — Reconocimiento de señas para la atención accesible al ciudadano sordo.
Prototipo de punto de atención para personas sordas en entidades públicas (HOLA, BIEN, MAL en LSC).
Maestría en Inteligencia Artificial, Universidad de La Salle.

Flujo:  video -> MediaPipe Holistic -> imagen de trayectoria (224x224)
        -> modelo publicado en Azure Custom Vision (endpoint REST) -> seña reconocida.

El video llega por dos caminos (grabarlo con la cámara o cargar un archivo) que terminan en el MISMO flujo.
Cada video se analiza UNA vez. El modelo reconoce la seña sin pistas; DESPUÉS la persona confirma cuál era
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
import requests
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from lsc.azure_cv import clasificar
from lsc.recorte import ajustar_fps, comprobar_captura, encuadre_inadecuado, recortar_sena
from lsc.trayectoria import desde_npz, dibujar_trayectoria
from interfaz import componentes as ui
from interfaz.grabador import grabar_con_camara
from interfaz.graficos import CONFIG as CONFIG_GRAFICOS, curvas_roc, matriz_confusion

CLASES = ["HOLA", "BIEN", "MAL"]
DESCRIPCION = {"HOLA": "Saludo: la mano abierta sube a la frente y sale hacia afuera.",
               "BIEN": "Pulgar hacia arriba frente al pecho.",
               "MAL": "Pulgar hacia abajo frente al pecho."}
# Mensaje de atención que acompaña a cada seña reconocida
MENSAJE = {"HOLA": "👋 Una persona sorda solicita atención",
           "BIEN": "Calificación del servicio: BIEN 👍",
           "MAL": "Calificación del servicio: MAL 👎"}
ENTRADA_CAMARA = "🎥 Grabar con la cámara"
ENTRADA_ARCHIVO = "📁 Cargar un video"
# Por debajo de esta confianza se sugiere repetir la seña o acudir al Centro de Relevo (solo es un aviso)
CONFIANZA_MINIMA = 0.60
AVISO_BAJA_CONFIANZA = ("No se reconoció con seguridad. Puede repetir la seña o solicitar interpretación "
                        "en línea con el Centro de Relevo.")

st.set_page_config(page_title=ui.NOMBRE_APP, page_icon="🤟", layout="wide")
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


# ---------- Consulta al modelo (los errores de red nunca muestran la URL del endpoint, que es privada) ----------
def consultar_azure(png):
    try:
        return clasificar(png, URL, CLAVE)
    except requests.exceptions.Timeout:
        raise RuntimeError("Azure tardó demasiado en responder. Intente de nuevo.") from None
    except requests.exceptions.HTTPError as e:
        codigo = e.response.status_code if e.response is not None else "desconocido"
        raise RuntimeError(f"Azure respondió con un error ({codigo}). Intente de nuevo en unos minutos.") from None
    except requests.exceptions.RequestException:
        raise RuntimeError("No se pudo conectar con Azure. Intente de nuevo en unos minutos.") from None


# ---------- Procesamiento (con caché: el mismo video nunca se consulta dos veces en Azure) ----------
@st.cache_data(show_spinner=False, max_entries=200)
def procesar(contenido: bytes, nombre: str, duracion_s: float | None = None, recortar: bool = False):
    """Archivo -> puntos -> imagen de trayectoria -> predicción de Azure.

    duracion_s: duración real medida por el grabador de cámara (None para los videos cargados).
    recortar: quita la quietud del inicio y del final (grabaciones con la cámara, de duración fija).
    Devuelve la imagen, las probabilidades, los fotogramas de la seña y detalles del video.
    """
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
            datos = extraer_de_video(ruta, duracion_s=duracion_s)
        finally:
            os.remove(ruta)
    info = {"frames_video": datos["frames"], "encuadre_inadecuado": encuadre_inadecuado(datos)}
    fps = datos.get("fps_puntos")
    if recortar and ext != ".npz":
        # Se lleva la grabación a los parámetros de las muestras de entrenamiento: solo la seña (sin la
        # quietud del inicio y del final) y al mismo ritmo de fotogramas.
        datos = recortar_sena(datos, fps)
        datos, fps = ajustar_fps(datos, fps)
    info["comprobacion"] = comprobar_captura(datos, fps)
    png = dibujar_trayectoria(datos["pose"], datos["mano_izq"], datos["mano_der"], datos["ancho"], datos["alto"])
    probs = consultar_azure(png)
    return png, probs, datos["frames"], info


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


# ---------- Análisis de UN video: el mismo flujo para la cámara y para el archivo cargado ----------
def analizar_video(pasos, contenido: bytes, nombre: str, duracion_s=None, recortar=False):
    """MediaPipe -> trayectoria -> Azure (con caché) -> tarjeta de resultado -> confirmación a ciegas."""
    if not (URL and CLAVE):
        st.error("El servicio no está conectado a Azure. Revise la configuración de los *secrets*.")
        st.stop()
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
                png, probs, frames, info = procesar(contenido, nombre, duracion_s, recortar)
            except Exception as e:
                estado.update(label="No se pudo procesar el video", state="error", expanded=True)
                st.error(f"No se pudo procesar el video: {e}")
                st.stop()
            duracion = time.perf_counter() - inicio
            detalle = (f"**{frames} fotogramas**" if info["frames_video"] == frames else
                       f"**{info['frames_video']} fotogramas** grabados, **{frames}** de la seña (sin la quietud, al ritmo de las muestras)")
            etapas.markdown(
                f":material/check_circle: Puntos clave detectados · {detalle}  \n"
                ":material/check_circle: Imagen de trayectoria generada  \n"
                ":material/check_circle: Predicción recibida del modelo en la nube")
            estado.update(label=f"Seña analizada en {duracion:.1f} s", state="complete", expanded=False)
        st.session_state["analizados"].add(huella)
    else:
        try:
            png, probs, frames, info = procesar(contenido, nombre, duracion_s, recortar)
        except Exception as e:
            st.error(f"No se pudo procesar el video: {e}")
            st.stop()
        st.caption(":material/bolt: Resultado recuperado de la caché: este video no se volvió a enviar a Azure.")

    ganadora = next(iter(probs))
    registro = next((h for h in st.session_state["historial"] if h["huella"] == huella), None)
    ui.indicador_pasos(pasos, 5 if registro else 4)

    if info["encuadre_inadecuado"]:          # antes del resultado: explica por qué puede no ser confiable
        ui.aviso_encuadre()
    col_video, col_img, col_res = st.columns([1.25, 0.95, 1.35], gap="medium")
    with col_video:
        with st.container(key="panel_video", height="stretch"):   # las tres tarjetas, a la misma altura
            ui.etiqueta("Video")
            if nombre.lower().endswith(".npz"):
                ui.estado_vacio("Archivo de puntos clave (.npz)", "No hay video para reproducir.")
            else:
                st.video(contenido, format="video/webm" if nombre.lower().endswith(".webm") else "video/mp4")
    with col_img:
        with st.container(key="panel_trayectoria", height="stretch"):
            ui.etiqueta("Imagen de trayectoria")
            st.image(png, width=224)
            ui.leyenda_trayectoria(frames)
    with col_res:
        aviso = AVISO_BAJA_CONFIANZA if probs.get(ganadora, 0.0) < CONFIANZA_MINIMA else ""
        ui.tarjeta_resultado(ganadora, probs, CLASES, DESCRIPCION.get(ganadora, ""), MENSAJE.get(ganadora, ""),
                             aviso)

    # ----- Comprobación: la captura frente a los parámetros de las muestras de entrenamiento -----
    filas = info["comprobacion"]
    fuera = sum(f["Cumple"] == "No" for f in filas)
    titulo = ("Comprobación de la captura: todo dentro de los parámetros de las muestras" if not fuera else
              f"Comprobación de la captura: {fuera} parámetro(s) fuera de los de las muestras")
    with st.expander(titulo, expanded=bool(fuera), icon=":material/fact_check:"):
        st.caption("El modelo se entrenó con videos grabados de una sola forma. Mientras más se parezca la captura "
                   "a esos videos, más confiable es el resultado.")
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch", key=f"comprobacion_{huella}")

    # ----- Confirmación: la persona indica DESPUÉS cuál seña era realmente -----
    # El modelo decide primero, sin pistas; la respuesta real se pide después para no influir en nada.
    ui.seccion("¿Qué seña hizo realmente la persona?",
               "El sistema ya respondió sin pistas. Esta confirmación solo se usa para evaluar el modelo.")
    if registro is None:
        botones = st.columns(len(CLASES), gap="medium")
        for col, clase in zip(botones, CLASES):
            if col.button(f"Era {clase}", key=f"real_{clase}_{huella}", width="stretch"):
                st.session_state["historial"].append({
                    "huella": huella, "video": nombre, "real": clase, "prediccion": ganadora,
                    **{f"p_{c}": probs.get(c, 0.0) for c in CLASES}, "frames": frames})
                st.rerun()
    else:
        ui.veredicto(registro["real"], registro["prediccion"])


# ---------- Barra lateral ----------
hist_lateral = st.session_state["historial"]
with st.sidebar:
    ui.tarjeta_proyecto()
    ui.titulo_lateral("Estado del servicio")
    ui.estado_conexion(bool(URL and CLAVE))
    if not (URL and CLAVE):
        st.caption("Configure AZURE_PREDICTION_URL y AZURE_PREDICTION_KEY en los *secrets* de Streamlit.")
    st.html('<div class="lsc-ficha" style="margin-top:.6rem">Modelo: <b>lsc-senas-v1</b><br>'
            'Plataforma: <b>Azure Custom Vision</b></div>')
    ui.titulo_lateral("Esta sesión")
    ui.contadores(len(hist_lateral), sum(h["real"] == h["prediccion"] for h in hist_lateral))
    if st.button("Reiniciar sesión", icon=":material/restart_alt:", width="stretch",
                 disabled=not hist_lateral, help="Borra las señas confirmadas y la evaluación de esta sesión."):
        st.session_state["historial"] = []
        st.rerun()
    ui.titulo_lateral("Intérprete")
    ui.nota_relevo()
    ui.titulo_lateral("Privacidad")
    ui.tarjeta_privacidad()

# ---------- Encabezado y pestañas ----------
ui.encabezado()
tab1, tab2 = st.tabs([":material/support_agent: Atención en ventanilla",
                      ":material/info: Cómo funciona"])

# ---------- Pestaña 1: atención en ventanilla + evaluación del modelo en la sesión ----------
with tab1:
    ui.bienvenida()
    pasos = st.empty()
    ui.indicador_pasos(pasos, 0)

    # ----- Entrada del video: dos caminos que entregan los bytes al MISMO flujo (analizar_video) -----
    modo = st.segmented_control("Forma de entregar el video", [ENTRADA_CAMARA, ENTRADA_ARCHIVO],
                                default=ENTRADA_CAMARA, required=True, key="modo_entrada",
                                label_visibility="collapsed")
    entrada = None                      # (contenido, nombre, duración real en segundos o None, recortar)
    c_info, c_entrada = st.columns([1, 1.15], gap="large")
    if modo == ENTRADA_ARCHIVO:
        with c_info:
            ui.tarjeta_instrucciones()
        with c_entrada:
            archivo = st.file_uploader("Cargue el video de la seña", type=["mp4", "mov", "avi", "npz"],
                                       help="El sistema reconoce la seña sin saber cuál es; después se confirma.")
        if archivo is not None:
            entrada = (archivo.getvalue(), archivo.name, None, False)
    else:
        with c_info:
            ui.tarjeta_instrucciones(camara=True)
        with c_entrada:
            grabacion = grabar_con_camara()
        if grabacion is not None:
            # La cámara graba un tiempo fijo: se recorta la quietud del inicio y del final.
            entrada = (grabacion["contenido"], grabacion["nombre"], grabacion["duracion_s"], True)

    if entrada is not None:
        analizar_video(pasos, *entrada)

    # ----- Métricas de la sesión: se actualizan con cada video -----
    hist = pd.DataFrame(st.session_state["historial"])
    if len(hist):
        ui.seccion("Evaluación del modelo en esta sesión",
                   "Sirve para medir el desempeño del modelo con personas que no participaron en el "
                   f"entrenamiento. Calculada con {len(hist)} seña(s) confirmada(s); se actualiza con cada "
                   "confirmación.")
        bloque_metricas(hist["real"].tolist(), hist["prediccion"].tolist(),
                        hist[[f"p_{c}" for c in CLASES]].to_numpy(), "Matriz de confusión (sesión)", "sesion")
        with st.expander("Historial de señas de la sesión", icon=":material/history:"):
            vista = hist.drop(columns=["huella"])
            st.dataframe(vista, hide_index=True, width="stretch", key="sesion_historial",
                         column_config={"video": "Video", "real": "Seña realizada", "prediccion": "Seña reconocida",
                                        "frames": st.column_config.NumberColumn("Fotogramas", format="%d"),
                                        **{f"p_{c}": st.column_config.NumberColumn(f"P({c})", format="%.3f")
                                           for c in CLASES}})
            st.download_button("Descargar historial (CSV)", vista.to_csv(index=False).encode("utf-8"),
                               "historial_sesion.csv", "text/csv", icon=":material/download:")
    elif entrada is None:
        ui.estado_vacio("Aún no hay señas en esta sesión",
                        "Grabe la seña con la cámara o cargue un video para iniciar la atención.")

# ---------- Pestaña 2: explicación para el funcionario ----------
with tab2:
    ui.banner_flujo()

    # a) Para qué sirve
    ui.seccion("Para qué sirve")
    st.html("""
<div class="lsc-tarjeta">
  <p class="lsc-parrafo"><b>LSC Atiende</b> es el prototipo de un <b>punto de atención accesible</b> para personas
  sordas en entidades públicas. Permite que la persona se comunique con la ventanilla en <b>Lengua de Señas
  Colombiana (LSC)</b> en dos momentos de la atención:</p>
  <ul class="lsc-lista">
    <li><b>Saludo de inicio.</b> Con la seña <b>HOLA</b>, la persona sorda avisa que necesita atención.</li>
    <li><b>Encuesta de satisfacción.</b> Con las señas <b>BIEN</b> o <b>MAL</b>, califica el servicio recibido.</li>
  </ul>
  <p class="lsc-parrafo">Piloto propuesto: <b>Secretaría de Educación de Valledupar</b>, con réplica en las
  Secretarías de Educación del país.</p>
</div>""")

    # b) Cómo se usa
    ui.seccion("Cómo se usa")
    st.html("""
<div class="lsc-tarjeta suave">
  <ol class="lsc-lista">
    <li>En la pestaña <b>Atención en ventanilla</b>, elija <b>Grabar con la cámara</b> o <b>Cargar un video</b>.</li>
    <li>Con la cámara: pulse <b>Activar cámara</b> y luego <b>Grabar seña</b>. Hay una cuenta regresiva de 5 segundos y se
        graban <b>4 segundos</b>.</li>
    <li>La persona hace <b>una sola seña</b>, de frente, de la cintura hacia arriba y con buena luz.
        <b>Empieza y termina con las manos abajo.</b></li>
    <li>En pocos segundos aparece la seña reconocida y el mensaje para el funcionario.</li>
    <li>Si el sistema no reconoce la seña con seguridad, se puede repetir o acudir al <b>Centro de Relevo</b>.</li>
  </ol>
</div>""")

    # c) Qué pasa por dentro
    ui.seccion("Qué pasa por dentro", "De la seña al resultado en cinco pasos.")
    st.html("""
<div class="lsc-flujo">
  <div class="lsc-etapa"><div class="lsc-num">1</div><h5>Video</h5>
    <p>La persona hace <b>una</b> seña frente a la cámara. El video se graba con la cámara del dispositivo
       (4 segundos; la app quita sola la quietud del inicio y del final) o se carga desde él.</p></div>
  <div class="lsc-etapa"><div class="lsc-num">2</div><h5>Puntos clave</h5>
    <p><i>MediaPipe Holistic</i> ubica en cada instante del video los puntos del cuerpo y de las manos.</p></div>
  <div class="lsc-etapa"><div class="lsc-num">3</div><h5>Imagen de trayectoria</h5>
    <p>El recorrido de las manos se dibuja en una sola imagen, coloreada según el tiempo
       (azul = inicio, rojo = final).</p></div>
  <div class="lsc-etapa"><div class="lsc-num">4</div><h5>Modelo en la nube</h5>
    <p>La imagen se envía al modelo publicado en <b>Azure Custom Vision</b>, que responde con la
       probabilidad de cada seña.</p></div>
  <div class="lsc-etapa"><div class="lsc-num">5</div><h5>Resultado</h5>
    <p>La ventanilla muestra la seña reconocida y su mensaje: inicio de la atención o calificación
       del servicio.</p></div>
</div>""")
    st.html("""
<div class="lsc-cifras">
  <div class="lsc-cifra"><b>3</b><span>señas: HOLA, BIEN, MAL</span></div>
  <div class="lsc-cifra"><b>2</b><span>señantes en el entrenamiento</span></div>
  <div class="lsc-cifra"><b>60</b><span>videos · 10 repeticiones por seña</span></div>
  <div class="lsc-cifra"><b>180</b><span>imágenes con aumento de datos</span></div>
</div>
<p class="lsc-nota">El modelo se entrenó con estos datos. En la pestaña <b>Atención en ventanilla</b> se evalúa su
desempeño con personas que no participaron en el entrenamiento.</p>""")

    c_priv, c_lim = st.columns(2, gap="large")
    with c_priv:
        # d) Privacidad
        ui.seccion("Privacidad")
        st.html("""
<div class="lsc-tarjeta suave"><ul class="lsc-lista">
  <li>El video se usa solo para el análisis y <b>no se guarda</b>.</li>
  <li>A Azure solo se envía la imagen de trayectoria, con la opción <b>nostore</b>: Azure no almacena las imágenes.</li>
  <li>La imagen de trayectoria está hecha de puntos y líneas: <b>no contiene rostros</b>.</li>
</ul></div>""")
    with c_lim:
        # e) Alcance y límites
        ui.seccion("Alcance y límites")
        st.html("""
<div class="lsc-aviso"><ul>
  <li>Hoy reconoce <b>tres señas</b> (HOLA, BIEN y MAL) y <b>una seña por video</b>.</li>
  <li><b>No reemplaza al intérprete.</b> Complementa al <b>Centro de Relevo</b> (MinTIC–Fenascol), al que se remite
      lo que el sistema no reconoce.</li>
  <li>Se entrenó con dos señantes: el desempeño puede variar con otras personas, cámaras o encuadres.</li>
  <li>Los videos cargados desde la cámara frontal de un celular suelen estar en <b>espejo</b>; los del
      entrenamiento no lo están.</li>
</ul></div>""")

    # f) Próximas fases
    ui.seccion("Próximas fases")
    st.html("""
<div class="lsc-tarjeta">
  <p class="lsc-parrafo"><b>Vocabulario de trámites</b> para la atención en las Secretarías de Educación:</p>
  <div class="lsc-chips">
    <span>matrícula</span><span>certificado</span><span>traslado</span><span>cita</span>
    <span>sí</span><span>no</span><span>gracias</span><span>ayuda</span><span>…</span>
  </div>
  <p class="lsc-parrafo" style="margin-top:.8rem">Después, el reconocimiento de <b>frases cortas</b>.</p>
</div>""")

ui.pie_pagina()
