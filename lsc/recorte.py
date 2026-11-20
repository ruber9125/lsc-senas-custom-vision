"""
Recorte automático de la seña y comprobación del encuadre (sobre los puntos clave ya extraídos).

Los clips con los que se entrenó el modelo contienen SOLO la seña: empiezan cuando la mano sale del
reposo y terminan cuando vuelve. La cámara, en cambio, graba un tiempo fijo, con quietud antes y después.
Esa quietud corre los colores de la imagen de trayectoria (azul = inicio, rojo = final). Aquí se quita,
para que la imagen se parezca a las del entrenamiento sin importar cuánto dure la grabación.

Los umbrales se calibraron con los 90 clips del conjunto de datos, de modo que un clip ya recortado
quede prácticamente igual.
"""
import numpy as np

from .encuadre import OBJETIVO

HOMBRO_IZQ, HOMBRO_DER = 11, 12
MUNECAS = [15, 16]          # muñecas de MediaPipe Pose (existen aunque no se detecte la mano)

UMBRAL = 0.15               # la mano "está en la seña" si sube más que esto sobre el reposo (anchos de hombro)
ALTURA_MINIMA = 0.40        # si la mano nunca sube al menos esto, no hay seña clara: no se recorta
MARGEN_INI_S = 0.15         # tiempo que se conserva antes de que la mano suba (mediana en los clips: 0.14 s)
MARGEN_FIN_S = 0.20         # tiempo que se conserva después de que la mano baja (mediana en los clips: 0.21 s)
SUAVIZADO_S = 0.15          # ventana de la mediana que quita los saltos de un solo fotograma
HUECO_S = 0.25              # pausas más cortas que esto no parten la seña
DURACION_MINIMA_S = 0.6     # un recorte más corto que esto se descarta (sería un error de detección)

FPS_MUESTRAS = 20.5         # fotogramas por segundo (mediana) de las muestras de entrenamiento: 16.7 a 25.6


def _centro_y_ancho(pose_px):
    """Punto medio y ancho de hombros (en píxeles), o (None, None) si no hay pose."""
    hi, hd = pose_px[:, HOMBRO_IZQ], pose_px[:, HOMBRO_DER]
    validos = ~np.isnan(hi).any(axis=1) & ~np.isnan(hd).any(axis=1)
    if not validos.any():
        return None, None
    centro = ((hi[validos] + hd[validos]) / 2).mean(axis=0)
    ancho = float(np.linalg.norm(hi[validos] - hd[validos], axis=1).mean())
    return (centro, ancho) if ancho > 0 else (None, None)


def _mediana_movil(x, k):
    if k <= 1 or len(x) < k:
        return x
    m = k // 2
    relleno = np.pad(x, m, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(relleno, k), axis=1)


def _extremo(activo, desde, paso, hueco):
    """Último fotograma activo al avanzar desde `desde` en la dirección `paso`, tolerando pausas cortas."""
    ultimo, i = desde, desde + paso
    while 0 <= i < len(activo) and abs(i - ultimo) <= hueco + 1:
        if activo[i]:
            ultimo = i
        i += paso
    return ultimo


def recortar_sena(datos, fps):
    """Devuelve una copia de `datos` solo con los fotogramas de la seña (y "frames" actualizado).

    datos: {"pose" (T, 33, >=2), "mano_izq", "mano_der", "ancho", "alto", ...}; fps: fotogramas por segundo
    de esos puntos. Si no se encuentra una seña clara, los datos se devuelven sin recortar.
    """
    pose = np.asarray(datos["pose"])
    T = len(pose)
    sin_cambio = dict(datos, frames=T)
    if T < 3 or not fps or fps <= 0:
        return sin_cambio
    px = np.array([float(datos["ancho"]), float(datos["alto"])])
    pose_px = pose[..., :2] * px
    centro, ancho = _centro_y_ancho(pose_px)
    if centro is None:
        return sin_cambio

    # Altura de cada muñeca en anchos de hombro (hacia abajo = positivo), sin saltos de un fotograma
    y = (pose_px[:, MUNECAS, 1] - centro[1]) / ancho
    y = np.where(np.isnan(y), np.nanmax(y, axis=0, initial=-np.inf, keepdims=True), y)
    if not np.isfinite(y).all():
        return sin_cambio
    k = max(1, round(SUAVIZADO_S * fps)) | 1                        # ventana impar
    hueco = round(HUECO_S * fps)
    margen_ini, margen_fin = round(MARGEN_INI_S * fps), round(MARGEN_FIN_S * fps)
    n_reposo = max(2, round(SUAVIZADO_S * fps))

    inicio, fin = T - 1, 0
    for j in range(y.shape[1]):
        yj = _mediana_movil(y[:, j], k)
        # El reposo se mide por separado al inicio y al final: la mano no siempre vuelve al mismo punto.
        reposo_ini, reposo_fin = np.median(yj[:n_reposo]), np.median(yj[-n_reposo:])
        pico = int(np.argmin(yj))
        sube, baja = reposo_ini - yj[pico], reposo_fin - yj[pico]
        if max(sube, baja) < ALTURA_MINIMA:
            continue                                                # esta mano no hace la seña
        # Si la grabación empieza (o termina) con la mano arriba, la seña quedó cortada: ese lado no se recorta.
        ini_j = _extremo(reposo_ini - yj > UMBRAL, pico, -1, hueco) - margen_ini if sube >= 0.5 * baja else 0
        fin_j = _extremo(reposo_fin - yj > UMBRAL, pico, +1, hueco) + margen_fin if baja >= 0.5 * sube else T - 1
        inicio, fin = min(inicio, ini_j), max(fin, fin_j)
    inicio, fin = max(0, inicio), min(T - 1, fin)
    if fin - inicio + 1 < DURACION_MINIMA_S * fps:                  # también cubre "ninguna mano se movió"
        return sin_cambio

    return seleccionar(datos, slice(inicio, fin + 1))


def seleccionar(datos, indices):
    """Copia de `datos` solo con esos fotogramas (pose, manos y rostro) y "frames" actualizado."""
    nuevo = dict(datos)
    for clave in ("pose", "mano_izq", "mano_der", "rostro"):
        if datos.get(clave) is not None:
            nuevo[clave] = np.asarray(datos[clave])[indices]
    nuevo["frames"] = len(nuevo["pose"])
    return nuevo


def cortar_como_las_muestras(datos, fps):
    """Deja solo la seña, cortada con el segmentador del proyecto de grado (el que cortó las muestras).

    Necesita la visibilidad de la pose. Si ese segmentador no detecta la seña (por ejemplo, la pose no trae
    visibilidad), se usa el recorte por altura de la muñeca de este módulo. Devuelve (datos, método)."""
    from .segmentacion import cortar_sena
    corte = cortar_sena(datos["pose"], fps, datos["ancho"], datos["alto"])
    if corte is not None:
        return seleccionar(datos, slice(corte[0], corte[1] + 1)), "segmentador"
    recortado = recortar_sena(datos, fps)
    return recortado, "altura" if recortado["frames"] < len(datos["pose"]) else "sin recorte"


def ajustar_fps(datos, fps, objetivo=FPS_MUESTRAS):
    """Deja los puntos con el ritmo de las muestras de entrenamiento (unos 20 fotogramas por segundo).

    Las muestras se grabaron a 17-26 fps; la cámara del navegador entrega 30. Con más fotogramas, la imagen
    de trayectoria queda con más puntos que las del entrenamiento. Se conservan los fotogramas más cercanos
    a cada instante (no se inventan puntos). Si el video ya es igual o más lento, no se toca.
    Devuelve (datos, fps resultante).
    """
    T = len(datos["pose"])
    if not fps or fps <= objetivo * 1.15 or T < 3:
        return dict(datos, frames=T), fps
    n = max(2, round(T / fps * objetivo))
    idx = np.unique(np.round(np.linspace(0, T - 1, n)).astype(int))
    return seleccionar(datos, idx), fps * len(idx) / T


def _numero(x, decimales=0):
    return f"{x:.{decimales}f}".replace(".", ",")


def comprobar_captura(datos, fps=None, encuadre=None):
    """Compara la captura con los parámetros de las 60 muestras de entrenamiento (S00 y S01).

    encuadre: resultado de lsc.encuadre.evaluar_grabacion (el criterio de encuadre del proyecto de grado).

    Devuelve una lista de filas {"Parámetro", "Esta captura", "Muestras de entrenamiento", "Cumple"}.
    "Muestras" es el rango medido en las muestras; "Cumple" usa una tolerancia que también cubre los 30
    videos de prueba (no usados en el entrenamiento), grabados en otras sesiones.
    """
    pose = np.asarray(datos["pose"])
    T = len(pose)
    W, H = float(datos["ancho"]), float(datos["alto"])
    filas = []

    def fila(parametro, valor, muestras, cumple):
        filas.append({"Parámetro": parametro, "Esta captura": valor, "Muestras de entrenamiento": muestras,
                      "Cumple": "Sí" if cumple else "No"})

    fila("Resolución del video", f"{int(W)} × {int(H)}", "1280 × 720 (horizontal)", W > H and 1.6 <= W / H <= 1.9)
    if fps:
        fila("Fotogramas por segundo", _numero(fps, 1), "16,7 a 25,6", 13 <= fps <= 27)
        fila("Duración de la seña", f"{_numero(T / fps, 1)} s", "1,5 a 2,9 s", 1.2 <= T / fps <= 3.2)
    fila("Fotogramas de la seña", str(T), "32 a 67", 28 <= T <= 70)

    e = encuadre or {}
    if not e.get("con_pose"):
        fila("Persona detectada", "No", "Sí", False)
        return filas
    o = OBJETIVO
    error = e["ojos"] / o.ojos - 1
    fila("Distancia a la cámara" + ("" if e.get("con_cara") else " (estimada por los hombros)"),
         "Igual que en las muestras" if abs(error) <= 0.005 else
         f"Se ve un {_numero(abs(error) * 100)} % más {'grande' if error > 0 else 'pequeño'}",
         "Tolerancia: 10 %", abs(error) <= o.tolerancia_salir)
    if e.get("con_cara"):
        fila("De frente a la cámara", "Sí" if e["desvio_nariz"] <= o.max_desvio_nariz else "Girado", "Sí",
             e["desvio_nariz"] <= o.max_desvio_nariz)
    fila("Persona centrada", f"{_numero(100 * e['centro_x'])} % desde la izquierda",
         f"{_numero(100 * (o.centro_x - o.tolerancia_x))} a {_numero(100 * (o.centro_x + o.tolerancia_x))} %",
         abs(e["centro_x"] - o.centro_x) <= o.tolerancia_x)
    relajo = 1 - o.tolerancia_salir
    fila("Espacio sobre los hombros", f"{_numero(e['holgura_arriba'], 2)} anchos de hombro",
         f"Al menos {_numero(o.holgura_arriba * relajo, 2)}", e["holgura_arriba"] >= o.holgura_arriba * relajo)
    fila("Espacio bajo los hombros", f"{_numero(e['holgura_abajo'], 2)} anchos de hombro",
         f"Al menos {_numero(o.holgura_abajo * relajo, 2)}", e["holgura_abajo"] >= o.holgura_abajo * relajo)

    con_izq = ~np.isnan(np.asarray(datos["mano_izq"])[:, 0, 0])
    con_der = ~np.isnan(np.asarray(datos["mano_der"])[:, 0, 0])
    fila("Fotogramas con la mano detectada", f"{_numero(100 * (con_izq | con_der).mean())} %", "76 a 94 %",
         (con_izq | con_der).mean() >= 0.55)
    derecha = con_der.sum() >= con_izq.sum()
    fila("Mano que hace la seña", "Derecha" if derecha else "Izquierda", "Derecha", bool(derecha))
    return filas
