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

HOMBRO_IZQ, HOMBRO_DER = 11, 12
MUNECAS = [15, 16]          # muñecas de MediaPipe Pose (existen aunque no se detecte la mano)

UMBRAL = 0.15               # la mano "está en la seña" si sube más que esto sobre el reposo (anchos de hombro)
ALTURA_MINIMA = 0.40        # si la mano nunca sube al menos esto, no hay seña clara: no se recorta
MARGEN_INI_S = 0.15         # tiempo que se conserva antes de que la mano suba (mediana en los clips: 0.14 s)
MARGEN_FIN_S = 0.20         # tiempo que se conserva después de que la mano baja (mediana en los clips: 0.21 s)
SUAVIZADO_S = 0.15          # ventana de la mediana que quita los saltos de un solo fotograma
HUECO_S = 0.25              # pausas más cortas que esto no parten la seña
DURACION_MINIMA_S = 0.6     # un recorte más corto que esto se descarta (sería un error de detección)

HOMBROS_MAX = 0.30          # hombros / lado mayor del video; en los clips de entrenamiento va de 0.18 a 0.25


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

    corte = slice(inicio, fin + 1)
    return dict(datos, pose=pose[corte], mano_izq=np.asarray(datos["mano_izq"])[corte],
                mano_der=np.asarray(datos["mano_der"])[corte], frames=fin - inicio + 1)


def fraccion_hombros(datos):
    """Ancho de hombros dividido por el lado mayor del video (None si no hay pose).

    Mide qué tan cerca está la persona de la cámara. Se usa el lado mayor para que un video vertical
    y uno horizontal de la misma cámara den el mismo valor.
    """
    px = np.array([float(datos["ancho"]), float(datos["alto"])])
    _, ancho = _centro_y_ancho(np.asarray(datos["pose"])[..., :2] * px)
    return None if ancho is None else ancho / float(px.max())


def muy_cerca(datos):
    """True si la persona está más cerca de la cámara que en los videos con los que se entrenó el modelo."""
    f = fraccion_hombros(datos)
    return f is not None and f > HOMBROS_MAX
