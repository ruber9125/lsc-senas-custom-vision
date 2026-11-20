"""
Corte de la seña con el MISMO criterio con el que se cortaron las muestras de entrenamiento.

Copiado del proyecto de grado (src/lsc/data/segmentacion.py y configs/captura.yaml, protocolo v1) y
adaptado para usarse sobre una grabación ya hecha, no en vivo. El criterio es movimiento y postura:

- velocidad de las muñecas, en anchos de hombro por segundo;
- altura de la muñeca más alta respecto a la línea de hombros, en anchos de hombro (negativo = debajo).

Una muñeca solo cuenta si es creíble: dentro del cuadro y con visibilidad suficiente. Con las manos en
reposo fuera del cuadro, MediaPipe inventa la muñeca y esa posición se mueve sola.
"""
from dataclasses import dataclass

import numpy as np

HOMBRO_IZQ, HOMBRO_DER = 11, 12
MUNECA_IZQ, MUNECA_DER = 15, 16


@dataclass(frozen=True)
class ParametrosSegmentacion:
    velocidad_inicio: float = 1.5       # por encima: el fotograma empuja hacia "seña"
    velocidad_fin: float = 0.8          # por debajo (y manos bajas): empuja hacia "reposo"
    altura_inicio: float = -0.75        # por encima: empuja hacia "seña"
    altura_fin: float = -0.95           # por debajo (y manos quietas): empuja hacia "reposo"
    frames_inicio: int = 5              # fotogramas seguidos para confirmar que empezó una seña
    frames_fin: int = 5                 # fotogramas seguidos para confirmar que volvió al reposo
    max_hueco_frames: int = 3           # fotogramas sin pose tolerados sin afectar la decisión
    margen_antes_frames: int = 3        # contexto guardado antes de la seña
    margen_despues_frames: int = 3      # contexto guardado después
    duracion_min_s: float = 0.3
    duracion_max_s: float = 4.0
    min_visibilidad_muneca: float = 0.3


PROTOCOLO_V1 = ParametrosSegmentacion()   # valores de configs/captura.yaml del proyecto de grado


def munecas_creibles(pose, min_visibilidad):
    """(33, 4) -> (33, 2) con las muñecas inventadas sustituidas, o None si ninguna es creíble.

    Una muñeca es creíble si está dentro del cuadro y su visibilidad llega a ``min_visibilidad``.
    Si solo una lo es, se usa esa para las dos (el segmentador mira el máximo de ambas)."""
    puntos = np.asarray(pose, dtype=np.float64)
    if puntos.shape[1] < 4 or np.isnan(puntos[[HOMBRO_IZQ, HOMBRO_DER], :2]).any():
        return None
    xy = puntos[:, :2].copy()
    creibles = [m for m in (MUNECA_IZQ, MUNECA_DER)
                if puntos[m, 3] >= min_visibilidad and not np.isnan(puntos[m, :2]).any()
                and 0.0 <= puntos[m, 0] <= 1.0 and 0.0 <= puntos[m, 1] <= 1.0]
    if not creibles:
        return None
    if len(creibles) == 1:
        xy[[MUNECA_IZQ, MUNECA_DER]] = xy[creibles[0]]
    return xy


def cortar_sena(pose, fps, ancho, alto, p: ParametrosSegmentacion = PROTOCOLO_V1):
    """Primer y último fotograma (inclusive) de la seña en una grabación, o None si no se detecta.

    pose: (T, 33, 4) con x, y normalizados y la visibilidad en la última columna. fps: fotogramas por
    segundo de esos puntos (los umbrales de fotogramas se calibraron a unos 15-20 fps).
    Es la misma máquina de estados del segmentador en vivo: reposo -> seña (tras ``frames_inicio``
    fotogramas seguidos de movimiento o manos arriba) -> reposo (tras ``frames_fin`` de quietud).
    Si la grabación termina antes de volver al reposo, la seña llega hasta el último fotograma activo.
    """
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 3 or pose.shape[2] < 4 or len(pose) < p.frames_inicio or not fps or fps <= 0:
        return None
    aspecto = float(ancho) / float(alto)
    capturando, racha_inicio, racha_fin, huecos = False, 0, 0, 0
    inicio_racha, inicio_sena, ultimo_activo, previo = 0, None, 0, None

    for i in range(len(pose)):
        t = i / fps
        xy = munecas_creibles(pose[i], p.min_visibilidad_muneca)
        puntos = None if xy is None else xy * (aspecto, 1.0)
        valido = puntos is not None and not np.isnan(puntos[[HOMBRO_IZQ, HOMBRO_DER, MUNECA_IZQ, MUNECA_DER]]).any()
        escala = float(np.linalg.norm(puntos[HOMBRO_IZQ] - puntos[HOMBRO_DER])) if valido else 0.0
        if escala == 0.0:
            huecos += 1
            if huecos <= p.max_hueco_frames:
                continue                                   # hueco tolerado: no cambia nada
            empuja_inicio, empuja_fin = False, True        # sin muñecas mucho tiempo: cuenta como reposo
        else:
            huecos = 0
            munecas = puntos[[MUNECA_IZQ, MUNECA_DER]]
            altura = float(np.max(puntos[[HOMBRO_IZQ, HOMBRO_DER], 1].mean() - munecas[:, 1])) / escala
            velocidad = 0.0
            if previo is not None and t > previo[1]:
                velocidad = float(np.linalg.norm(munecas - previo[0], axis=1).max()) / (t - previo[1]) / escala
            previo = (munecas, t)
            empuja_inicio = velocidad > p.velocidad_inicio or altura > p.altura_inicio
            empuja_fin = velocidad < p.velocidad_fin and altura < p.altura_fin

        if not capturando:
            racha_inicio = racha_inicio + 1 if empuja_inicio else 0
            if racha_inicio == 1:
                inicio_racha = i
            if racha_inicio >= p.frames_inicio:
                capturando, inicio_sena, ultimo_activo, racha_fin = True, inicio_racha, i, 0
            continue
        if not empuja_fin:
            racha_fin, ultimo_activo = 0, i
        else:
            racha_fin += 1
            if racha_fin >= p.frames_fin:
                break                                      # la seña cerró: no se busca una segunda

    if inicio_sena is None or (ultimo_activo - inicio_sena) / fps < p.duracion_min_s:
        return None
    return max(0, inicio_sena - p.margen_antes_frames), min(len(pose) - 1, ultimo_activo + p.margen_despues_frames)
