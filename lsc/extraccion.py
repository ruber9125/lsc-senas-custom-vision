"""
Video -> puntos clave con MediaPipe Holistic.

Usa el mismo modelo (holistic_landmarker.task) y la misma configuración que el sistema de captura
de la tesis con el que se grabaron S00 y S01: modo VIDEO y umbrales de confianza de 0.2.
"""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODELO = Path(__file__).resolve().parent.parent / "models" / "holistic_landmarker.task"
CONFIANZA = 0.2          # igual que configs/captura.yaml de la tesis
FPS_OBJETIVO = 23.5      # fps aproximados de las grabaciones de entrenamiento
MAX_FRAMES = 150         # límite de seguridad (~6 s): la app es para UNA seña corta


def _arreglo(puntos, n):
    out = np.full((n, 3), np.nan, dtype=np.float32)
    for i, p in enumerate(puntos or []):
        out[i] = [p.x, p.y, p.z]
    return out


def _fps_estimado(ruta_video, duracion_s):
    """fps real de un video cuyo contenedor no lo informa bien (típico en los .webm del navegador).

    Cuenta los fotogramas leyéndolos uno a uno hasta el final (no se confía en CAP_PROP_FRAME_COUNT)
    y los divide por la duración: la que midió el grabador o, si no se conoce, la marca de tiempo
    del último fotograma. Devuelve None si no se puede estimar.
    """
    cap = cv2.VideoCapture(str(ruta_video))
    n, ultimo_ms = 0, 0.0
    while cap.grab():
        n += 1
        ultimo_ms = max(ultimo_ms, cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    cap.release()
    if n == 0:
        return None
    if duracion_s and duracion_s > 0:
        duracion = float(duracion_s)
    elif n > 1 and ultimo_ms > 0:
        duracion = ultimo_ms / 1000 * n / (n - 1)      # la última marca es el INICIO del último fotograma
    else:
        return None
    fps = n / duracion
    return fps if 0 < fps <= 120 else None


def extraer_de_video(ruta_video, duracion_s=None):
    """Devuelve pose, mano_izq, mano_der (NaN donde no se detectó), ancho, alto y frames procesados.

    duracion_s (opcional): duración real de la grabación en segundos, medida por el grabador de la
    cámara. Si se indica, los fps se calculan como fotogramas / duración (no se confía en el contenedor).
    Sin ella (videos cargados) se usan los fps del video, y solo se estiman si son inválidos (<= 0 o > 120).
    """
    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        raise ValueError("No se pudo abrir el video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fps_valido = 0 < fps <= 120                        # también descarta NaN
    # Los .webm del navegador no guardan sus fps y OpenCV los adivina: a veces da 1000 y a veces un valor
    # creíble pero falso (60 con 15 reales), que haría descartar 2 de cada 3 fotogramas. Por eso, si el
    # grabador midió la duración, los fps se calculan siempre como fotogramas / duración.
    if duracion_s or not fps_valido:
        fps = _fps_estimado(ruta_video, duracion_s) or (fps if fps_valido else 30.0)
    # Si el video tiene más fps que las grabaciones de entrenamiento (p. ej. 30 o 60),
    # se toma 1 de cada `paso` fotogramas para que la densidad de puntos sea parecida.
    paso = max(1, round(fps / FPS_OBJETIVO))

    opciones = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELO)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=CONFIANZA,
        min_pose_landmarks_confidence=CONFIANZA,
        min_hand_landmarks_confidence=CONFIANZA,
    )
    pose, izq, der = [], [], []
    ancho = alto = None
    i = 0
    with vision.HolisticLandmarker.create_from_options(opciones) as holistic:
        while len(pose) < MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            if i % paso == 0:
                alto, ancho = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                r = holistic.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                              int(i * 1000 / fps))
                pose.append(_arreglo(r.pose_landmarks, 33))
                izq.append(_arreglo(r.left_hand_landmarks, 21))
                der.append(_arreglo(r.right_hand_landmarks, 21))
            i += 1
    cap.release()

    pose, izq, der = np.array(pose), np.array(izq), np.array(der)
    if len(pose) == 0:
        raise ValueError("El video no tiene fotogramas legibles.")
    if np.isnan(pose[:, 11:13, 0]).all():
        raise ValueError("No se detectó a ninguna persona (hombros) en el video.")
    if np.isnan(izq[:, 0, 0]).all() and np.isnan(der[:, 0, 0]).all():
        raise ValueError("No se detectaron manos en el video.")
    return {"pose": pose, "mano_izq": izq, "mano_der": der, "ancho": ancho, "alto": alto,
            "frames": len(pose), "fps_video": fps, "fps_puntos": fps / paso}
