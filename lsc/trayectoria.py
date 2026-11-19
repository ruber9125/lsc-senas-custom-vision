"""
Imagen de trayectoria: resume una seña (secuencia de puntos clave) en UNA imagen de 224x224.

Es EXACTAMENTE el mismo dibujo que generó las imágenes de entrenamiento
(scripts/generar_imagenes.py). Si el dibujo cambiara, el modelo vería imágenes distintas
a las que aprendió y fallaría: el preprocesamiento en producción debe ser idéntico al del entrenamiento.
"""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TAM = 224
HOMBRO_IZQ, HOMBRO_DER = 11, 12
TORSO = [(11, 12), (0, 11), (0, 12)]   # hombros y nariz (índices de MediaPipe Pose)
BRAZOS = [13, 14, 15, 16]               # codos y muñecas


def dibujar_trayectoria(pose, mano_izq, mano_der, ancho, alto):
    """pose: (T, 33, >=2), manos: (T, 21, >=2), coordenadas de MediaPipe (fracción del ancho/alto).
    Devuelve la imagen PNG en bytes."""
    px = np.array([float(ancho), float(alto)])
    pose = np.asarray(pose)[..., :2] * px
    manos = [np.asarray(mano_izq)[..., :2] * px, np.asarray(mano_der)[..., :2] * px]
    T = len(pose)

    # Normalización: centro en el punto medio de los hombros, escala = ancho de hombros
    hi, hd = pose[:, HOMBRO_IZQ], pose[:, HOMBRO_DER]
    centro = np.nanmean((hi + hd) / 2, axis=0)
    escala = np.nanmean(np.linalg.norm(hi - hd, axis=1))
    norm = lambda p: (p - centro) / escala

    fig = plt.figure(figsize=(TAM / 100, TAM / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    colores = plt.cm.turbo(np.linspace(0, 1, T))       # azul = inicio, rojo = final

    for a, b in TORSO:
        p = norm(pose[0, [a, b]])
        ax.plot(p[:, 0], p[:, 1], color=(0.4, 0.4, 0.4), lw=1)

    for mano in manos:
        muneca = mano[:, 0]
        ok = ~np.isnan(muneca).any(axis=1)
        if ok.sum() > 1:
            p = norm(muneca[ok])
            ax.plot(p[:, 0], p[:, 1], color=(0.75, 0.75, 0.75), lw=0.6, alpha=0.6)

    for t in range(T):
        for mano in manos:
            h = mano[t]
            if np.isnan(h).any():
                continue
            p = norm(h)
            ax.scatter(p[:, 0], p[:, 1], s=1.5, color=colores[t], linewidths=0)
        p = norm(pose[t, BRAZOS])
        ax.scatter(p[:, 0], p[:, 1], s=3, color=colores[t], linewidths=0)

    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(1.9, -1.4)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="black")
    plt.close(fig)
    return buf.getvalue()


def desde_npz(archivo):
    """Lee un .npz del sistema de captura de la tesis (ruta o archivo subido)."""
    z = np.load(archivo)
    return {"pose": z["pose"], "mano_izq": z["mano_izq"], "mano_der": z["mano_der"],
            "ancho": int(z["ancho"]), "alto": int(z["alto"])}
