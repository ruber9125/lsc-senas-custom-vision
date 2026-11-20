"""
Encuadre: el mismo criterio con el que se grabaron las muestras de entrenamiento.

Copiado del proyecto de grado (scripts/encuadre.py y la sección `encuadre` de configs/captura.yaml) y
adaptado para evaluar una grabación ya hecha. Qué se exige:

- persona de frente (nariz centrada entre los ojos);
- DISTANCIA en banda, medida con la distancia entre las esquinas externas de los ojos (no cambia con la
  contextura). Sin cara, se estima con el ancho de hombros y la relación hombros/ojos;
- persona centrada;
- que la SEÑA quepa: anchos de hombro libres por encima y por debajo de la línea de hombros.

La altura de la línea de hombros (`hombros_y`) es solo una referencia para colocarse, no una condición.
"""
from dataclasses import dataclass

import numpy as np

HOMBRO_IZQ, HOMBRO_DER = 11, 12                    # MediaPipe Pose
OJO_DER, OJO_IZQ, NARIZ = 33, 263, 1               # MediaPipe FaceMesh: esquinas externas de los ojos y nariz
PUNTOS_ROSTRO = [OJO_DER, OJO_IZQ, NARIZ]          # los únicos que se guardan del rostro (no se dibujan)
RATIO_HOMBROS_OJOS = 3.16                          # medido en las dos personas grabadas (3,0 a 3,3)

# Proporciones de la cabeza respecto a la distancia entre ojos, para dibujar la zona de la guía.
CARA_ANCHO, CARA_ALTO, CARA_SOBRE_HOMBROS = 1.6, 1.35, 1.88


@dataclass(frozen=True)
class ObjetivoEncuadre:
    """Dónde debe quedar la persona (valores de configs/captura.yaml, calibrados con las muestras)."""
    ojos: float = 0.121               # distancia entre ojos, como fracción del ALTO de la imagen
    tolerancia_entrar: float = 0.07   # banda estricta para dar el visto bueno
    tolerancia_salir: float = 0.10    # banda ancha antes de volver a avisar
    hombros_y: float = 0.56           # altura de REFERENCIA de la línea de hombros (0 arriba, 1 abajo)
    tolerancia_y: float = 0.03        # ancho de la franja dibujada; no es una condición
    centro_x: float = 0.524
    tolerancia_x: float = 0.06
    max_desvio_nariz: float = 0.18    # desvío de la nariz respecto al centro de los ojos (fracción de ojos)
    distancia_referencia_cm: float = 60.0
    holgura_arriba: float = 1.30      # anchos de hombro que deben caber SOBRE la línea de hombros
    holgura_abajo: float = 1.00       # anchos de hombro que deben caber BAJO la línea de hombros


OBJETIVO = ObjetivoEncuadre()


def medir(pose_xy, ojos_nariz_xy, aspecto, ratio_hombros_ojos=RATIO_HOMBROS_OJOS):
    """Medidas del encuadre en un fotograma. Coordenadas normalizadas; ojos_nariz_xy = (3, 2) o None."""
    escala = np.array([aspecto, 1.0])
    m = {"ojos": float("nan"), "hombros_y": float("nan"), "centro_x": float("nan"), "hombros": float("nan"),
         "desvio_nariz": float("nan"), "con_cara": False, "con_pose": False}
    if pose_xy is not None:
        hombros = np.asarray(pose_xy, dtype=np.float64)[[HOMBRO_IZQ, HOMBRO_DER], :2]
        if not np.isnan(hombros).any():
            m["con_pose"] = True
            m["hombros_y"] = float(hombros[:, 1].mean())
            m["centro_x"] = float(hombros[:, 0].mean())
            m["hombros"] = float(np.linalg.norm((hombros[0] - hombros[1]) * escala))
    if ojos_nariz_xy is not None:
        cara = np.asarray(ojos_nariz_xy, dtype=np.float64)[:, :2]
        if not np.isnan(cara).any():
            ojo_d, ojo_i, nariz = cara[0] * escala, cara[1] * escala, cara[2] * escala
            m["con_cara"] = True
            m["ojos"] = float(np.linalg.norm(ojo_d - ojo_i))
            m["desvio_nariz"] = abs(float(nariz[0] - (ojo_d[0] + ojo_i[0]) / 2)) / max(m["ojos"], 1e-6)
    if not m["con_cara"] and m["con_pose"] and ratio_hombros_ojos:
        m["ojos"] = m["hombros"] / ratio_hombros_ojos       # sin cara: se estima con los hombros
    return m


def evaluar(m, objetivo: ObjetivoEncuadre = OBJETIVO, banda_ancha=True):
    """Qué hacer para quedar bien encuadrado: estado, instrucción corta, detalle y ok.

    banda_ancha: usa la tolerancia de salida (±10 %) y relaja la holgura en la misma proporción, como hace
    la guía en vivo una vez que la persona ya estaba bien ubicada."""
    tolerancia = objetivo.tolerancia_salir if banda_ancha else objetivo.tolerancia_entrar
    relajo = 1.0 - objetivo.tolerancia_salir if banda_ancha else 1.0
    if not m.get("con_pose"):
        return {"estado": "sin_persona", "instruccion": "Ubíquese frente a la cámara", "detalle": "", "ok": False}
    if m.get("con_cara") and m["desvio_nariz"] > objetivo.max_desvio_nariz:
        return {"estado": "girado", "instruccion": "Mire de frente a la cámara", "ok": False,
                "detalle": "de lado, los ojos se ven más juntos y la distancia sale mal"}
    ojos = m.get("ojos", float("nan"))
    if not np.isfinite(ojos) or ojos <= 1e-6:
        return {"estado": "sin_cara", "instruccion": "Mire de frente a la cámara", "ok": False,
                "detalle": "no se ve la cara y no hay con qué estimar la distancia"}
    error = ojos / objetivo.ojos - 1
    if abs(error) > tolerancia:
        # La distancia va como 1/tamaño aparente: si los ojos se ven pequeños, la persona está lejos.
        centimetros = abs(objetivo.distancia_referencia_cm * (objetivo.ojos / ojos - 1))
        paso = max(5, int(round(centimetros / 5) * 5))
        return {"estado": "cerca" if error > 0 else "lejos", "ok": False,
                "instruccion": f"Aléjese unos {paso} cm" if error > 0 else f"Acérquese unos {paso} cm",
                "detalle": f"se ve un {abs(error) * 100:.0f} % más {'grande' if error > 0 else 'pequeño'} "
                           "que en las muestras de entrenamiento"}
    if abs(m["centro_x"] - objetivo.centro_x) > objetivo.tolerancia_x:
        return {"estado": "descentrado", "instruccion": "Ubíquese en el centro de la imagen", "ok": False,
                "detalle": ""}
    arriba, abajo = m["hombros_y"] / m["hombros"], (1.0 - m["hombros_y"]) / m["hombros"]
    if arriba < objetivo.holgura_arriba * relajo or abajo < objetivo.holgura_abajo * relajo:
        # ¿Se arregla apuntando la cámara a otro sitio, o la seña entera no cabe a esta distancia?
        if (objetivo.holgura_arriba + objetivo.holgura_abajo) * m["hombros"] > 1.0:
            return {"estado": "manos_fuera", "instruccion": "Aléjese: las manos se salen de la imagen", "ok": False,
                    "detalle": "la seña no cabe completa a esta distancia"}
        falta_abajo = abajo < objetivo.holgura_abajo * relajo
        return {"estado": "manos_fuera", "ok": False,
                "instruccion": "Incline la cámara hacia abajo" if falta_abajo else "Incline la cámara hacia arriba",
                "detalle": ("las manos se salen por abajo al hacer la seña" if falta_abajo else
                            "las manos se salen por arriba al hacer la seña")}
    return {"estado": "listo", "instruccion": "Encuadre correcto", "detalle": "", "ok": True}


def evaluar_grabacion(datos, objetivo: ObjetivoEncuadre = OBJETIVO):
    """Evalúa el encuadre de una grabación completa con la mediana de sus fotogramas.

    datos: {"pose" (T, 33, >=2), "ancho", "alto", y opcionalmente "rostro" (T, 3, >=2) con ojos y nariz}.
    Devuelve las medidas (medianas) junto con la evaluación; añade las holguras medidas."""
    pose = np.asarray(datos["pose"])
    rostro = datos.get("rostro")
    aspecto = float(datos["ancho"]) / float(datos["alto"])
    por_frame = [medir(pose[i], None if rostro is None else rostro[i], aspecto) for i in range(len(pose))]
    con_pose = [m for m in por_frame if m["con_pose"]]
    if not con_pose:
        return {"con_pose": False, "con_cara": False, **evaluar({"con_pose": False}, objetivo)}
    con_cara = [m for m in con_pose if m["con_cara"]]
    base = con_cara if len(con_cara) >= max(3, len(con_pose) // 4) else con_pose
    mediana = lambda k, lista: float(np.nanmedian([m[k] for m in lista]))
    m = {"con_pose": True, "con_cara": base is con_cara, "ojos": mediana("ojos", base),
         "hombros": mediana("hombros", con_pose), "hombros_y": mediana("hombros_y", con_pose),
         "centro_x": mediana("centro_x", con_pose),
         "desvio_nariz": mediana("desvio_nariz", con_cara) if base is con_cara else float("nan")}
    m["holgura_arriba"] = m["hombros_y"] / m["hombros"]
    m["holgura_abajo"] = (1.0 - m["hombros_y"]) / m["hombros"]
    return {**m, **evaluar(m, objetivo)}
