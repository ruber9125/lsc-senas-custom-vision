"""
Cliente del endpoint de Azure Custom Vision (modelo publicado como lsc-senas-v1).

La URL y la clave NO van en el código: se leen de los "secrets" de Streamlit
(.streamlit/secrets.toml en local, o el panel Secrets en Streamlit Community Cloud).
"""
import requests


def url_sin_almacenar(url):
    """Custom Vision guarda por defecto cada imagen consultada. La variante /image/nostore no guarda nada:
    más privacidad para quien usa la app y el proyecto no se llena de imágenes."""
    url = url.strip().rstrip("/")
    return url + "/nostore" if url.endswith("/image") else url


def clasificar(png_bytes, url, clave, timeout=30):
    """Envía UNA imagen al modelo. Devuelve {etiqueta: probabilidad}, ordenado de mayor a menor."""
    r = requests.post(
        url_sin_almacenar(url),
        headers={"Prediction-Key": clave, "Content-Type": "application/octet-stream"},
        data=png_bytes,
        timeout=timeout,
    )
    if r.status_code == 401:
        raise RuntimeError("Azure rechazó la clave (401). Revisa AZURE_PREDICTION_KEY.")
    if r.status_code == 404:
        raise RuntimeError("Azure no encontró el modelo (404). Revisa AZURE_PREDICTION_URL y que esté publicado.")
    r.raise_for_status()
    preds = r.json()["predictions"]
    return dict(sorted(((p["tagName"], p["probability"]) for p in preds), key=lambda x: -x[1]))
