# Reconocimiento de señas LSC (HOLA · BIEN · MAL) con Azure Custom Vision

Aplicación web del **Proyecto Final (Actividad 6) de Visión por Computador** — Maestría en Inteligencia Artificial, Universidad de La Salle.

Autor: **Ing. Rubert Eduardo Quintero Orozco**

El usuario **graba con la cámara** del dispositivo (computador, celular o tableta) o **carga** un video corto con
**una seña** de la Lengua de Señas Colombiana, y la app la reconoce automáticamente.

## Flujo

```
Video (.mp4)  →  MediaPipe Holistic  →  Imagen de trayectoria 224×224  →  Azure Custom Vision (endpoint)  →  Seña
               (cuerpo + manos)        (azul = inicio, rojo = final)      (modelo lsc-senas-v1)
```

La evaluación es **a ciegas**: el modelo predice primero y, después, la persona confirma cuál seña hizo
("Era HOLA / Era BIEN / Era MAL"). Con esas confirmaciones la app calcula en la sesión la exactitud, la precisión,
el recall, el F1, la matriz de confusión y las curvas ROC/AUC.

## Estructura

| Archivo | Función |
|---|---|
| `app.py` | Interfaz Streamlit: reconocer una seña, métricas de la sesión y explicación |
| `interfaz/` | Presentación: estilos (`estilos.py`), componentes visuales (`componentes.py`) y gráficos Plotly (`graficos.py`) |
| `interfaz/grabador.py` | Lado de Python del grabador: recibe el video de la cámara y lo entrega al mismo flujo que los videos cargados |
| `componentes_web/grabador/index.html` | Grabador con la cámara (componente web propio, sin paquetes nuevos): cuenta regresiva 3-2-1 y 4 segundos de grabación |
| `lsc/extraccion.py` | Video → puntos clave con MediaPipe Holistic (mismo modelo y umbrales que la captura de los datos) |
| `lsc/trayectoria.py` | Puntos clave → imagen de trayectoria. **Idéntica** píxel a píxel a la usada en el entrenamiento |
| `lsc/azure_cv.py` | Envía la imagen al endpoint de Azure (variante `nostore`: Azure no guarda las imágenes) |
| `models/holistic_landmarker.task` | Modelo de MediaPipe Holistic (Google) |
| `assets/` | Logo institucional e ilustración del flujo (imagen generada, sin personas reales) |
| `.streamlit/config.toml` | Tema visual (paleta institucional) |
| `.streamlit/secrets.toml.ejemplo` | Plantilla de configuración, **sin datos reales** |
| `requirements.txt` | Librerías de Python |
| `packages.txt` | Librerías del sistema que se instalan en Streamlit Community Cloud (Linux) |
| `iniciar_app.bat` | Ejecuta la app en Windows con doble clic |

## Ejecutar en el computador (Windows)

1. En `.streamlit/`, copiar `secrets.toml.ejemplo` como `secrets.toml` y pegar la URL y la clave del modelo.
2. Doble clic en `iniciar_app.bat` (la primera vez instala las librerías).

## Despliegue en Streamlit Community Cloud

1. **Repositorio.** Subir esta carpeta a un repositorio de GitHub, con `app.py` en la raíz.
   `secrets.toml` y `credenciales.txt` no se suben: están en `.gitignore`.
2. **Crear la app.** En share.streamlit.io → *Create app* → elegir el repositorio, la rama `main` y `app.py`
   como archivo principal.
3. **Versión de Python.** En *Advanced settings* elegir **Python 3.12** (3.13 también sirve). Todas las librerías
   de `requirements.txt` tienen paquetes para Linux en esas versiones.
4. **Secrets.** En *Advanced settings → Secrets* pegar, con los valores reales:

   ```toml
   AZURE_PREDICTION_URL = "..."
   AZURE_PREDICTION_KEY = "..."
   APP_PASSWORD = "..."        # opcional
   ```

5. **Librerías del sistema.** `packages.txt` instala las librerías gráficas que faltan en un servidor Linux sin
   escritorio:
   - `libgl1` y `libglib2.0-0`: las necesita OpenCV (dependencia de MediaPipe).
   - `libegl1` y `libgles2`: las necesita MediaPipe (`libEGL.so.1` y `libGLESv2.so.2`).

   Sin ellas la app falla con `libGL.so.1` o `libEGL.so.1: cannot open shared object file`.

### Contraseña de acceso (opcional)

Si en los secrets existe `APP_PASSWORD` con algún valor, la app pide esa contraseña antes de mostrar nada.
Si no existe o está vacía, la app funciona sin contraseña. Se recomienda usarla en el despliegue público:
cada predicción consume el recurso de Azure, que es de pago.

## Seguridad y privacidad

- La URL y la clave de Azure nunca están en el código ni en el repositorio: se leen de los *secrets* de Streamlit.
- Las consultas usan `.../image/nostore`: Azure no almacena las imágenes enviadas.
- A Azure solo se envía la imagen de trayectoria, nunca el video.
- El repositorio no contiene videos, puntos clave ni fotos de los señantes.
