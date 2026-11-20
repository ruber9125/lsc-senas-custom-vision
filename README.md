# LSC Atiende

**Reconocimiento de señas para la atención accesible al ciudadano sordo.**

Prototipo de un punto de atención accesible para personas sordas en entidades públicas. Piloto propuesto en la
Secretaría de Educación de Valledupar, con réplica en las Secretarías de Educación del país.

Autor: **Ing. Rubert Eduardo Quintero Orozco** — Maestría en Inteligencia Artificial, Universidad de La Salle.

## Propósito

La persona sorda se comunica con la ventanilla en **Lengua de Señas Colombiana (LSC)** en dos momentos de la atención:

| Seña | Uso en la atención | Mensaje en pantalla |
|---|---|---|
| **HOLA** | Iniciar la atención | "Una persona sorda solicita atención" |
| **BIEN** | Calificar el servicio: bien | "Calificación del servicio: BIEN" |
| **MAL** | Calificar el servicio: mal | "Calificación del servicio: MAL" |

El sistema **no reemplaza al intérprete**: complementa al Centro de Relevo (MinTIC–Fenascol), al que se remite lo que
el sistema no reconoce. Si la confianza de una predicción es menor al 60 %, la app sugiere repetir la seña o solicitar
interpretación en línea.

## Cómo se usa

1. En la pestaña **Atención en ventanilla**, elegir **Grabar con la cámara** o **Cargar un video**.
2. Con la cámara: pulsar **Activar cámara** y luego **Grabar seña** (cuenta regresiva de 5 segundos y 4 segundos de grabación).
3. La persona hace **una sola seña**, de frente, de la cintura hacia arriba y con buena luz, empezando y terminando
   con las manos abajo.
4. La app muestra la seña reconocida, la confianza y el mensaje para el funcionario.
5. Para evaluar el modelo, se confirma cuál seña se hizo realmente ("Era HOLA / Era BIEN / Era MAL").

La pestaña **Cómo funciona** explica el sistema en lenguaje claro para el funcionario.

## Qué pasa por dentro

```
Video  →  MediaPipe Holistic  →  Imagen de trayectoria 224×224  →  Azure Custom Vision (endpoint)  →  Seña
          (cuerpo + manos)        (azul = inicio, rojo = final)      (modelo lsc-senas-v1)
```

La evaluación es **a ciegas**: el modelo predice primero y la confirmación se pide después. Con esas confirmaciones
la app calcula en la sesión la exactitud, la precisión, el recall, el F1, la matriz de confusión y las curvas ROC/AUC,
lo que permite medir el desempeño con personas que no participaron en el entrenamiento.

**Alcance actual:** tres señas y una seña por video. **Próximas fases:** vocabulario de trámites (matrícula,
certificado, traslado, cita, sí, no, gracias, ayuda…) y frases cortas.

## Estructura

| Archivo | Función |
|---|---|
| `app.py` | Interfaz Streamlit: atención en ventanilla, evaluación del modelo en la sesión y explicación |
| `interfaz/` | Presentación: estilos (`estilos.py`), componentes visuales (`componentes.py`) y gráficos Plotly (`graficos.py`) |
| `interfaz/grabador.py` | Lado de Python del grabador: recibe el video de la cámara y lo entrega al mismo flujo que los videos cargados |
| `componentes_web/grabador/index.html` | Grabador con la cámara (componente web propio, sin paquetes nuevos): guía de encuadre, cuenta regresiva de 5 segundos y 4 segundos de grabación |
| `lsc/extraccion.py` | Video → puntos clave con MediaPipe Holistic (mismo modelo y umbrales que la captura de los datos) |
| `lsc/recorte.py` | Lleva las grabaciones con la cámara a los parámetros de las muestras de entrenamiento (solo la seña, al mismo ritmo de fotogramas) y compara cada captura con esas muestras: duración, tamaño y posición de la persona, mano detectada |
| `lsc/encuadre.py` | Criterio de encuadre del sistema de captura con el que se grabaron las muestras: distancia a la cámara medida con los ojos, persona centrada y de frente, y espacio para que quepa la seña |
| `lsc/segmentacion.py` | Segmentador con el que se cortaron las muestras (velocidad y altura de las muñecas): se usa para dejar solo la seña en las grabaciones con la cámara |
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
- El video se usa solo para el análisis y no se guarda. A Azure solo se envía la imagen de trayectoria, que no
  contiene rostros.
- El repositorio no contiene videos, puntos clave ni fotos de los señantes.
