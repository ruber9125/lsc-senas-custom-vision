@echo off
chcp 65001 >nul
title LSC Atiende
cd /d "%~dp0"

if not exist ".streamlit\secrets.toml" (
    echo.
    echo  FALTA LA CONFIGURACION DE AZURE.
    echo  1. En la carpeta .streamlit copia "secrets.toml.ejemplo" y renombralo "secrets.toml"
    echo  2. Abrelo con el Bloc de notas y pega la URL y la clave de tu modelo.
    echo.
    pause
    exit /b
)

rem 1) Crear el entorno de Python si no existe
if not exist ".venv\Scripts\python.exe" (
    echo Creando el entorno de Python...
    py -m venv .venv || python -m venv .venv
)

rem 2) Instalar las librerias si falta alguna (se revisa cada vez; si ya estan, no reinstala nada)
".venv\Scripts\python.exe" -c "import streamlit, mediapipe, cv2, sklearn, matplotlib, pandas, requests, plotly" 2>nul
if errorlevel 1 (
    echo.
    echo Instalando las librerias de la app. La primera vez tarda varios minutos, no cierres esta ventana...
    echo.
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo  ERROR instalando las librerias. Copia los mensajes de arriba y envialos.
        pause
        exit /b
    )
)

echo.
echo Abriendo la aplicacion en el navegador...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
