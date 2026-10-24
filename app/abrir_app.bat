@echo off
REM Abre el tablero de triaje en el navegador. Funciona con doble clic.
REM La primera vez instala las librerias que falten (puede tardar unos minutos).
cd /d "%~dp0"
title Detector de danos sinteticos

REM ---------- Activar Anaconda ----------
set "ACTIVAR="
for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "%LOCALAPPDATA%\anaconda3") do (
    if not defined ACTIVAR if exist "%%~D\Scripts\activate.bat" set "ACTIVAR=%%~D\Scripts\activate.bat"
)
if defined ACTIVAR (
    call "%ACTIVAR%"
) else (
    where python >nul 2>nul || (
        echo No encontre Anaconda ni Python. Instala Anaconda o abri este archivo desde el Anaconda Prompt.
        pause
        exit /b 1
    )
)

REM ---------- Modelo ----------
if not exist "modelo" mkdir "modelo"
if not exist "modelo\mejor_modelo.pt" if exist "%USERPROFILE%\Downloads\mejor_modelo.pt" (
    echo Moviendo mejor_modelo.pt desde Descargas a la carpeta modelo...
    move "%USERPROFILE%\Downloads\mejor_modelo.pt" "modelo\mejor_modelo.pt" >nul
)
if not exist "modelo\config.json" if exist "%USERPROFILE%\Downloads\config.json" (
    copy "%USERPROFILE%\Downloads\config.json" "modelo\config.json" >nul
)
if not exist "modelo\mejor_modelo.pt" (
    echo.
    echo Falta el modelo. Copia config.json y mejor_modelo.pt de la carpeta 04_modelo_v2 del Drive
    echo a la carpeta modelo, al lado de este archivo, y volve a ejecutar.
    echo.
    pause
    exit /b 1
)

REM ---------- Librerias (solo la primera vez) ----------
python -c "import streamlit, timm, pytorch_grad_cam, torch" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Primera vez: instalando las librerias necesarias. Puede tardar varios minutos...
    echo.
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo No se pudieron instalar las librerias. Mandale una captura de esta ventana a Claude.
        pause
        exit /b 1
    )
)

REM ---------- Abrir la app ----------
echo.
echo Abriendo la app en el navegador. Para cerrarla, cerra esta ventana.
echo.
python -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
pause
