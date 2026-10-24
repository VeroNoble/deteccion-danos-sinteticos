@echo off
REM Abre el tablero de triaje en el navegador.
REM Ejecutar desde el "Anaconda Prompt" (o con doble clic si Python esta en el PATH).
cd /d "%~dp0"
if not exist "modelo" mkdir "modelo"

REM Si el modelo todavia esta en Descargas, se mueve a la carpeta modelo (solo la primera vez).
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
python -m streamlit run app.py
pause
