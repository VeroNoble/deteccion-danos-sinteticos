@echo off
REM Abre el tablero de triaje en el navegador.
REM Ejecutar desde el "Anaconda Prompt" (o con doble clic si Python esta en el PATH).
cd /d "%~dp0"
if not exist "modelo\mejor_modelo.pt" (
    echo.
    echo Falta el modelo. Copia config.json y mejor_modelo.pt de la carpeta 04_modelo_v2 del Drive
    echo a la carpeta app\modelo\ y volve a ejecutar.
    echo.
    pause
    exit /b 1
)
python -m streamlit run app.py
pause
