# Tablero de triaje (Streamlit)

Prototipo web del detector: se sube la foto de un auto y devuelve la probabilidad de que alguna zona haya sido editada con IA, el color del semáforo con la acción sugerida y un mapa de calor (Grad-CAM) con la zona sospechosa.

| Semáforo | Probabilidad | Acción |
| --- | --- | --- |
| 🟢 Verde | < 40 % | Sigue el circuito habitual |
| 🟡 Amarillo | 40 % – 70 % | Revisión de un perito con el mapa de calor |
| 🔴 Rojo | ≥ 70 % | Peritaje presencial o videoperitaje |

La lógica de predicción está en [`inferencia.py`](inferencia.py) y es la misma que `analizar_foto` del notebook 04: la foto se lleva a 768×512, se analiza en 6 recortes de 256×256 sin redimensionar y vale el recorte más sospechoso.

## El modelo no está en el repositorio

El notebook 04 guarda en `04_modelo/` del Drive dos archivos: `config.json` y `mejor_modelo.pt`. La app los busca en este orden:

1. En la carpeta `app/modelo/` (ignorada por git).
2. En un repositorio **privado** de modelos de Hugging Face, indicado con la variable `MODELO_HF_REPO` y el token de lectura en `HF_TOKEN`.

Así el código puede ser público sin publicar el modelo entrenado.

## Correrla en la compu

```bash
pip install -r app/requirements.txt
# copiar config.json y mejor_modelo.pt a app/modelo/
streamlit run app/app.py
```

Se abre en el navegador, en `http://localhost:8501`.

## Publicarla en Hugging Face Spaces

1. Crear un repositorio de **modelo privado** en Hugging Face y subirle `config.json` y `mejor_modelo.pt`.
2. Crear un **Space** con Streamlit y subirle `app.py`, `inferencia.py` y `requirements.txt`.
3. En la configuración del Space, agregar los secretos `MODELO_HF_REPO` (por ejemplo `usuario/detector-danos`) y `HF_TOKEN` (un token de solo lectura).

El Space gratuito no tiene GPU: cada foto tarda unos segundos, alcanza para una demo.

## Cuidados

- **Las fotos no se guardan:** la app las analiza en memoria y no las escribe en disco ni en registros; se descartan al terminar la sesión. Si la app corre en un servidor externo (por ejemplo Hugging Face), la foto igual viaja hasta ese servidor mientras se analiza.

- Es un prototipo de estudio entrenado con fotos públicas: los umbrales no están calibrados para uso real.
- El modelo **no rechaza siniestros**: prioriza qué fotos revisar.
- No usar la versión pública con fotos de siniestros reales: contienen datos personales (patentes, caras). Para eso, correrla en un entorno interno.
