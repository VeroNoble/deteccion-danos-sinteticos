# Detección de daños sintéticos en fotos de siniestros automotores

> 🚧 **Proyecto en desarrollo.** Etapa actual: construcción y validación del dataset piloto. El entrenamiento del modelo es el próximo paso.

Proyecto personal de visión por computadora para detectar fotos de vehículos manipuladas con IA generativa, con aplicación al triaje de siniestros automotores.

## El problema

Las herramientas de IA generativa permiten tomar una foto real de un auto y agregarle, con pocos clics, una abolladura, una raspadura o un paragolpes roto (*inpainting*). En un seguro automotor, eso abre la puerta a reclamar daños que no existen.

Este tipo de fraude rara vez usa autos generados al 100%: lo habitual es **editar una zona puntual de una foto auténtica**. Por eso el proyecto no intenta reconocer "la forma de un choque", que en la realidad es caótica, sino que lo plantea como un **análisis forense de consistencia**: ¿los píxeles de la zona dañada comparten la misma historia física (ruido del sensor, compresión, iluminación, textura) que el resto de la foto?

## Diseño del dataset

El dataset se genera de forma automática a partir de fotos reales de autos (imágenes de uso libre con atribución y fotos propias). El diseño apunta a que el modelo aprenda a detectar **la edición**, no **el daño**:

| Tipo | Etiqueta | Para qué sirve |
| --- | --- | --- |
| `real_sano` — foto real sin daño | Auténtica (0) | Caso base |
| `real_danado` — foto real con daño verdadero | Auténtica (0) | Control: un daño real **no** debe disparar la alarma |
| `inpaint_danio` — foto real con daño agregado por IA | Manipulada (1) | El caso de fraude a detectar |
| `inpaint_control` — foto real con un panel *limpio* repintado por IA | Manipulada (1) | Control: la edición se detecta aunque no haya daño |
| `ia_sano` / `ia_danio` — auto generado completamente por IA | Manipulada (1) | Imágenes 100% sintéticas |

Otras decisiones del pipeline:

- **Tres modelos generativos distintos** (Stable Diffusion 1.5 Inpainting, SDXL Inpainting y SDXL Turbo), para que el modelo no aprenda las huellas de una sola herramienta. Las imágenes 100% sintéticas usan modelos de autos del mercado argentino.
- **Recompresión JPEG con calidad aleatoria (75–95)** de todas las imágenes, auténticas y manipuladas, para que el formato del archivo no funcione como atajo.
- **Trazabilidad completa:** cada imagen manipulada registra su foto de origen (`base_id`), la zona editada, el porcentaje de la imagen cubierto por la máscara, el prompt, la semilla, el modelo y sus parámetros. Esto hace que el dataset sea reproducible.

El dataset piloto tiene **23 imágenes** (11 auténticas y 12 manipuladas). Los manifiestos están en [`data/`](data/); las imágenes no se incluyen en el repositorio.

## Arquitectura planificada

- **Modelo:** ConvNeXt-Tiny preentrenado (PyTorch + `timm`), clasificación binaria auténtica / manipulada.
- **Aumento de datos:** `albumentations`, incluyendo compresión JPEG agresiva para simular fotos enviadas por WhatsApp.
- **Explicabilidad:** Grad-CAM, para mostrarle al perito qué zona de la foto marcó el modelo.
- **Prototipo de uso:** tablero en Streamlit con un esquema de semáforo.

| Semáforo | Probabilidad de manipulación | Acción |
| --- | --- | --- |
| 🟢 Verde | < 40% | El siniestro sigue su circuito habitual |
| 🟡 Amarillo | 40% – 70% | Revisión de un perito con el mapa de calor |
| 🔴 Rojo | ≥ 70% | Peritaje presencial o videoperitaje |

El modelo **no rechaza siniestros**: prioriza la revisión humana. Los umbrales son iniciales y se van a calibrar con datos de validación y según la capacidad de revisión.

El detalle técnico completo está en [`docs/especificacion_tecnica.md`](docs/especificacion_tecnica.md).

## Decisiones abiertas

- **Resolución de entrada.** La zona editada ocupa entre el 3% y el 8% de la imagen. Redimensionar la foto completa a 256x256 borra justo el ruido fino que el modelo necesita analizar, así que se evalúa trabajar con recortes (*parches*) en la resolución original.
- **Partición sin fuga de datos.** Todas las versiones derivadas de una misma foto base van del mismo lado de la partición (train o test), usando `base_id`.
- **Métricas.** Recall y precision sobre la clase manipulada y curva precision-recall, evaluando por separado imágenes muy comprimidas y de generadores que el modelo no vio durante el entrenamiento.

## Hoja de ruta

- [x] Especificación técnica del sistema
- [x] Pipeline de generación de imágenes manipuladas con controles
- [x] Dataset piloto con trazabilidad completa
- [ ] Escalar el dataset (cientos o miles de imágenes por clase)
- [ ] Entrenamiento y evaluación de ConvNeXt-Tiny
- [ ] Evaluación con un generador no visto en el entrenamiento
- [ ] Grad-CAM y prototipo en Streamlit

## Herramientas

Python · PyTorch · timm · albumentations · OpenCV · Pillow · diffusers (Stable Diffusion) · pytorch-grad-cam · Streamlit

## Nota sobre los datos

El proyecto usa únicamente imágenes de uso libre y fotos propias. No incluye fotos ni datos de siniestros reales.
