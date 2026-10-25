# Detección de daños sintéticos en fotos de siniestros automotores

Proyecto personal de visión por computadora que detecta fotos de autos manipuladas con IA generativa, para ayudar a priorizar la revisión de siniestros. El modelo estima la probabilidad de que una foto haya sido editada, la clasifica en un semáforo y marca la zona sospechosa con un mapa de calor. Corre en una computadora común, sin que las fotos salgan a internet.

![Foto original, foto con un golpe agregado por IA y mapa de calor del modelo](docs/img/ejemplo_mapa_calor.jpg)

**Estado:** prototipo funcional (modelo v2 entrenado y app local). Detecta bien las ediciones de los generadores con los que se entrenó. Con herramientas nuevas, como Gemini o ChatGPT, hay señal, pero también reacciona al cambio de tamaño de la foto (ver [Prueba con Gemini y ChatGPT](#prueba-con-gemini-y-chatgpt)).

## El problema

Las herramientas de IA generativa permiten tomar una foto real de un auto y agregarle, con pocos clics, una abolladura, una raspadura o un paragolpes roto (*inpainting*). En un seguro automotor, eso abre la puerta a reclamar daños que no existen.

Este tipo de fraude rara vez usa autos generados al 100 %: lo habitual es **editar una zona puntual de una foto auténtica**. Por eso el proyecto no intenta reconocer "la forma de un choque", que en la realidad es caótica, sino que lo plantea como un **análisis forense de consistencia**: ¿los píxeles de la zona dañada comparten la misma historia física (ruido del sensor, compresión, textura) que el resto de la foto?

## Resultados (modelo v2)

Evaluado con **141 fotos que el modelo nunca vio** (41 auténticas y 100 manipuladas), separadas por foto de origen para que ninguna versión de un mismo auto quede en entrenamiento y en test a la vez.

| Métrica | Valor | Qué significa |
| --- | --- | --- |
| Recall (manipuladas) | **83 %** | De cada 100 fotos editadas, detecta 83 |
| Precisión | **81 %** | De cada 100 alertas, 81 son correctas |
| AUC ROC | **0,78** | Ordena bien de más a menos sospechosa (0,5 = azar) |
| AUC precision-recall | 0,91 | |
| Exactitud | 74 % | |
| AUC con compresión JPEG calidad 30 | 0,69 | Robustez a fotos muy recomprimidas, como las de WhatsApp |

**Matriz de confusión (corte 0,5)**

| | Predijo auténtica | Predijo manipulada |
| --- | --- | --- |
| Auténtica (41) | 21 | 20 |
| Manipulada (100) | 17 | 83 |

**Acierto por tipo de foto**

| Tipo | Acierto |
| --- | --- |
| Auto 100 % generado por IA | 100 % |
| Golpe agregado por IA (`inpaint_danio`) | 76 % |
| Panel limpio repintado por IA (`inpaint_control`) | 74 % |
| Real sin daño | 54 % |
| Real con daño verdadero (n = 6) | 33 % |

El modelo detecta casi igual la edición con daño que la edición sin daño: aprendió a reconocer **la edición**, no el golpe. El mapa de calor (Grad-CAM) se concentra unas **7 veces más** en la zona editada que en el resto de la foto.

**Corrección de falsas alarmas (v1 → v2).** En la primera versión, el 71 % de las fotos reales sin daño daba alerta. Durante el entrenamiento, el recorte de cada foto real caía siempre en la misma zona que la edición de su "hermana" manipulada, y el modelo aprendía a desconfiar de esas zonas. En la v2, la mitad de esos recortes se toma al azar (`GUIADO_REALES = 0.5`). Las falsas alarmas bajaron al 46 % y el AUC subió de 0,75 a 0,78.

## Prueba con Gemini y ChatGPT

Para medir el rendimiento con herramientas que el modelo no vio, se tomaron las 19 fotos reales sin daño del test que el modelo clasificaba en verde y se les agregó un golpe con Gemini (Nano Banana) y ChatGPT.

![Fotos editadas con Gemini y mapa de calor del modelo](docs/img/prueba_gemini.jpg)

| Qué se analizó | Verde | Amarillo | Rojo |
| --- | --- | --- | --- |
| Originales (19) | 18 | 1 | 0 |
| Editadas con Gemini (18) | 3 | 1 | 14 |
| Editadas con ChatGPT (2) | 0 | 0 | 2 |
| Originales sin editar, solo agrandadas y vueltas a achicar (19) | 11 | 5 | 3 |

A primera vista detecta 16 de 20 ediciones. Pero estas herramientas devuelven la imagen más grande (2528×1686 o 1536×1024), y llevarla de nuevo a 768×512 alcanza para que **8 de 19 fotos reales sin editar den alerta**. Comparada con su original del mismo tamaño, la editada sube claramente en 13 de 18 casos: hay señal, pero hoy el modelo también reacciona al reescalado. Es lo primero a corregir en la próxima versión.

## Diseño del dataset

El dataset final tiene **968 imágenes**: 282 auténticas y 686 manipuladas. Se genera de forma automática a partir de fotos reales de autos de [Wikimedia Commons](https://commons.wikimedia.org/) con licencias libres (ver [`data/atribuciones.csv`](data/atribuciones.csv)). El diseño apunta a que el modelo aprenda a detectar **la edición**, no **el daño**:

| Tipo | Etiqueta | Cantidad | Para qué sirve |
| --- | --- | --- | --- |
| `real_sano`: foto real sin daño | Auténtica (0) | 244 | Caso base |
| `real_danado`: foto real con daño verdadero | Auténtica (0) | 38 | Control: un daño real **no** debe disparar la alarma |
| `inpaint_danio`: foto real con daño agregado por IA | Manipulada (1) | 243 | El caso de fraude a detectar |
| `inpaint_control`: foto real con un panel *limpio* repintado por IA | Manipulada (1) | 243 | Control: la edición se detecta aunque no haya daño |
| `ia_sano` / `ia_danio`: auto generado completamente por IA | Manipulada (1) | 200 | Imágenes 100 % sintéticas |

Otras decisiones del pipeline:

- **Tres modelos generativos distintos** (Stable Diffusion 1.5 Inpainting, SDXL Inpainting y SDXL Turbo), para que el modelo no aprenda las huellas de una sola herramienta. Las imágenes 100 % sintéticas usan modelos de autos del mercado argentino.
- **Mismo formato para todas** (768×512 px) y **recompresión JPEG con calidad aleatoria (75–95)**, para que el tamaño o la compresión no delaten la clase.
- **Trazabilidad completa:** cada imagen manipulada registra su foto de origen (`base_id`), la zona editada, el porcentaje cubierto por la máscara, el prompt, la semilla, el modelo y sus parámetros.
- **Detector de atajos:** antes de entrenar, una regresión logística intenta adivinar la clase usando solo datos del archivo (calidad JPEG, peso), con validación cruzada agrupada por foto base. En el dataset final da AUC 0,46 con la compresión y 0,58 con el peso: no hay atajos importantes.
- **Partición sin fuga de datos:** 680 imágenes de entrenamiento, 147 de validación y 141 de test, agrupadas por `base_id`.

Los manifiestos están en [`data/`](data/). Las imágenes no se incluyen en el repositorio.

## Notebooks

| Notebook | Qué hace |
| --- | --- |
| [`01_fotos_reales.ipynb`](notebooks/01_fotos_reales.ipynb) | Descarga fotos reales de autos (sanos y con daño) desde Wikimedia Commons, registra autor y licencia, y las lleva a un formato común |
| [`02_fotos_falsas.ipynb`](notebooks/02_fotos_falsas.ipynb) | Genera las imágenes manipuladas con modelos de difusión: daño agregado, controles sin daño y autos 100 % sintéticos |
| [`03_unificar_dataset.ipynb`](notebooks/03_unificar_dataset.ipynb) | Aplica la misma compresión JPEG a todas, arma el dataset final y corre el detector de atajos |
| [`04_entrenamiento_convnext.ipynb`](notebooks/04_entrenamiento_convnext.ipynb) | Entrena ConvNeXt-Tiny con recortes en resolución original y lo evalúa: acierto por tipo y por generador, robustez a la compresión JPEG, Grad-CAM contra las máscaras de edición y semáforo |

Los notebooks se desarrollaron en Google Colab (GPU T4).

## Arquitectura del modelo

- **Modelo:** ConvNeXt-Tiny preentrenado en ImageNet (PyTorch + `timm`), con *fine-tuning* para clasificación binaria auténtica / manipulada.
- **Entrada por recortes:** el modelo ve recortes de 256×256 **sin achicar** la foto, para no borrar las huellas finas de la edición. En evaluación, la foto se divide en 6 recortes y vale el más sospechoso.
- **Entrenamiento:** AdamW, *cosine schedule*, pesos por clase y *early stopping* según el AUC de validación (mejor época: 16).
- **Aumento de datos:** volteo, cambios de luz y compresión JPEG entre 40 y 95 para simular fotos enviadas por WhatsApp.
- **Explicabilidad:** Grad-CAM, para mostrarle al perito qué zona de la foto marcó el modelo. Como se conoce la máscara de cada edición, se mide si el mapa de calor apunta a la zona editada.

El detalle técnico está en [`docs/especificacion_tecnica.md`](docs/especificacion_tecnica.md).

## App de triaje

En [`app/`](app/) hay una app en Streamlit: se sube una foto y devuelve la probabilidad, el semáforo y el mapa de calor. **Corre en la propia computadora (`localhost`) y procesa las fotos en memoria, sin guardarlas ni enviarlas a ningún servidor.** Las instrucciones de instalación están en [`app/README.md`](app/README.md).

| Semáforo | Probabilidad de manipulación | Acción sugerida |
| --- | --- | --- |
| 🟢 Verde | < 40 % | El siniestro sigue su circuito habitual |
| 🟡 Amarillo | 40 % – 70 % | Revisión de un perito con el mapa de calor |
| 🔴 Rojo | ≥ 70 % | Peritaje presencial o videoperitaje |

El modelo **no rechaza siniestros**: prioriza la revisión humana.

## Limitaciones

- **Fotos reales con daño:** el modelo desconfía de varias (4 de 6 en el test dieron rojo), porque vio pocos ejemplos (38).
- **Cambio de tamaño:** agrandar o achicar una foto real puede disparar una alerta. Las fotos de celular siempre se achican, así que esto daría falsas alarmas en la práctica.
- **Test chico:** con 141 fotos, los porcentajes pueden moverse varios puntos.
- **Sobreajuste:** la pérdida de entrenamiento baja mucho más que la de validación.
- **Fotos públicas:** está entrenado con fotos de Commons, no con fotos reales de siniestros.

## Hoja de ruta

- [x] Especificación técnica del sistema
- [x] Pipeline de generación de imágenes manipuladas con controles
- [x] Dataset escalado (968 imágenes) con trazabilidad y detector de atajos
- [x] Entrenamiento y evaluación de ConvNeXt-Tiny (v2: AUC 0,78)
- [x] Corrección de falsas alarmas en fotos reales (71 % → 46 %)
- [x] App de triaje en Streamlit con el modelo entrenado
- [x] Prueba con generadores no vistos (Gemini y ChatGPT)
- [ ] Entrenar con fotos reales reescaladas, para que el cambio de tamaño deje de parecer fraude
- [ ] Sumar ediciones de herramientas actuales al entrenamiento
- [ ] Más fotos reales con daño, incluidas fotos de celular
- [ ] Calibrar los cortes del semáforo y comparar con un detector de referencia (TruFor)

## Herramientas

**Pipeline de datos:** Python · PyTorch · diffusers (Stable Diffusion) · OpenCV · Pillow · pandas · scikit-learn

**Modelo y app:** timm (ConvNeXt) · albumentations · pytorch-grad-cam · Streamlit

## Nota sobre los datos

El proyecto usa únicamente imágenes de Wikimedia Commons con licencia libre. No incluye fotos ni datos de siniestros reales. Las imágenes de ejemplo de este README son fotos de Commons y versiones editadas de ellas: [Hyundai i20](https://commons.wikimedia.org/wiki/File:Hyundai_i20_Erlk%C3%B6nig_01_(cropped).jpg) (LSDSL, CC BY-SA 3.0), [Volkswagen Golf](https://commons.wikimedia.org/wiki/File:2018_Volkswagen_Golf_R-Line_TSi_EVO_S-A_1.5.jpg) (Vauxford, CC BY-SA 4.0) y [Volkswagen Gol](https://commons.wikimedia.org/wiki/File:Volkswagen_Gol_1.6_trend_2011.jpg) (RL GNZLZ, CC BY-SA 2.0). Las versiones editadas se comparten bajo la misma licencia.
