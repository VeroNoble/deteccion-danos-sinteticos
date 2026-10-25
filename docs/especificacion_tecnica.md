**Documento de Especificación Técnica  
Sistema Forense de Detección de Daños Sintéticos en Siniestros
Automotores**

# 1. Resumen Ejecutivo y Enfoque del Problema

El fraude automotor mediante herramientas generativas no suele apoyarse
en vehículos generados al 100% desde cero, sino en manipulación
localizada (inpainting): se toma una fotografía real del vehículo
asegurado y se edita digitalmente un panel (óptica, paragolpes,
guardabarros) agregando abolladuras, raspaduras o roturas artificiales.

El enfoque técnico de este desarrollo descarta la clasificación basada
exclusivamente en 'la forma del choque', ya que las deformaciones
mecánicas reales son inherentemente caóticas. En su lugar, el sistema
aborda el problema como un análisis de consistencia forense de señal:
evalúa si los píxeles que componen el área del daño comparten la misma
historia física (grano del sensor, compresión, iluminación y resolución
de textura) que el resto de la carrocería.

# 2. Stack Tecnológico y Herramientas

| **Componente**           | **Herramienta**             | **Rol en el Proyecto**               | **Justificación Técnica**                                                                        |
|--------------------------|-----------------------------|--------------------------------------|--------------------------------------------------------------------------------------------------|
| Lenguaje de Programación | Python 3.10+                | Entorno base                         | Compatibilidad con ecosistemas de computer vision y Deep Learning.                               |
| Framework Deep Learning  | PyTorch                     | Entrenamiento e inferencia           | Control granular sobre gradientes, capas intermedias y tensores.                                 |
| Repositorio de Modelos   | timm (PyTorch Image Models) | Acceso a arquitecturas preentrenadas | Implementaciones estandarizadas de backbones convolucionales y transformers.                     |
| Aumento de Datos         | albumentations              | Robustez frente a compresión         | Soporte nativo para simulación de artefactos de compresión JPEG agresivos (WhatsApp/mensajería). |
| Visión Computacional     | OpenCV (cv2) & Pillow       | Preprocesamiento                     | Lectura, conversión de espacios de color (BGR a RGB) y redimensionamiento.                       |
| Explicabilidad           | pytorch-grad-cam            | Interpretabilidad                    | Generación de mapas de calor para auditoría visual humana.                                       |
| Interfaz de Triaje       | Streamlit                   | Prototipo de validación              | Creación de tablero interactivo para peritos sin sobrecostos de frontend.                        |

# 3. Arquitectura del Modelo y Algoritmos

## 3.1 Backbone Principal: ConvNeXt (Variante Tiny)

Para la extracción de características se adopta ConvNeXt-Tiny, una red
convolucional modernizada con principios de Vision Transformers:  
• Ventaja sobre CNNs clásicas: Emplea convoluciones depthwise de ventana
amplia (7x7), LayerNorm y activaciones GELU, optimizando la captura de
dependencias de textura fina.  
• Ventaja sobre ViTs puros: Mantiene sesgos inductivos espaciales
naturales, lo que facilita el aprendizaje de bordes y micro-ruido sin
requerir millones de imágenes de entrenamiento.

## 3.2 Función de Pérdida y Optimización

• Función de Pérdida: Entropía Cruzada (Cross-Entropy Loss) con salida
Softmax sobre dos clases: Auténtica (0) vs. Sintética / Manipulada (1).  
• Optimizador: AdamW con tasa de aprendizaje base 1e-4 y regularización
por decaimiento de peso (weight decay) de 1e-2 para mitigar el
sobreajuste (overfitting).

## 3.3 Módulo de Explicabilidad: Grad-CAM

Para transformar el modelo de 'caja negra' a un sistema auditable, se
implementa Grad-CAM (Gradient-weighted Class Activation Mapping) sobre
la última etapa convolucional del modelo:  
1. Se calcula el gradiente del score de la clase sintética respecto a
los mapas de activación.  
2. Se calculan los pesos de importancia relativa mediante un promediado
global (Global Average Pooling).  
3. Se genera un mapa de calor proyectando una combinación lineal de las
activaciones ponderadas con ReLU.  
4. La máscara resultante se superpone sobre la imagen original para
colorear de rojo las regiones donde el modelo detectó la manipulación.

# 4. Pipeline de Procesamiento y Flujo de Clasificación

El flujo de ejecución consta de 5 etapas secuenciales:  
1. Ingesta y Preprocesamiento: Redimensionamiento a 256x256 px,
normalización z-score y conversión a tensor (C, H, W).  
2. Extracción de Features: ConvNeXt evalúa patrones de ruido, anomalías
en bordes de fusión y discontinuidades.  
3. Clasificación: Capa densa final que genera los logits y se normaliza
con función Softmax para obtener las probabilidades.  
4. Generación de Explicabilidad: El hook de Grad-CAM extrae las
activaciones espaciales y genera el mapa de calor.  
5. Salida en Dashboard: Streamlit presenta el resultado lado a lado con
el diagnóstico del semáforo.

# 5. Criterios Operativos de Decisión (Triaje de Siniestros)

El sistema opera bajo un esquema de semáforo probabilístico para evitar
rechazos erróneos:  
  
• Bajo Riesgo (P_IA \< 40% - Semáforo Verde): Ruido y transiciones de
textura coherentes con una captura real. El reclamo continúa su curso
habitual de liquidación.  
  
• Riesgo Medio (40% \<= P_IA \< 70% - Semáforo Amarillo): Pérdida de
información en alta frecuencia o compresión mixta sospechosa. Se deriva
a un perito interno para inspección visual del mapa de calor.  
  
• Alto Riesgo (P_IA \>= 70% - Semáforo Rojo): Fuerte correlación con
artefactos de difusión/inpainting. Se interrumpe la liquidación y se
exige peritaje presencial o videoperitaje en vivo.

# 6. Resultados de la Versión 2 (octubre 2026)

Evaluación sobre 141 imágenes de test no vistas (41 auténticas, 100 manipuladas), separadas por foto de origen:

| Métrica | Valor |
| --- | --- |
| AUC ROC | 0,78 |
| AUC precision-recall | 0,91 |
| Recall (manipuladas, corte 0,5) | 83 % |
| Precisión | 81 % |
| AUC con JPEG calidad 30 | 0,69 |

Hallazgos principales:

• Grad-CAM concentra unas 7 veces más activación en la zona editada que en el resto de la imagen.

• Con recortes de las fotos reales tomados la mitad de las veces al azar (`GUIADO_REALES = 0.5`), las falsas alarmas en fotos reales sin daño bajaron del 71 % al 46 %.

• Con ediciones de Gemini y ChatGPT (generadores no vistos), 16 de 20 dieron rojo, pero el reescalado por sí solo hizo que 8 de 19 fotos reales sin editar dieran alerta. El próximo entrenamiento debe incluir fotos reales reescaladas como auténticas.

El detalle completo está en el README del repositorio.
