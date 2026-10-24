"""Tablero de triaje: ¿esta foto de un auto fue editada con IA?

Uso local:   streamlit run app/app.py
Modelo:      app/modelo/config.json + app/modelo/mejor_modelo.pt (salida del notebook 04, carpeta 04_modelo)
             o, si no están, se descargan de un repo PRIVADO de Hugging Face indicado en MODELO_HF_REPO
             (con el token en HF_TOKEN). Así el código puede ser público y el modelo no.
"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from inferencia import ACCIONES, Detector, superponer

CARPETA_LOCAL = Path(__file__).parent / "modelo"
def pct(v):
    """Porcentaje truncado (69,96 % se muestra 69 %), para que coincida con los cortes del semáforo."""
    return f"{int(v * 100)}%"


COLORES = {"verde": "#1e8e3e", "amarillo": "#e3a008", "rojo": "#d93025"}
NOMBRES = {"verde": "VERDE", "amarillo": "AMARILLO", "rojo": "ROJO"}

st.set_page_config(page_title="Detector de daños sintéticos", page_icon="🚗", layout="wide")


@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar_detector():
    if (CARPETA_LOCAL / "config.json").exists():
        return Detector(CARPETA_LOCAL)
    repo = os.environ.get("MODELO_HF_REPO")
    if repo:
        from huggingface_hub import snapshot_download
        carpeta = snapshot_download(repo_id=repo, token=os.environ.get("HF_TOKEN"),
                                    allow_patterns=["config.json", "*.pt"])
        return Detector(carpeta)
    return None


st.title("Detector de daños sintéticos")
st.write("Subí la foto de un auto y el modelo estima si alguna zona fue **editada con IA**, "
         "por ejemplo un golpe agregado por *inpainting*. El mapa de calor marca la zona que le resultó sospechosa.")

detector = cargar_detector()
if detector is None:
    st.error("No se encontró el modelo. Copiá `config.json` y `mejor_modelo.pt` (carpeta `04_modelo` del Drive) "
             "a `app/modelo/`, o configurá `MODELO_HF_REPO` y `HF_TOKEN`.")
    st.stop()

with st.sidebar:
    st.header("Cómo leer el resultado")
    st.markdown(
        f"- **Verde** (< {detector.cfg['corte_amarillo']:.0%}): {ACCIONES['verde']}\n"
        f"- **Amarillo** ({detector.cfg['corte_amarillo']:.0%} a {detector.cfg['corte_rojo']:.0%}): {ACCIONES['amarillo']}\n"
        f"- **Rojo** (≥ {detector.cfg['corte_rojo']:.0%}): {ACCIONES['rojo']}"
    )
    st.caption("El modelo **no rechaza siniestros**: prioriza qué fotos revisar. Es un prototipo de estudio "
               "entrenado con fotos públicas; sus umbrales no están calibrados para uso real.")
    st.caption("No subas fotos con datos personales de terceros (patentes, caras) ni fotos de siniestros reales.")

archivo = st.file_uploader("Foto del auto", type=["jpg", "jpeg", "png", "webp"])
if archivo is None:
    st.info("Esperando una foto. Funciona mejor con fotos apaisadas donde el auto se vea entero.")
    st.stop()

with st.spinner("Analizando…"):
    res = detector.analizar(archivo)

p, color = res["prob_manipulada"], res["semaforo"]
st.markdown(
    f"<div style='padding:14px 18px;border-radius:10px;background:{COLORES[color]};color:white;"
    f"font-size:1.15rem;margin:6px 0 14px'><b>{NOMBRES[color]}</b> · probabilidad de manipulación "
    f"<b>{pct(p)}</b> · {res['accion']}</div>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
col1.image(res["imagen"], caption="Foto analizada (recorte 3:2, 768×512)", use_container_width=True)
col2.image(superponer(res["imagen"], res["mapa"]), caption="Mapa de calor: rojo = zona más sospechosa",
           use_container_width=True)

with st.expander("Detalle por zona"):
    st.write("La foto se analiza en 6 zonas de 256×256 sin achicarla, para no perder el detalle fino. "
             "La probabilidad de la foto es la de la zona más sospechosa.")
    zonas = res["prob_por_zona"]
    st.table(pd.DataFrame([[pct(v) for v in fila] for fila in zonas],
                          index=["Arriba", "Abajo"][:zonas.shape[0]],
                          columns=["Izquierda", "Centro", "Derecha"][:zonas.shape[1]]))

st.caption("Si la foto no tenía proporción 3:2 o era de otro tamaño, se recortó al centro y se redimensionó: "
           "eso puede alterar el detalle fino que el modelo analiza.")
