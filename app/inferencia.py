"""Inferencia del detector de fotos manipuladas.

Misma lógica que la sección 11 del notebook 04 (`analizar_foto`): la foto se lleva a 768x512, se divide en
6 recortes de 256x256 sin redimensionar, la probabilidad de la foto es la del recorte más sospechoso y el
mapa de calor (Grad-CAM) se arma recorte por recorte, ponderado por la probabilidad de cada uno.
"""
import json
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

ACCIONES = {
    "verde": "Sigue el circuito habitual del siniestro.",
    "amarillo": "Revisión de un perito con el mapa de calor.",
    "rojo": "Peritaje presencial o videoperitaje.",
}


class Detector:
    def __init__(self, carpeta_modelo, dispositivo=None):
        carpeta = Path(carpeta_modelo)
        self.cfg = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
        self.W, self.H = self.cfg["ancho"], self.cfg["alto"]
        self.T = self.cfg["tam_parche"]
        self.media = np.array(self.cfg["media"], dtype=np.float32)
        self.desvio = np.array(self.cfg["desvio"], dtype=np.float32)
        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")

        self.modelo = timm.create_model(self.cfg["nombre_modelo"], pretrained=False, num_classes=2)
        pesos = torch.load(carpeta / self.cfg["archivo_pesos"], map_location="cpu")
        self.modelo.load_state_dict(pesos)
        self.modelo.to(self.dispositivo).eval()
        self.capa_cam = [self.modelo.stages[self.cfg.get("capa_gradcam", -1)]]

    # ---------- preparación ----------
    def preparar(self, imagen):
        """Acepta una ruta o una imagen PIL y devuelve un array 512x768x3 (recorte central 3:2)."""
        img = Image.open(imagen) if not isinstance(imagen, Image.Image) else imagen
        img = ImageOps.exif_transpose(img).convert("RGB")
        if img.size != (self.W, self.H):
            img = ImageOps.fit(img, (self.W, self.H), Image.BICUBIC)
        return np.asarray(img)

    def _recortes(self, arr):
        return [arr[y:y + self.T, x:x + self.T] for y in range(0, self.H, self.T) for x in range(0, self.W, self.T)]

    def _tensor(self, recortes):
        x = np.stack(recortes).astype(np.float32) / 255.0
        x = (x - self.media) / self.desvio
        return torch.from_numpy(x.transpose(0, 3, 1, 2)).float().to(self.dispositivo)

    # ---------- predicción ----------
    @torch.no_grad()
    def prob_recortes(self, arr):
        x = self._tensor(self._recortes(arr))
        return torch.softmax(self.modelo(x).float(), dim=1)[:, 1].cpu().numpy()

    def mapa_calor(self, arr, probs):
        x = self._tensor(self._recortes(arr))
        with GradCAM(model=self.modelo, target_layers=self.capa_cam) as cam:
            mapas = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)] * len(probs))
        mapa = np.zeros((self.H, self.W), dtype=np.float32)
        k = 0
        for y in range(0, self.H, self.T):
            for xx in range(0, self.W, self.T):
                mapa[y:y + self.T, xx:xx + self.T] = mapas[k] * probs[k]
                k += 1
        return mapa

    def semaforo(self, p):
        if p >= self.cfg["corte_rojo"]:
            return "rojo"
        if p >= self.cfg["corte_amarillo"]:
            return "amarillo"
        return "verde"

    def analizar(self, imagen, con_mapa=True):
        arr = self.preparar(imagen)
        probs = self.prob_recortes(arr)
        p = float(probs.max())
        color = self.semaforo(p)
        return {
            "imagen": arr,
            "prob_manipulada": p,
            "prob_por_zona": probs.reshape(self.H // self.T, self.W // self.T),
            "semaforo": color,
            "accion": ACCIONES[color],
            "mapa": self.mapa_calor(arr, probs) if con_mapa else None,
        }


def superponer(arr, mapa, alfa=0.45):
    """Mapa de calor coloreado (de azul a rojo) sobre la foto, como imagen PIL."""
    import matplotlib

    colores = matplotlib.colormaps["jet"](np.clip(mapa, 0, 1))[..., :3]
    mezcla = (1 - alfa) * arr.astype(np.float32) / 255.0 + alfa * colores
    return Image.fromarray(np.clip(mezcla * 255, 0, 255).astype(np.uint8))
