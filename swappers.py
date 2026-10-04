"""Modelli di sostituzione del volto, intercambiabili.

Deep-Live-Cam espone un solo scambiatore (`inswapper_128`). La ricerca di
riferimento ne indica altri con un comportamento migliore per il nostro caso,
e soprattutto un dettaglio importante: **nessuno di questi modelli deve
trapiantare i capelli della fonte**.

Dal paper di SimSwap: "wFM-id+ e' piu' incline a introdurre i capelli del
volto sorgente. Questo non e' desiderato, visto che stiamo sostituendo solo
i volti". E la guida comparativa riporta che SimSwap mantiene capelli,
linea della mandibola, tono della pelle e illuminazione del target, grazie
alla Weak Feature Matching Loss.

Convenzione di I/O (verificata su `insightface/model_zoo/inswapper.py`, che
e' la stessa libreria usata da Deep-Live-Cam):
  - allineamento col template arcface;
  - input  = RGB / 255, quindi in [0, 1]  (NON [-1, 1]);
  - output = [0, 1], da riportare a uint8 con reverse dei canali;
  - l'embedding va moltiplicato per `emap`, l'ultimo initializer del grafo
    ONNX, e poi rinormalizzato. Senza questa matrice l'identita' non entra
    nel modello e l'output resta il volto target pressoché invariato.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# nome -> (file, dimensione crop, embedding da mappare con emap, ha maschera)
SWAPPERS = {
    "inswapper": ("inswapper_128.onnx", 128, True, False),
    "simswap": ("simswap_256.onnx", 256, False, False),
    "hififace": ("hififace_unofficial_256.onnx", 256, False, True),
}

# ellisse che copre il viso nel crop del template arcface
# Allargata e spostata leggermente verso il basso per coprire meglio
# naso e parte superiore del mento (riduce glitch tipici).
CROP_MASK = {"cx": 0.50, "cy": 0.58, "ax": 0.43, "ay": 0.50}


def estimate_norm(kps, size: int) -> np.ndarray:
    """Matrice che allinea i 5 landmark del viso al crop di `size`."""
    from insightface.utils import face_align
    return face_align.estimate_norm(np.asarray(kps, np.float32), size)


def align(img: np.ndarray, kps, size: int) -> np.ndarray:
    M = estimate_norm(kps, size)
    return cv2.warpAffine(img, M, (size, size), borderValue=0.0)


def to_tensor(img: np.ndarray) -> np.ndarray:
    """uint8 BGR -> float32 NCHW RGB in [0, 1] (equivalente a blobFromImage
    con swapRB=True e mean=0, std=255)."""
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))[None]


def from_tensor(x: np.ndarray) -> np.ndarray:
    """float32 NCHW in [0, 1] -> uint8 BGR."""
    x = np.clip(np.transpose(x[0], (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
    return x[..., ::-1]


def crop_mask(size: int) -> np.ndarray:
    """Maschera ellittica del viso, nello spazio del crop del modello.
    Blur più aggressivo per bordi più soft (riduce glitch sul naso)."""
    m = np.zeros((size, size), np.uint8)
    c = CROP_MASK
    cv2.ellipse(m, (int(size * c["cx"]), int(size * c["cy"])),
                (int(size * c["ax"]), int(size * c["ay"])), 0, 0, 360, 255, -1)
    k = max(5, size // 12)  # blur più forte
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(m, (k, k), 0)


def load_emap(model_path: Path):
    """Ultimo initializer del grafo: la matrice di mappatura dell'embedding."""
    import onnx
    from onnx import numpy_helper
    model = onnx.load(str(model_path))
    return numpy_helper.to_array(model.graph.initializer[-1])


class Swapper:
    """Scambiatore di volti, caricato da un ONNX a scelta."""

    def __init__(self, name: str, models_dir: Path, providers=None):
        if name not in SWAPPERS:
            raise ValueError(f"scambiatore sconosciuto: {name}. "
                             f"Disponibili: {', '.join(SWAPPERS)}")
        import onnxruntime as ort
        self.name = name
        self.file, self.size, self.use_emap, self.has_mask = SWAPPERS[name]
        path = Path(models_dir) / self.file
        if not path.exists():
            raise FileNotFoundError(f"modello mancante: {path}")
        self.emap = load_emap(path) if self.use_emap else None
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.in_names = [i.name for i in self.session.get_inputs()]
        self.out_names = [o.name for o in self.session.get_outputs()]
        self.mask = crop_mask(self.size)

    def latent(self, embedding: np.ndarray) -> np.ndarray:
        """Prepara l'embedding di riferimento nel formato atteso dal modello."""
        e = np.asarray(embedding, np.float32).reshape(1, -1)
        if self.emap is not None:
            e = np.dot(e, self.emap)
            e /= np.linalg.norm(e)
        return e

    def _run(self, crop: np.ndarray, embedding: np.ndarray):
        feed = {self.in_names[0]: to_tensor(crop), self.in_names[1]: self.latent(embedding)}
        wanted = [self.out_names[0]] + ([self.out_names[1]] if self.has_mask else [])
        res = self.session.run(wanted, feed)
        return res[0], (res[1] if self.has_mask else None)

    def swap(self, frame: np.ndarray, target_face, embedding: np.ndarray) -> np.ndarray:
        """Sostituisce il volto di `target_face` con l'identita' di `embedding`."""
        M = estimate_norm(np.asarray(target_face.kps, np.float32), self.size)
        crop = cv2.warpAffine(frame, M, (self.size, self.size), borderValue=0.0)
        out, model_mask = self._run(crop, embedding)
        face = from_tensor(out)

        Minv = cv2.invertAffineTransform(M)
        h, w = frame.shape[:2]
        pasted = cv2.warpAffine(face, Minv, (w, h), borderValue=0)
        mask = cv2.warpAffine(self.mask, Minv, (w, h), borderValue=0)

        if model_mask is not None:
            mm = cv2.resize(model_mask[0, 0], (self.size, self.size))
            mm = cv2.warpAffine((mm > 0.5).astype(np.uint8) * 255, Minv,
                                (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            if (mm > 127).mean() > 0.25 * max((mask > 127).mean(), 1e-6):
                mask = cv2.bitwise_and(mm, mask)

        a = (mask.astype(np.float32) / 255.0)[..., None]
        return np.clip(frame.astype(np.float32) * (1 - a)
                       + pasted.astype(np.float32) * a, 0, 255).astype(np.uint8)


class Enhancer:
    """Miglioratore del volto (GPEN-BFR-256), da usare dopo lo swap."""

    def __init__(self, models_dir: Path, providers=None):
        import onnxruntime as ort
        self.path = Path(models_dir) / "gpen_bfr_256.onnx"
        if not self.path.exists():
            raise FileNotFoundError(f"modello mancante: {self.path}")
        self.session = ort.InferenceSession(str(self.path), providers=providers)
        self.size = 256

    def enhance(self, img: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
        """Migliora il volto nel rettangolo dato, in coordinate del frame."""
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < 8:
            return img
        small = cv2.resize(crop, (self.size, self.size))
        out = self.session.run(["output"], {"input": to_tensor(small)})[0]
        fixed = cv2.resize(from_tensor(out), (crop.shape[1], crop.shape[0]))
        res = img.copy()
        res[y1:y2, x1:x2] = fixed
        return res