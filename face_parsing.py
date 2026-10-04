"""Parsing semantico del viso e della testa via BiSeNet (19 classi CelebAMask-HQ).

Serve alla modalita' headswap: inswapper sostituisce solo il volto interno
(dall'ellisse dei 5 landmark) e non sa nulla dei capelli. Per estendere la
sostituzione a capelli, orecchie e cappello serve una segmentazione che
sappia dove finisce il viso e dove inizia la chioma.

Modello: yakhyo/face-parsing, BiSeNet ResNet18, licenza MIT, 51 MB.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = ("https://github.com/yakhyo/face-parsing/releases/download/"
             "weights/resnet18.onnx")
MODEL_SIZE = 51_000_000

CLASS_NAMES = {
    0: "background", 1: "skin", 2: "l_brow", 3: "r_brow", 4: "l_eye", 5: "r_eye",
    6: "eyeglass", 7: "l_ear", 8: "r_ear", 9: "earring", 10: "nose", 11: "mouth",
    12: "u_lip", 13: "l_lip", 14: "neck", 15: "necklace", 16: "cloth", 17: "hair",
    18: "hat",
}

FACE_CLASSES = {1, 2, 3, 4, 5, 10, 11, 12, 13}
HEAD_CLASSES = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 17, 18}
HAIR_CLASSES = {17, 18}
NECK_CLASSES = {14, 15}
EXCLUDE_FROM_HEAD = {0, 14, 15, 16}

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def ensure_parsing_model(models_dir: Path) -> Path:
    path = models_dir / "bisenet_resnet18.onnx"
    if path.exists() and path.stat().st_size > MODEL_SIZE * 0.8:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.rename(path)
    return path


def expanded_box(bbox, img_shape, pad_x=0.35, pad_top=0.55, pad_bottom=0.30):
    """Rettangolo allargato per contenere i capelli, che stanno sopra la bbox."""
    h, w = img_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw, bh = x2 - x1, y2 - y1
    return (max(0, int(x1 - bw * pad_x)),
            max(0, int(y1 - bh * pad_top)),
            min(w, int(x2 + bw * pad_x)),
            min(h, int(y2 + bh * pad_bottom)))


def feather(mask: np.ndarray, radius: int = 9) -> np.ndarray:
    """Sfuma i bordi della maschera per evitare lo stacco visibile."""
    k = radius if radius % 2 else radius + 1
    return cv2.GaussianBlur(mask, (k, k), 0)


class FaceParser:
    """Restituisce la mappa delle classi per un ritaglio di viso."""

    def __init__(self, models_dir: Path, providers=None, size: int = 512):
        import onnxruntime as ort
        self.size = size
        self.providers = providers or ort.get_available_providers()
        self.session = ort.InferenceSession(str(ensure_parsing_model(models_dir)),
                                            providers=self.providers)

    def parse_crop(self, img: np.ndarray, box) -> np.ndarray:
        """Class map delle dimensioni del ritaglio indicato da box."""
        x1, y1, x2, y2 = box
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((0, 0), np.uint8)
        ch, cw = crop.shape[:2]
        scale = np.array([cw / self.size, ch / self.size, cw / self.size, ch / self.size],
                         np.float32)
        norm = cv2.resize(crop, (self.size, self.size)).astype(np.float32) / 255.0
        norm = (norm - _MEAN) / _STD
        tensor = np.transpose(norm, (2, 0, 1))[None].astype(np.float32)
        labels = self.session.run(["output"], {"input": tensor})[0][0].argmax(0)
        up = cv2.resize(labels.astype(np.uint8), (cw, ch),
                        interpolation=cv2.INTER_NEAREST)
        return np.pad(up, ((y1, img.shape[0] - y2), (x1, img.shape[1] - x2)),
                      constant_values=0)[..., :img.shape[0], :img.shape[1]]

    def parse(self, img: np.ndarray, bbox, pad: bool = True):
        """(class_map, box) del viso, con contesto per i capelli se pad=True."""
        box = expanded_box(bbox, img.shape) if pad else \
            tuple(int(v) for v in bbox[:4])
        return self.parse_crop(img, box), box

    @staticmethod
    def mask_for(labels: np.ndarray, classes) -> np.ndarray:
        return np.isin(labels, list(classes)).astype(np.uint8) * 255

    @staticmethod
    def feather(mask: np.ndarray, radius: int = 9) -> np.ndarray:
        return feather(mask, radius)


def inner_face_mask(img_shape, kps, scale: float = 1.0) -> np.ndarray:
    """Ellisse dei 5 landmark: la regione che inswapper sostituisce davvero."""
    mask = np.zeros(img_shape[:2], np.uint8)
    pts = np.asarray(kps, np.float32)
    centre = pts[:4].mean(axis=0)
    span = pts[2] - pts[0]
    cv2.ellipse(mask, (int(centre[0]), int(centre[1])),
                (int(abs(span[0]) * scale), int(abs(span[1]) * scale)),
                float(np.degrees(np.arctan2(span[1], span[0]))), 0, 360, 255, -1)
    return mask


def landmark_head_mask(img_shape, kps, grow_x: float = 0.55,
                       grow_y_up: float = 0.75) -> np.ndarray:
    """Testa approssimata dai soli 5 landmark del viso.

    Serve quando il segmentatore produce una maschera implausibile. Diversamente
    da BiSeNet non indovina nulla: e' un'ellisse centrata sui 5 landmark e
    scalata, quindi segue il movimento del volto per costruzione.
    """
    mask = np.zeros(img_shape[:2], np.uint8)
    pts = np.asarray(kps, np.float32)
    if pts.shape[0] < 5:
        return mask
    centre = pts[:4].mean(axis=0)
    span = pts[2] - pts[0]
    ax = max(8.0, abs(float(span[0])) * (1.0 + grow_x))
    ay = max(8.0, abs(float(span[1])) * (1.0 + grow_y_up))
    # il centro va spostato in alto per includere capelli e fronte
    cy = centre[1] - ay * grow_y_up * 0.45
    cv2.ellipse(mask, (int(centre[0]), int(cy)), (int(ax), int(ay)),
                0, 0, 360, 255, -1)
    return mask