"""Modalita' head swap.

Il principio: **il video fornisce geometria e movimento, la foto solo
l'aspetto**. Questo e' l'unico modo per avere una testa che si muove con un
modello che non genera immagini.

Perche' non si puo' incollare la capigliatura della foto
-------------------------------------------------------
La prima versione face cosi': BiSeNet segmentava la foto, la testa veniva
warpiata sulla geometria del target con una similar-transform dai 5 landmark e
i pixel incollati sopra la maschera. Il risultato era una **foto statica
incollata**: i pixel della foto sono identici in ogni frame, quindi non
rispondono al movimento del video. Il bounding box seguiva il detector, ma
dentro restava la fotografia. Misurato: rapporto d'area della testa fino a
15.3x rispetto all'originale, e capelli che comparivano anche dove non c'era
volto.

Cosa fa invece (v3.2 — no floating hair)
---------------------------------------
1. Maschera ellipse-first + rinforzo semantico leggero (evita melting).
2. `inswapper` rigenera il volto interno nella posa del target.
3. Upscale selettivo moderato della regione facciale (1.25x–1.5x).
4. Ricoloratura capelli/pelle in spazio LAB (nessun trapianto geometrico).
5. Poisson blending leggero per integrare senza distorcere.
6. Temporal smoothing REGIONALE: occhi/naso/bocca pesano solo il 40%.
7. Il collo non viene mai toccato.
8. Trapianto geometrico capelli DISATTIVATO di default (evita sticker/cutout).
   Riattivabile con --hair-transfer 1 (accetta possibili aloni).
Profili: --quality fast | high (default) | maximum


Nessun modello generativo di testo e' coinvolto: gli ONNX disponibili per il
face/head swap sono solo inswapper e GFPGAN. I prompt IP-Adapter
dell'enunciato servirebbero per un setup diverso (diffusione +
ControlNet) e non sono applicabili qui. I prompt sono comunque conservati in
swap_config.json, per completezza.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from insightface.utils import face_align

from face_parsing import (FACE_CLASSES, HAIR_CLASSES, HEAD_CLASSES,
                          NECK_CLASSES, SEMANTIC_FACE_CLASSES, HIGH_FREQ_CLASSES,
                          FaceParser, feather, inner_face_mask,
                          landmark_head_mask, semantic_face_mask, high_freq_mask)

# Oltre questi limiti l'output di BiSeNet e' considerato sbagliato: su un
# volto piccolo e generato dall'AI il segmentatore etichetta a volte un terzo
# dell'immagine come "testa". Un volto realistico occupa 0.4x-6x l'area del
# bounding box; fuori da questo intervallo si ripiega sulla geometria.
MIN_HEAD_RATIO = 0.4
MAX_HEAD_RATIO = 6.0


def similarity_from_landmarks(src_kps: np.ndarray, dst_kps: np.ndarray) -> np.ndarray:
    """Trasformazione che porta i 5 landmark della foto su quelli del video."""
    src = np.asarray(src_kps, np.float32)
    dst = np.asarray(dst_kps, np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    if M is None:
        sc = float(np.linalg.norm(dst[4] - dst[0])) or 1.0
        M = np.array([[sc, 0, dst[0][0] - sc * dst[0][0]],
                      [0, sc, dst[0][1] - sc * dst[0][1]]], np.float32)
    return M.astype(np.float32)


def sane_head_mask(labels: np.ndarray, face_box, kps) -> tuple[np.ndarray, bool]:
    """Maschera testa del target, con controllo di plausibilita'.

    Ritorna (maschera, e_fallenuta). Se BiSeNet produce una maschera assurda
    si usa un'ellisse ricavata dai 5 landmark, che segue il movimento anche
    quando il segmentatore sbaglia.
    """
    fb_area = max(1.0, (face_box[2] - face_box[0]) * (face_box[3] - face_box[1]))
    mask = FaceParser.mask_for(labels, HEAD_CLASSES)
    area = float((mask > 0).sum())
    if area and MIN_HEAD_RATIO * fb_area <= area <= MAX_HEAD_RATIO * fb_area:
        return mask, False
    return landmark_head_mask(labels.shape[:2], kps), True


_DEBUG = bool(os.environ.get("HEADSWAP_DEBUG"))


class HeadSwapper:
    """Sostituzione della testa: volto da inswapper, capelli ricolorati."""

    # Profili di qualità predefiniti
    QUALITY = {
        # hair=False di default: il trapianto geometrico crea effetto sticker/cutout
        # e capelli fluttuanti. Si tiene solo il ricoloramento (color transfer).
        # Per riattivare: --hair-transfer 1
        "fast":     {"blend": 9,  "temporal": 0.26, "face_scale": 1.00, "poisson": False, "semantic": False, "hair": False},
        "high":     {"blend": 12, "temporal": 0.30, "face_scale": 1.25, "poisson": True,  "semantic": True,  "hair": False},
        "maximum":  {"blend": 14, "temporal": 0.38, "face_scale": 1.5,  "poisson": True,  "semantic": True,  "hair": False},
    }

    def __init__(self, models_dir, providers, swapper_fn, mode: str = "headswap",
                 blend: int | None = None, harmonize: float = 1.0,
                 temporal: float | None = None, face_scale: float | None = None,
                 use_poisson: bool | None = None, use_semantic: bool | None = None,
                 hair: bool | None = None, quality: str | None = None):
        # I default sono None per distinguere "non passato" da "passato":
        # il preset --quality fornisce i valori, ma un override esplicito
        # della CLI deve poter vincere sul preset.
        q = self.QUALITY.get(quality, {}) if quality else {}
        blend = q.get("blend", 13) if blend is None else blend
        temporal = q.get("temporal", 0.38) if temporal is None else temporal
        face_scale = q.get("face_scale", 1.25) if face_scale is None else face_scale
        use_poisson = q.get("poisson", True) if use_poisson is None else use_poisson
        use_semantic = q.get("semantic", True) if use_semantic is None else use_semantic
        q = q or {}
        self.mode = mode
        self.swapper_fn = swapper_fn
        self.blend = blend
        self.hair_transfer = bool(q.get("hair", False)) if hair is None else bool(hair)  # default OFF
        self.harmonize = harmonize
        self.temporal = temporal
        self.face_scale = max(1.0, min(face_scale, 2.5))  # clamp
        self.use_poisson = use_poisson
        self.use_semantic = use_semantic
        self.parser = FaceParser(models_dir, providers)
        self._src_cache: dict = {}
        self._prev: dict = {}
        self.fallbacks = 0
