"""Modalita' head swap: estende la sostituzione di inswapper ai capelli.

inswapper sostituisce solo l'ellisse interna dei 5 landmark. Per un head swap
servono anche capelli, orecchie e cappello, quindi:

1. il volto interno viene sostituito da inswapper (identita' del riferimento)
2. BiSeNet segmenta il video e la foto per separare capelli/orecchie/cappello
   dal collo e dall'abbigliamento
3. la regione "testa" della foto viene warpiata sulla geometria del target
   tramite similar-transform dei 5 landmark
4. il bordo viene sfumato per evitare lo stacco sul collo

Nessun modello generativo di testo (IP-Adapter, inpainting) e' coinvolto: gli
onnx disponibili per il face/head swap sono solo inswapper e GFPGAN. I prompt
IP-Adapter dell'enunciato servirebbero per un setup diverso (diffusion + ControlNet)
e non sono applicabili qui. I prompt sono comunque conservati nel config
generato dall'analizzatore, per completezza.
"""

from __future__ import annotations

import cv2
import numpy as np

from face_parsing import (FACE_CLASSES, HAIR_CLASSES, HEAD_CLASSES,
                          NECK_CLASSES, FaceParser, feather, inner_face_mask)


def similarity_from_landmarks(src_kps: np.ndarray, dst_kps: np.ndarray) -> np.ndarray:
    """Trasformazione che porta i 5 landmark della foto su quelli del video."""
    src = np.asarray(src_kps, np.float32)
    dst = np.asarray(dst_kps, np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    if M is None:
        h, w = dst[0][1] - dst[0][0], dst[0][0] - dst[4][0]
        M = np.array([[h, 0, dst[0][0]], [0, h, dst[0][1]]], np.float32)
    return M.astype(np.float32)


class HeadSwapper:
    """Gestisce i modelli necessari alla sostituzione di testa e volto."""

    def __init__(self, models_dir, providers, swapper_fn, mode: str = "headswap",
                 blend: int = 9, recolor: bool = True):
        self.mode = mode
        self.swapper_fn = swapper_fn
        self.blend = blend
        self.recolor = recolor
        self.parser = FaceParser(models_dir, providers)
        self._src_cache: dict = {}

    def prepare_source(self, src_img: np.ndarray, src_face) -> dict:
        """Analizza la foto di riferimento una volta sola: maschere, warping, tonalita'."""
        key = id(src_face)
        if key in self._src_cache:
            return self._src_cache[key]
        labels, box = self.parser.parse(src_img, src_face.bbox)
        h, w = src_img.shape[:2]
        entry = {
            "_face": src_face,
            "img": src_img,
            "labels": labels,
            "box": box,
            "head": FaceParser.mask_for(labels, HEAD_CLASSES),
            "hair": FaceParser.mask_for(labels, HAIR_CLASSES),
            "kps": np.asarray(src_face.kps, np.float32),
            "h": h, "w": w,
        }
        self._src_cache[key] = entry
        return entry

    @staticmethod
    def _skin_mean(img: np.ndarray, labels: np.ndarray):
        """Colore medio della pelle della foto, per uniformare l'illuminazione."""
        sel = labels == 1
        if sel.sum() < 50:
            return None
        patch = img[sel]
        return np.array([patch[:, i].mean() for i in range(3)], np.float32)

    def swap(self, frame: np.ndarray, src_entry: dict, target_face, kps_ref=None) -> np.ndarray:
        """Applica head swap o face swap a un volto target nel frame."""
        if self.mode == "faceswap":
            return self.swapper_fn(src_entry.get("_face"), target_face, frame)

        out = frame.copy()
        dst_kps = np.asarray(target_face.kps, np.float32)
        M = similarity_from_landmarks(src_entry["kps"], dst_kps)
        h, w = frame.shape[:2]

        warped_img = cv2.warpAffine(src_entry["img"], M, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT_101)
        warped_labels = cv2.warpAffine(src_entry["labels"], M, (w, h),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_head = FaceParser.mask_for(warped_labels, HEAD_CLASSES)
        warped_hair = FaceParser.mask_for(warped_labels, HAIR_CLASSES)

        target_labels, _ = self.parser.parse(frame, target_face.bbox)
        target_head = FaceParser.mask_for(target_labels, HEAD_CLASSES)
        target_hair = FaceParser.mask_for(target_labels, HAIR_CLASSES)

        inner = feather(inner_face_mask(frame.shape, dst_kps, 1.0), self.blend)

        # Allinea l'illuminazione: il rapporto dei colori della pelle si applica
        # SOLO ai pixel della testa di riferimento.Applicarlo al frame intero
        # sporcherebbe l'intero video (errore che gonfiava il fondo di ~53/255).
        if self.recolor:
            src_mean = self._skin_mean(warped_img, warped_labels)
            dst_mean = self._skin_mean(frame, target_labels)
            if src_mean is not None and dst_mean is not None:
                ratio = (src_mean / np.maximum(dst_mean, 1.0)).astype(np.float32)
                warped_img = np.clip(warped_img.astype(np.float32) * ratio,
                                     0, 255).astype(np.uint8)

        region = cv2.bitwise_or(target_head, warped_head)
        region = cv2.bitwise_and(region, cv2.bitwise_not(inner))
        alpha = (feather(region, self.blend).astype(np.float32) / 255.0)[..., None]
        # La protezione del collo va applicata DOPO la sfumatura: sottrarla prima
        # non serve, perche' il blur del bordo inferiore della testa si espande
        # di nuovo nel collo. Clampa alpha a zero li'.
        neck = FaceParser.mask_for(target_labels, NECK_CLASSES)
        if neck.any():
            alpha = alpha * (1.0 - feather(neck, 5).astype(np.float32)[..., None] / 255.0)

        out = np.clip(out.astype(np.float32) * (1 - alpha)
                      + warped_img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        # inswapper gestisce l'interno del volto; sul percorso GPU scrive in
        # place, ma su quello CPU ritorna una copia: si assegna sempre.
        out = self.swapper_fn(src_entry.get("_face"), target_face, out)

        if self.blend > 1:
            sm = cv2.GaussianBlur(out, (0, 0), 1.2)
            edge = alpha.max(axis=2)
            out = np.clip(out.astype(np.float32) * (1 - edge[..., None] * 0.35)
                          + sm.astype(np.float32) * (edge[..., None] * 0.35),
                          0, 255).astype(np.uint8)
        return out