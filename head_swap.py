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

Cosa fa invece
-------------
1. `inswapper` rigenera il volto interno nella posa del target a ogni frame:
   e' lui a portare il movimento, ed e' la ragione per cui il volto si muove.
2. La capigliatura viene ricolorata, non trapiantata: dentro la maschera
   capelli del *video* si allineano media e deviazione standard alla
   capigliatura della *foto*. I pixel restano quelli del video, quindi
   ombreggiamenti, movimenti e capelli che svolazzano restano al loro posto.
   Solo il colore cambia.
3. La tonalita' della pelle viene allineata all'illuminazione del target,
   limitata alla testa e mai all'intero frame.
4. Il collo non viene mai toccato.

Nessun modello generativo di testo e' coinvolto: gli ONNX disponibili per il
face/head swap sono solo inswapper e GFPGAN. I prompt IP-Adapter
dell'enunciato servirebbero per un setup diverso (diffusione +
ControlNet) e non sono applicabili qui. I prompt sono comunque conservati in
swap_config.json, per completezza.
"""

from __future__ import annotations

import cv2
import numpy as np

from face_parsing import (FACE_CLASSES, HAIR_CLASSES, HEAD_CLASSES,
                          NECK_CLASSES, FaceParser, feather, inner_face_mask,
                          landmark_head_mask)

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


class HeadSwapper:
    """Sostituzione della testa: volto da inswapper, capelli ricolorati."""

    def __init__(self, models_dir, providers, swapper_fn, mode: str = "headswap",
                 blend: int = 9, harmonize: float = 1.0, temporal: float = 0.0):
        self.mode = mode
        self.swapper_fn = swapper_fn
        self.blend = blend
        self.harmonize = harmonize
        self.temporal = temporal
        self.parser = FaceParser(models_dir, providers)
        self._src_cache: dict = {}
        self._prev: dict = {}
        self.fallbacks = 0

    def prepare_source(self, src_img: np.ndarray, src_face) -> dict:
        """Analizza la foto una volta sola: statistiche di capelli e pelle."""
        key = id(src_face)
        if key in self._src_cache:
            return self._src_cache[key]
        labels, box = self.parser.parse(src_img, src_face.bbox)
        entry = {
            "_face": src_face,
            "img": src_img,
            "labels": labels,
            "box": box,
            "hair_stats": self._stats(src_img, labels, HAIR_CLASSES),
            "skin_stats": self._stats(src_img, labels, FACE_CLASSES),
            "kps": np.asarray(src_face.kps, np.float32),
        }
        self._src_cache[key] = entry
        return entry

    @staticmethod
    def _stats(img: np.ndarray, labels: np.ndarray, classes):
        """Media e deviazione standard per canale, su una classe."""
        m = np.isin(labels, list(classes))
        n = int(m.sum())
        if n < 40:
            return None
        px = img[m].astype(np.float32)
        return px.mean(axis=0), px.std(axis=0) + 1e-3, n

    @staticmethod
    def _apply_stats(dst: np.ndarray, mask: np.ndarray, stats, soft: int = 15):
        """Allinea media e deviazione standard dentro la maschera.

        I pixel non vengono sostituiti: vengono riscalati e traslati. La
        struttura spaziale resta, quindi un capello che si muove continua a
        muoversi; cambia solo il colore.

        La miscelazione passa per un alpha sfumato invece di selezionare i
        pixel con una soglia: con una maschera binaria il ricolor finiva con
        un bordo netto, visibile come un contorno ("disegno") attorno a
        capelli e pelle. Le statistiche sono calcolate solo sull'interno
        della maschera, cosi' il bordo non le altera.
        """
        if stats is None:
            return dst
        mean, std, _ = stats

        hard = (mask > 0).astype(np.uint8)
        if not hard.any():
            return dst

        # interno: le statistiche non devono essere inquinate dal bordo
        k = 5 if 5 % 2 else 6
        inner = cv2.erode(hard, np.ones((k, k), np.uint8), iterations=1)
        if int(inner.sum()) < 40:
            inner = hard

        px = dst[inner > 0].astype(np.float32)
        cur_mean = px.mean(axis=0)
        cur_std = px.std(axis=0) + 1e-6
        new = np.clip((dst.astype(np.float32) - cur_mean) / cur_std * std + mean,
                      0, 255)

        a = (feather(hard, soft).astype(np.float32) / 255.0)[..., None]
        return np.clip(dst.astype(np.float32) * (1.0 - a) + new * a,
                       0, 255).astype(np.uint8)

    def swap(self, frame: np.ndarray, src_entry: dict, target_face) -> np.ndarray:
        """Applica head swap o face swap a un volto target nel frame."""
        if self.mode == "faceswap":
            return self.swapper_fn(src_entry.get("_face"), target_face, frame)

        out = frame.copy()
        kps = np.asarray(target_face.kps, np.float32)
        target_labels, _ = self.parser.parse(frame, target_face.bbox)

        head, fell_back = sane_head_mask(target_labels, target_face.bbox, kps)
        if fell_back:
            self.fallbacks += 1
        inner = feather(inner_face_mask(frame.shape, kps, 1.0), self.blend)
        neck = FaceParser.mask_for(target_labels, NECK_CLASSES)

        # Il ricoloratura agisce solo su capelli e pelle, mai su collo e sfondo.
        # `head` e' la maschera sanificata: senza questa intersezione BiSeNet
        # puo' etichettare come "capelli" mezzo fotogramma (297934 px nel test)
        # e il ricoloratura stravolgere l'immagine.
        capelli = FaceParser.mask_for(target_labels, HAIR_CLASSES)
        capelli = cv2.bitwise_and(capelli, head)
        if capelli.any():
            capelli = cv2.bitwise_and(capelli, cv2.bitwise_not(feather(neck, 5)))
            out = self._apply_stats(out, capelli, src_entry["hair_stats"])
        if self.harmonize > 0 and src_entry["skin_stats"] is not None:
            pelle = FaceParser.mask_for(target_labels, FACE_CLASSES)
            pelle = cv2.bitwise_and(pelle, head)
            pelle = cv2.bitwise_and(pelle, cv2.bitwise_not(feather(neck, 5)))
            if pelle.any():
                out = self._apply_stats(out, pelle, src_entry["skin_stats"])

        # Il bordo della zona toccata viene sfumato per non lasciare un salto.
        bordo = cv2.bitwise_or(capelli, FaceParser.mask_for(target_labels, FACE_CLASSES))
        bordo = cv2.bitwise_and(bordo, cv2.bitwise_not(inner))
        bordo = cv2.bitwise_and(bordo, cv2.bitwise_not(feather(neck, 5)))
        if bordo.any():
            a = feather(bordo, self.blend).astype(np.float32)[..., None] / 255.0
            sm = cv2.GaussianBlur(out, (0, 0), 1.0)
            out = np.clip(out.astype(np.float32) * (1 - a * 0.5)
                          + sm.astype(np.float32) * (a * 0.5), 0, 255).astype(np.uint8)

        # inswapper genera il volto nella posa del target: e' la fonte del
        # movimento. Scrive in place su GPU, restituisce una copia su CPU,
        # quindi il valore di ritorno va sempre assegnato.
        out = self.swapper_fn(src_entry.get("_face"), target_face, out)
        if self.temporal > 0:
            out = self._temporal_blend(out, target_face, src_entry)
        return out

    def _temporal_blend(self, out: np.ndarray, target_face, src_entry: dict,
                        size: int = 128) -> np.ndarray:
        """Riduce il flickering mescolando il fotogramma precedente.

        Il paper "Temporal Optimization for Face Swapping Video based on
        Consistency Inheritance" (ACM MM 2024) tratta l'incoerenza fra
        fotogrammi come un disturbo nel dominio del tempo. Riaddestrare un
        modello per risolverlo non e' pratico qui, ma una parte del
        vantaggio si ottiene-media: il fotogramma precedente viene portato
        nella posizione attuale con i landmark e usato come prior del
        fotogramma corrente, dentro la sola regione del volto.
        """
        from insightface.utils import face_align

        key = id(src_entry.get("_face"))
        kps = np.asarray(target_face.kps, np.float32)
        prev = self._prev.get(key)

        if prev is None:
            self._prev[key] = (kps.copy(), out.copy())
            return out

        prev_kps, prev_img = prev
        M_prev = face_align.estimate_norm(prev_kps, size)
        M_cur = face_align.estimate_norm(kps, size)

        # frame precedente -> frame corrente
        A = np.vstack([M_cur, [0, 0, 1]]) @ np.linalg.inv(
            np.vstack([M_prev, [0, 0, 1]]))
        warped = cv2.warpAffine(prev_img, A[:2], (out.shape[1], out.shape[0]),
                                borderValue=0)

        mask = feather(inner_face_mask(out.shape, kps, 1.0), self.blend)
        # zone dove il warp non ha coperto nulla vanno escluse
        valid = cv2.warpAffine(np.full(prev_img.shape[:2], 255, np.uint8),
                               A[:2], (out.shape[1], out.shape[0]),
                               flags=cv2.INTER_NEAREST, borderValue=0)
        a = (mask.astype(np.float32) / 255.0 * self.temporal)
        a *= (valid.astype(np.float32) / 255.0)
        a = a[..., None]

        out = np.clip(out.astype(np.float32) * (1 - a)
                      + warped.astype(np.float32) * a, 0, 255).astype(np.uint8)
        self._prev[key] = (kps.copy(), out.copy())
        return out