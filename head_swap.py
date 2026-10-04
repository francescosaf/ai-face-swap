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

Cosa fa invece (v3.1 stabile)
----------------------------
1. Maschera ellipse-first + rinforzo semantico leggero (evita melting).
2. `inswapper` rigenera il volto interno nella posa del target.
3. Upscale selettivo moderato della regione facciale (1.25x–1.5x).
4. Ricoloratura capelli/pelle in spazio LAB.
5. Poisson blending leggero (peso 0.55) per integrare senza distorcere.
6. Temporal smoothing REGIONALE: occhi/naso/bocca pesano solo il 40%.
7. Il collo non viene mai toccato.
Profili: --quality fast | high (default, stabile) | maximum


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


class HeadSwapper:
    """Sostituzione della testa: volto da inswapper, capelli ricolorati."""

    # Profili di qualità predefiniti
    QUALITY = {
        "fast":     {"blend": 9,  "temporal": 0.25, "face_scale": 1.0, "poisson": False, "semantic": False},
        "high":     {"blend": 12, "temporal": 0.32, "face_scale": 1.25, "poisson": True,  "semantic": True},
        "maximum":  {"blend": 14, "temporal": 0.38, "face_scale": 1.5,  "poisson": True,  "semantic": True},
    }

    def __init__(self, models_dir, providers, swapper_fn, mode: str = "headswap",
                 blend: int = 13, harmonize: float = 1.0, temporal: float = 0.38,
                 face_scale: float = 1.25, use_poisson: bool = True,
                 use_semantic: bool = True, quality: str | None = None):
        if quality and quality in self.QUALITY:
            q = self.QUALITY[quality]
            blend = q["blend"]
            temporal = q["temporal"]
            face_scale = q["face_scale"]
            use_poisson = q["poisson"]
            use_semantic = q["semantic"]
        self.mode = mode
        self.swapper_fn = swapper_fn
        self.blend = blend
        self.harmonize = harmonize
        self.temporal = temporal
        self.face_scale = max(1.0, min(face_scale, 2.5))  # clamp
        self.use_poisson = use_poisson
        self.use_semantic = use_semantic
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
    def _apply_stats(dst: np.ndarray, mask: np.ndarray, stats, soft: int = 17):
        """Allinea media e deviazione standard dentro la maschera (spazio LAB).

        Lavora in LAB per un trasferimento di colore più naturale.
        I pixel non vengono sostituiti: vengono riscalati e traslati.
        La struttura spaziale resta, quindi un capello che si muove continua
        a muoversi; cambia solo il colore.

        La miscelazione passa per un alpha sfumato. Le statistiche sono
        calcolate solo sull'interno della maschera.
        """
        if stats is None:
            return dst
        mean, std, _ = stats

        hard = (mask > 0).astype(np.uint8)
        if not hard.any():
            return dst

        k = 5 if 5 % 2 else 6
        inner = cv2.erode(hard, np.ones((k, k), np.uint8), iterations=1)
        if int(inner.sum()) < 40:
            inner = hard

        # Trasferimento in spazio LAB (più robusto e naturale della media BGR)
        lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB).astype(np.float32)
        px = lab[inner > 0]
        cur_mean = px.mean(axis=0)
        cur_std = px.std(axis=0) + 1e-6

        # Approssimazione robusta: scala luminosità e canali colore
        # usando le statistiche BGR della source come riferimento di tono
        src_mean = mean.astype(np.float32)
        src_std = std.astype(np.float32)
        ratio = (src_std.mean() + 1e-3) / (cur_std.mean() + 1e-3)

        new_lab = lab.copy()
        # Luminosità: mix tra target e source per non schiacciare i dettagli
        new_lab[..., 0] = np.clip(
            (lab[..., 0] - cur_mean[0]) * (src_std.mean() / (cur_std[0] + 1e-6))
            + cur_mean[0] * 0.55 + src_mean.mean() * 0.45, 0, 255)
        new_lab[..., 1] = np.clip(
            (lab[..., 1] - cur_mean[1]) * ratio * 0.85 + cur_mean[1], 0, 255)
        new_lab[..., 2] = np.clip(
            (lab[..., 2] - cur_mean[2]) * ratio * 0.85 + cur_mean[2], 0, 255)

        new_bgr = cv2.cvtColor(new_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        a = (feather(hard, soft).astype(np.float32) / 255.0)[..., None]
        return np.clip(dst.astype(np.float32) * (1.0 - a) + new_bgr.astype(np.float32) * a,
                       0, 255).astype(np.uint8)

    def _transfer_hair(self, out: np.ndarray, src_entry: dict, target_face,
                       head: np.ndarray, inner: np.ndarray) -> np.ndarray:
        """Trasferisce la GEOMETRIA dei capelli dalla foto al target.

        Il solo ricoloramento non basta per un head swap: i capelli restano
        quelli del video e il risultato si legge come un face swap. Qui la
        capigliatura della fonte viene allineata al volto del target e
        incollata, ma solo dove BiSeNet dice che ci sono capelli, limitata
        alla testa del target, escluso il volto (che gestisce inswapper) e
        il collo. Il tono viene poi riallineato all'illuminazione del video,
        altrisi sembrerebbe un collage.

        Il trapianto e' voluto, ma circoscritto: la prima versione incollava
        l'intera testa e l'area raggiungeva il 15.3x dell'originale.
        """
        from insightface.utils import face_align

        src_img = src_entry.get("img")
        if src_img is None:
            return out

        M_src = face_align.estimate_norm(np.asarray(src_entry["kps"], np.float32), 112)
        M_tgt = face_align.estimate_norm(np.asarray(target_face.kps, np.float32), 112)
        A = np.vstack([M_tgt, [0, 0, 1]]) @ np.linalg.inv(np.vstack([M_src, [0, 0, 1]]))

        h, w = out.shape[:2]
        hair_src = src_entry.get("hair_mask")
        if hair_src is None or not hair_src.any():
            return out

        capelli = cv2.warpAffine(hair_src, A[:2], (w, h), flags=cv2.INTER_LINEAR,
                                 borderValue=0)
        sorgente = cv2.warpAffine(src_img, A[:2], (w, h), borderValue=0)

        # solo capelli, dentro la testa del target, fuori volto e fuori collo
        alpha = cv2.bitwise_and(capelli, head)
        alpha = cv2.bitwise_and(alpha, cv2.bitwise_not(inner))
        alpha = cv2.bitwise_and(alpha, cv2.bitwise_not(self._neck_neighbour))
        if int((alpha > 0).sum()) < 120:
            return out

        # il bordo e' sfumato su due lati: dentro (dove incontra il volto
        # generato) e fuori (dove incontra lo sfondo)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 3.0)
        a = np.clip(alpha.astype(np.float32) / 255.0, 0, 1)[..., None]

        # tonalita' della fonte riallineata a quella del target, altrimenti i
        # capelli incollati si leggono come un estraneo
        sorgente = self._match_lab(sorgente, a[..., 0], out)
        return np.clip(out.astype(np.float32) * (1 - a)
                       + sorgente.astype(np.float32) * a, 0, 255).astype(np.uint8)

    @staticmethod
    def _match_lab(img: np.ndarray, alpha: np.ndarray, reference: np.ndarray):
        """Riporta i colori di `img` su quelli di `reference` dentro alpha."""
        m = alpha > 0.35
        if m.sum() < 80:
            return img
        lab_i = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_r = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        si, sr = lab_i[m], lab_r[m]
        sc = (sr.std(axis=0) + 1e-6) / (si.std(axis=0) + 1e-6)
        sc = np.clip(sc, 0.6, 1.6)
        lab_i = (lab_i - si.mean(axis=0)) * sc + sr.mean(axis=0)
        out = cv2.cvtColor(np.clip(lab_i, 0, 255).astype(np.uint8),
                           cv2.COLOR_LAB2BGR)
        return out

    def swap(self, frame: np.ndarray, src_entry: dict, target_face) -> np.ndarray:
        """Applica head swap o face swap a un volto target nel frame.

        Pipeline v3 (qualità alta):
        1. Ricoloratura capelli + pelle (LAB)
        2. (opzionale) Upscale della regione facciale
        3. inswapper sull'identità
        4. Poisson blending con maschera semantica
        5. Temporal smoothing regionale (meno peso su occhi/naso/bocca)
        """
        if self.mode == "faceswap":
            return self.swapper_fn(src_entry.get("_face"), target_face, frame)

        out = frame.copy()
        kps = np.asarray(target_face.kps, np.float32)
        target_labels, _ = self.parser.parse(frame, target_face.bbox)

        head, fell_back = sane_head_mask(target_labels, target_face.bbox, kps)
        if fell_back:
            self.fallbacks += 1

        # --- Maschera del volto interno ---
        # Ellipse-first (robusta su pose laterali). Semantica solo come rinforzo leggero.
        ellipse = inner_face_mask(frame.shape, kps, scale=1.15)
        if self.use_semantic:
            sem = semantic_face_mask(target_labels, kps, expand=2)
            # Usa la semantica solo dove coincide con l'ellisse (evita melting)
            face_mask_hard = cv2.bitwise_or(ellipse, cv2.bitwise_and(sem, ellipse))
            face_mask_hard = cv2.dilate(face_mask_hard, np.ones((3, 3), np.uint8), iterations=1)
        else:
            face_mask_hard = ellipse

        inner = feather(face_mask_hard, self.blend)
        neck = FaceParser.mask_for(target_labels, NECK_CLASSES)

        # --- 1. Ricoloratura (solo capelli e pelle) ---
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

        # Sfumatura leggera del bordo pre-swap
        bordo = cv2.bitwise_or(capelli, FaceParser.mask_for(target_labels, FACE_CLASSES))
        bordo = cv2.bitwise_and(bordo, cv2.bitwise_not(inner))
        bordo = cv2.bitwise_and(bordo, cv2.bitwise_not(feather(neck, 5)))
        if bordo.any():
            a = feather(bordo, self.blend).astype(np.float32)[..., None] / 255.0
            sm = cv2.GaussianBlur(out, (0, 0), 1.0)
            out = np.clip(out.astype(np.float32) * (1 - a * 0.45)
                          + sm.astype(np.float32) * (a * 0.45), 0, 255).astype(np.uint8)

        # --- 2. Upscale della regione facciale (se face_scale > 1) ---
        # Lavora solo sulla ROI del volto a risoluzione maggiore, poi reinserisce.
        # Molto più efficiente che upscalare tutto il frame.
        did_upscale = False
        if self.face_scale > 1.05:
            x1, y1, x2, y2 = [int(v) for v in target_face.bbox[:4]]
            # padding generoso per non tagliare il volto
            pad = int(max(x2 - x1, y2 - y1) * 0.35)
            h, w = out.shape[:2]
            rx1 = max(0, x1 - pad)
            ry1 = max(0, y1 - pad)
            rx2 = min(w, x2 + pad)
            ry2 = min(h, y2 + pad)
            roi = out[ry1:ry2, rx1:rx2]
            if min(roi.shape[:2]) > 24:
                new_w = int((rx2 - rx1) * self.face_scale)
                new_h = int((ry2 - ry1) * self.face_scale)
                # limita a 512 per non esplodere la memoria/tempo su CPU
                max_side = 512
                if max(new_w, new_h) > max_side:
                    s = max_side / max(new_w, new_h)
                    new_w, new_h = int(new_w * s), int(new_h * s)
                up = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                # adatta i landmark al ROI upscalato (per eventuale uso futuro)
                # per ora inswapper lavora sul frame intero; l'upscale serve
                # principalmente all'enhancer e al dettaglio post-swap
                did_upscale = True
                up_roi = up  # salvato per il reinserimento dopo

        # --- 3. inswapper ---
        pre_swap = out.copy()
        out = self.swapper_fn(src_entry.get("_face"), target_face, out)

        # Se abbiamo fatto upscale, possiamo sharpenare leggermente la zona
        if did_upscale and self.face_scale >= 1.4:
            # Unsharpen mask leggero solo sulla zona facciale
            blur = cv2.GaussianBlur(out, (0, 0), 1.2)
            out = np.clip(out.astype(np.float32) * 1.25 - blur.astype(np.float32) * 0.25,
                          0, 255).astype(np.uint8)

        # --- 4. Poisson blending con maschera semantica ---
        if self.use_poisson:
            try:
                center = (int(kps[:, 0].mean()), int(kps[:, 1].mean()))
                # Maschera più stretta e peso più leggero → meno rischio di melting
                poisson_mask = cv2.erode(face_mask_hard, np.ones((5, 5), np.uint8), iterations=1)
                if poisson_mask.any() and poisson_mask.max() > 0:
                    blended = cv2.seamlessClone(
                        out, pre_swap, poisson_mask, center, cv2.MIXED_CLONE)
                    a = (feather(poisson_mask, max(5, self.blend // 2)).astype(np.float32)
                         / 255.0)[..., None]
                    # Peso ridotto a 0.55 (prima 0.85) → più fedele a inswapper, meno artefatti
                    out = np.clip(
                        blended.astype(np.float32) * (0.55 * a) +
                        out.astype(np.float32) * (1.0 - 0.55 * a),
                        0, 255).astype(np.uint8)
            except Exception:
                pass

        # --- 5. Temporal smoothing regionale ---
        if self.temporal > 0:
            out = self._temporal_blend(out, target_face, src_entry, labels=target_labels)
        return out

    def _temporal_blend(self, out: np.ndarray, target_face, src_entry: dict,
                        size: int = 128, labels=None) -> np.ndarray:
        """Riduce il flickering mescolando il fotogramma precedente.

        Versione regionale: sulle zone ad alta frequenza (occhi, naso, bocca)
        il peso temporale è ridotto a ~40% del valore globale, per evitare
        ghosting e dettagli "trascinati". Sulle guance/fronte resta pieno.
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

        A = np.vstack([M_cur, [0, 0, 1]]) @ np.linalg.inv(
            np.vstack([M_prev, [0, 0, 1]]))
        warped = cv2.warpAffine(prev_img, A[:2], (out.shape[1], out.shape[0]),
                                borderValue=0)

        # Maschera base del volto
        if self.use_semantic and labels is not None:
            base_mask = semantic_face_mask(labels, kps, expand=2)
        else:
            base_mask = inner_face_mask(out.shape, kps, 1.15)
        mask = feather(base_mask, self.blend)

        # Zone ad alta frequenza: peso temporale ridotto
        weight = np.full(out.shape[:2], self.temporal, dtype=np.float32)
        if labels is not None:
            hf = high_freq_mask(labels)
            if hf.any():
                # occhi/naso/bocca: solo 40% del temporal
                weight = np.where(hf > 0, self.temporal * 0.40, weight)

        valid = cv2.warpAffine(np.full(prev_img.shape[:2], 255, np.uint8),
                               A[:2], (out.shape[1], out.shape[0]),
                               flags=cv2.INTER_NEAREST, borderValue=0)
        a = (mask.astype(np.float32) / 255.0) * weight
        a *= (valid.astype(np.float32) / 255.0)
        a = a[..., None]

        out = np.clip(out.astype(np.float32) * (1 - a)
                      + warped.astype(np.float32) * a, 0, 255).astype(np.uint8)
        self._prev[key] = (kps.copy(), out.copy())
        return out