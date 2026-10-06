#!/usr/bin/env python3
"""Swapper per modelli a 256px: HyperSwap e HIFiFace.

inswapper_128 lavora a 112px e soprattutto condiziona sulla struttura del
volto target: per questo conserva gli accessori del video (occhiali,
montatura) e la capigliatura resta quella del video, ricolorata. Non e' un
limite di blending, e' il modello.

HyperSwap e HIFiFace scambiano un crop 256 dell' intera testa e restituiscono
anche una `mask`: la si puo' usare al posto di una maschera geometrica, e
questo e' il punto per cui sparisce l'effetto sticker.

Firma (verificata ispezionando i .onnx):
    IN  source  [1, 512]   embedding normalizzato del volto di riferimento
    IN  target  [1, 3, 256, 256]
    OUT output  [1, 3, 256, 256]
    OUT mask    [1, 1, 256, 256]   (HyperSwap e HIFiFace)

Normalizzazione: RGB, /255, (x - 0.5) / 0.5, poi HWC -> CHW.

Perche' esiste la scaletta di tentativi
--------------------------------------
Su un volto di ~100px in un video 720x1280 HyperSwap fallisce senza
avvisi: l'inferenza termina con successo, la maschera e' valida, ma il
risultato e' identico all'input e nel video resta il viso originale con gli
occhiali. La causa misurata non e' la dimensione del volto ma la sfocatura
da movimento: i frame in cui fallisce hanno nitidezza (varianza del
Laplaciano) di 30-66, quelli in cui funziona di 79-145.

Ingrandire il ritaglio NON risolve e va tenuto disattivato: l'input esce
dalla distribuzione del modello, la maschera torna quasi a zero e lo swap
non viene applicato (diff medio 0.46/255, 6 pixel cambiati su 11592).

Quello che funziona e' la sfocatura inversa (unsharp mask) del ritaglio
prima dell'inferenza, con tentativi progressivi: sui frame sfocati porta lo
scambio da 1-4% di pixel cambiati a 16-30%, mentre sui frame gia' nitidi
non cambia il risultato.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from face_templates import warp_template  # noqa: E402

MEDIA = 0.5
DEVIAZIONE = 0.5

# Nel crop da 256 con template arcface_128 gli occhi stanno a x=92 e x=163,
# quindi l'oculare "di riferimento" del modello e' di circa 70px.
OCULARE_RIF = 70.0
# disattivato: vedi la nota sullo zoom nel docstring
ZOOM_MAX = 1.0

# Scaletta di recupero: (sigma, quantita') della sfocatura inversa.
# La prima voce e' (0, 0), cioe' nessuna correzione: i frame gia' buoni
# vengono scambiati al primo colpo e restano identici a prima.
DEBLUR = ((0.0, 0.0), (1.5, 1.2), (2.0, 1.5))


class Swapper256:
    """Swapper a 256px con maschera del modello."""

    def __init__(self, model_path: Path, size: int = 256,
                 feather: float = 6.0, restringi: float = 1.0, zoom: float = 1.0,
                 soglia: float = 0.08, deblur=DEBLUR):
        so = ort.SessionOptions()
        # i modelli hanno centinaia di initializer anche fra gli input: senza
        # questo onnxruntime emette migliaia di warning e rallenta l'avvio
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), so,
                                            providers=["CPUExecutionProvider"])
        self.size = size
        # il template arcface_128 e' tarato su un crop 256: con size diverso il
        # volto non e' centrato bene, quindi si resta a 256
        self.template = warp_template(size)
        self.feather = float(feather)
        self.restringi = float(restringi)
        self.zoom_max = float(np.clip(zoom, 1.0, 3.0))
        self.soglia = float(soglia)
        self.deblur = tuple(tuple(v) for v in deblur)
        # ultimo scambio riuscito (testa 256 + sua maschera), per i frame in
        # cui nessun tentativo funziona
        self._ultimo_buono = None
        self._falliti = 0
        self._riusi = 0
        self._deblur_usati = 0
        ingressi = {i.name: i for i in self.session.get_inputs()}
        if not {"source", "target"} <= set(ingressi):
            raise ValueError(f"ingressi inattesi: {sorted(ingressi)}")
        # l'ordine degli input non e' garantito: la forma va letta da 'target'
        tgt = ingressi["target"].shape
        h, w = int(tgt[-2]), int(tgt[-1])
        if (w, h) != (size, size):
            raise ValueError(f"il modello vuole {w}x{h}, non {size}x{size}")

    # --- utility ------------------------------------------------------
    @staticmethod
    def _cambiata(a: np.ndarray, b: np.ndarray, bbox) -> float:
        """Frazione di pixel del bbox cambiati oltre la soglia.

        Serve a capire se il modello ha davvero scambiato. Sotto 8% lo swap
        e' da considerare fallito anche se l'inferenza e' andata a buon
        fine: sui frame reali si osservano 1-4% quando fallisce e 15-34%
        quando funziona.
        """
        h, w = a.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return 1.0
        if a.shape != b.shape:
            return 1.0
        d = np.abs(a[y1:y2, x1:x2].astype(np.int16)
                   - b[y1:y2, x1:x2].astype(np.int16)).max(axis=2)
        return float((d > 10).mean())

    def _matrice(self, kps) -> np.ndarray:
        M, _ = cv2.estimateAffinePartial2D(
            np.asarray(kps, dtype=np.float32), self.template,
            method=cv2.RANSAC, ransacReprojThreshold=100)
        if M is None:
            raise ValueError("allineamento del volto fallito")
        return M

    def _zoom(self, kps, M) -> float:
        """Quanto ingrandire il ritaglio per arrivare all'oculare del modello."""
        reale = float(np.linalg.norm(np.asarray(kps)[0] - np.asarray(kps)[1]))
        if reale <= 1.0:
            return 1.0
        return float(np.clip(OCULARE_RIF / reale, 1.0, self.zoom_max))

    def _matrice_e_zoom(self, kps) -> np.ndarray:
        M = self._matrice(kps)
        z = self._zoom(kps, M)
        if z > 1.0:
            c = (self.size - 1) / 2.0
            Z = np.array([[z, 0.0, (1.0 - z) * c],
                          [0.0, z, (1.0 - z) * c],
                          [0.0, 0.0, 1.0]], dtype=np.float32)
            # M e' 2x3: la composizione va fatta in coordinate omogenee
            M = (Z @ np.vstack([M, [0.0, 0.0, 1.0]]))[:2]
        return M

    @staticmethod
    def _nitido(ritaglio: np.ndarray, sigma: float, q: float) -> np.ndarray:
        if sigma <= 0 or q <= 0:
            return ritaglio
        sfocato = cv2.GaussianBlur(ritaglio, (0, 0), sigma)
        return np.clip(ritaglio.astype(np.float32) * (1.0 + q)
                       - sfocato.astype(np.float32) * q, 0, 255).astype(np.uint8)

    def _infernza(self, frame: np.ndarray, M: np.ndarray,
                  emb: np.ndarray, sigma: float, q: float):
        """Ritaglia, corregge la sfocatura e infersce. Ritorna testa+maschera."""
        ritaglio = cv2.warpAffine(frame, M, (self.size, self.size),
                                  flags=cv2.INTER_LINEAR)
        ritaglio = self._nitido(ritaglio, sigma, q)
        x = ritaglio[:, :, ::-1].astype(np.float32) / 255.0
        x = (x - MEDIA) / DEVIAZIONE
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)

        uscite = self.session.run(None, {"source": emb, "target": x})
        out = uscite[0][0].transpose(1, 2, 0)          # CHW -> HWC
        maschera = uscite[1][0, 0] if len(uscite) > 1 else None

        out = out * DEVIAZIONE + MEDIA
        out = np.clip(out, 0, 1)[:, :, ::-1] * 255.0
        out = np.ascontiguousarray(out.astype(np.uint8))

        # la maschera del modello dice dove il modello ha ricostruito: si
        # sfuma, altrimenti il bordo riflesso torna a sembrare un ritaglio
        if maschera is None:
            maschera = np.ones((self.size, self.size), np.float32)
        maschera = np.clip(maschera.astype(np.float32), 0, 1)
        if self.restringi != 1.0:
            k = self.restringi
            c = (self.size - 1) / 2.0
            X, Y = np.meshgrid(np.arange(self.size), np.arange(self.size))
            maschera = maschera * (((X - c) / (c * k)) ** 2
                                   + ((Y - c) / (c * k)) ** 2 <= 1.0).astype(np.float32)
        if self.feather > 0:
            maschera = np.clip(
                cv2.GaussianBlur(maschera, (0, 0), self.feather), 0, 1)
        return out, maschera, ritaglio

    def _incolla(self, frame: np.ndarray, testa: np.ndarray,
                 maschera: np.ndarray, inv: np.ndarray, dim) -> np.ndarray:
        a = cv2.warpAffine(maschera, inv, dim,
                           flags=cv2.INTER_LINEAR)[..., None].astype(np.float32)
        return np.clip(frame.astype(np.float32) * (1.0 - a)
                       + cv2.warpAffine(testa, inv, dim,
                                        flags=cv2.INTER_LINEAR).astype(np.float32) * a,
                       0, 255).astype(np.uint8)

    # --- API ----------------------------------------------------------
    def swap(self, source_face, target_face, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        M = self._matrice_e_zoom(target_face.kps)
        inv = cv2.invertAffineTransform(M)
        emb = np.ascontiguousarray(
            np.asarray(source_face.normed_embedding,
                       dtype=np.float32).reshape(1, -1))
        bbox = target_face.bbox

        # primo tentativo senza correzioni: i frame gia' buoni si fermano qui
        testa, maschera, _ = self._infernza(frame, M, emb, 0.0, 0.0)
        out = self._incolla(frame, testa, maschera, inv, (w, h))
        if self._cambiata(frame, out, bbox) >= self.soglia:
            self._ultimo_buono = (testa, maschera)
            return out

        # il volto e' sfocato: si prova a ripulirlo prima di arrendersi
        for sigma, q in self.deblur[1:]:
            testa2, maschera2, _ = self._infernza(frame, M, emb, sigma, q)
            out2 = self._incolla(frame, testa2, maschera2, inv, (w, h))
            if self._cambiata(frame, out2, bbox) >= self.soglia:
                self._deblur_usati += 1
                self._ultimo_buono = (testa2, maschera2)
                return out2

        # nessun tentativo ha scambiato: meglio una testa che "congela" di un
        # fotogramma che lascia tornare il viso originale con gli occhiali.
        # Si riusa l'ultimo scambio riuscito, riallineato alla posizione
        # attuale, portando con se' anche la sua maschera.
        self._falliti += 1
        if self._ultimo_buono is not None:
            testa3, maschera3 = self._ultimo_buono
            riuso = self._incolla(frame, testa3, maschera3, inv, (w, h))
            if self._cambiata(frame, riuso, bbox) >= self.soglia:
                self._riusi += 1
                return riuso
        return out


MODELLI = {
    "hyperswap": "swapper/hyperswap_1a_256.onnx",
    "hififace": "hififace_unofficial_256.onnx",
}


def carica(nome: str, models_dir: Path, **kw) -> Swapper256:
    if nome not in MODELLI:
        raise ValueError(f"modello sconosciuto: {nome} (scegli fra "
                         f"{', '.join(MODELLI)})")
    return Swapper256(Path(models_dir) / MODELLI[nome], **kw)