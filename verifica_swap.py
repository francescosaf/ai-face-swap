#!/usr/bin/env python3
"""Verifica oggettiva di un faceswap/headswap.

Guardare il video non basta: un risultato puo' sembrare plausibile ed essere
una sovrapposizione statica, oppure un'immagine distrutta. La metrica decisiva
e' la DISTANZA DI IDENTITA': se lo swap funziona, il volto in output deve
avvicinarsi alla foto di riferimento.

  identita' residua = 1 - coseno(embedding(volto in output), embedding(foto))

  valore alto (~0.9) = il volto e' ancora quello originale, nessuno swap
  valore basso (~0.1) = il volto e' stato sostituito

Attenzione: confrontare due video gia' codificati dà numeri falsi. Con x264 i
blocchi vicini a una zona molto cambiata vengono riquantizzati diversamente, e
una differenza media di 32/255 su una fascia sottile puo' essere solo codice.
Per questo la metrica primaria e' l'identita', che e' indipendente dal codec.

Uso:
    python verifica_swap.py --videoorig v.mp4 --videoswap s.mp4 --rif foto1.jpg
    python verifica_swap.py --videoorig v.mp4 --videoswap s.mp4 --rif foto1.jpg \
                           --modalita headswap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dlfolder"))

IDENTITA_OK = 0.45      # sotto questa soglia il volto e' davvero stato sostituito
IDENTITA_MIGLIORE = 0.30
SFONDO_MAX = 8.0        # oltre, il composite tocca il fondo
DISTRUZIONE_MAX = 130.0  # oltre, la zona volto e'rumore


def grab(path: Path, idx: int):
    cap = cv2.VideoCapture(str(path))
    if idx < 0:
        idx = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def identita(img, e_rif):
    from modules.face_analyser import get_one_face
    f = get_one_face(img)
    if f is None or f.normed_embedding is None:
        return None
    return 1.0 - float(f.normed_embedding @ e_rif)


def bbox_face(img):
    from modules.face_analyser import get_many_faces
    fs = get_many_faces(img)
    if not fs:
        return None
    return max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).bbox


def maschera(shape, box, pad=0.35):
    H, W = shape[:2]
    x1, y1, x2, y2 = box[:4]
    bw, bh = x2 - x1, y2 - y1
    m = np.zeros(shape[:2], bool)
    m[int(max(0, y1 - bh * pad)):int(min(H, y2 + bh * pad)),
      int(max(0, x1 - bw * pad)):int(min(W, x2 + bw * pad))] = True
    return m


def mad(a, b, m=None):
    if m is not None:
        a, b = a[m], b[m]
    if a.size == 0:
        return 0.0
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description="Verifica oggettiva di un faceswap.")
    ap.add_argument("--videoorig", type=Path, required=True)
    ap.add_argument("--videoswap", type=Path, required=True)
    ap.add_argument("--rif", type=Path, required=True, help="foto di riferimento usata")
    ap.add_argument("--modalita", choices=["faceswap", "headswap"], default="faceswap")
    ap.add_argument("--frame", nargs="*", type=int, default=[0, 5, 10, 15, 19],
                    help="indici dei frame da confrontare")
    args = ap.parse_args()

    from modules.face_analyser import get_one_face
    rif = cv2.imread(str(args.rif))
    if rif is None:
        print(f"[errore] foto non leggibile: {args.rif}")
        return 1
    e_rif = get_one_face(rif).normed_embedding

    campioni = []
    for i in args.frame:
        a = grab(args.videoorig, i)
        b = grab(args.videoswap, i)
        if a is None or b is None:
            continue
        campioni.append((i, a, b))

    if not campioni:
        print("[errore] nessun frame confrontabile")
        return 1

    print("=" * 70)
    print(f"  VERIFICA {args.modalita.upper()}   riferimento: {args.rif.name}")
    print("=" * 70)
    print(f"  {'frame':>6} {'orig vs rif':>13} {'swap vs rif':>13} {'variazione':>12}")
    print("  " + "-" * 46)
    d_o, d_s, d_face, d_bg = [], [], [], []
    for i, a, b in campioni:
        oa, ob = identita(a, e_rif), identita(b, e_rif)
        box = bbox_face(a)
        if oa is None or ob is None or box is None:
            print(f"  {i:>6} {'volto non rilevato':>13}")
            continue
        fm = maschera(a.shape, box)
        d_o.append(oa); d_s.append(ob)
        d_face.append(mad(a, b, fm)); d_bg.append(mad(a, b, ~fm))
        print(f"  {i:>6} {oa:>13.3f} {ob:>13.3f} {oa-ob:>12.3f}")
    print("  " + "-" * 46)

    if not d_s:
        print("[errore] nessun volto rilevato nell'output")
        return 1

    mo, ms = float(np.mean(d_o)), float(np.mean(d_s))
    print(f"\n  identita' residua media : originale {mo:.3f} -> swap {ms:.3f}")
    print(f"  zona volto  (pixel)     : {np.mean(d_face):.2f}")
    print(f"  fuori volto (pixel)     : {np.mean(d_bg):.2f}")

    esito = []
    if ms >= IDENTITA_OK:
        esito.append(f"FALLITO: identita' residua {ms:.3f} >= {IDENTITA_OK}, "
                     f"il volto non e' stato sostituito")
    if mo - ms < 0.3:
        esito.append("FALLITO: il volto si e' mosso di poco, somiglia a un no-op")
    if np.mean(d_bg) > SFONDO_MAX:
        esito.append(f"FALLITO: fuori dal volto cambia di {np.mean(d_bg):.2f}, "
                     f"il composite tocca il fondo")
    if np.mean(d_face) > DISTRUZIONE_MAX:
        esito.append(f"FALLITO: la zona volto e' distrutta ({np.mean(d_face):.2f})")
    if args.modalita == "headswap" and np.mean(d_face) < 12:
        esito.append("ATTENZIONE: i capelli non sembrano trasferiti, "
                     "la zona testa cambia poco")

    print("\n  " + "=" * 46)
    if esito:
        for e in esito:
            print(f"  {e}")
        print("\n  ESITO: da correggere")
        return 1
    giudizio = "ottimo" if ms < IDENTITA_MIGLIORE else "accettabile"
    print(f"  volto sostituito, sfondo intatto, qualita' {giudizio}")
    print("\n  ESITO: coerente")
    return 0


if __name__ == "__main__":
    sys.exit(main())