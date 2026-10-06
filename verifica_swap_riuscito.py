#!/usr/bin/env python3
"""Verifica che lo head swap sia realmente avvenuto, persona per persona.

Se per un volto la somiglianza con la foto assegnata NON aumenta rispetto
allo stesso volto nel frame originale, lo swap e' fallito in silenzio: in
video resta il soggetto di partenza. E' il caso da cui spiegazioni come
"nel video c'e' un uomo con gli occhiali che nella foto non ci sono": non
sono occhiali sopravvissuti, e' una faccia mai sostituita.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "dlfolder"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from modules.face_analyser import get_many_faces  # noqa: E402

SOGLIA = 10.0      # px di distanza inter-oculare sotto la quale non si misura
MINIMO = 30.0      # idem, in px: sotto questa scala il confronto e' rumore


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def distanza_oculare(kps) -> float:
    return float(np.linalg.norm(np.asarray(kps[0]) - np.asarray(kps[1])))


def main() -> int:
    ap = argparse.ArgumentParser(description="Lo swap e' avvenuto davvero?")
    ap.add_argument("--originale", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--photo", required=True, nargs="+", type=Path)
    ap.add_argument("--passo", type=int, default=3, help="campiona ogni N frame")
    a = ap.parse_args()

    # embedding delle foto di riferimento
    foto: list[tuple[str, np.ndarray]] = []
    for p in a.photo:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[!] foto non leggibile: {p}")
            continue
        fs = get_many_faces(img)
        if not fs:
            print(f"[!] nessun volto nella foto: {p.name}")
            continue
        f = max(fs, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        foto.append((p.name, f.normed_embedding))
    if not foto:
        print("[!] nessuna foto utilizzabile")
        return 1
    nomi = [n for n, _ in foto]
    E = np.vstack([e for _, e in foto])

    co = cv2.VideoCapture(str(a.originale))
    cn = cv2.VideoCapture(str(a.output))
    if not co.isOpened() or not cn.isOpened():
        print("[!] video non leggibile")
        return 1

    campioni: list[dict] = []
    i = 0
    while True:
        ok_o, f_o = co.read()
        ok_n, f_n = cn.read()
        if not ok_o or not ok_n:
            break
        if i % a.passo == 0:
            vo = get_many_faces(f_o)
            vn = get_many_faces(f_n)
            for fo in vo:
                if distanza_oculare(fo.kps) < MINIMO:
                    continue
                partner = next((fn for fn in vn if iou(fo.bbox, fn.bbox) > 0.5), None)
                if partner is None or distanza_oculare(partner.kps) < MINIMO:
                    continue
                so = E @ fo.normed_embedding
                sn = E @ partner.normed_embedding
                campioni.append({
                    "foto_o": nomi[int(np.argmax(so))], "s_o": float(np.max(so)),
                    "foto_n": nomi[int(np.argmax(sn))], "s_n": float(np.max(sn)),
                    "distanza": distanza_oculare(fo.kps),
                })
        i += 1
    co.release()
    cn.release()

    if not campioni:
        print("[!] nessun volto confrontabile (video troppo piccolo o detection fallita)")
        return 1

    print("=" * 74)
    print(f"  SWAP RIUSCITO?   {a.output.name}")
    print("=" * 74)
    print(f"  coppie di volti confrontate : {len(campioni)}")
    print(f"  somiglianza media con la foto PRIMA : {np.mean([c['s_o'] for c in campioni]):.3f}")
    print(f"  somiglianza media con la foto DOPO  : {np.mean([c['s_n'] for c in campioni]):.3f}")
    print(f"  miglioramento medio            : "
          f"{np.mean([c['s_n'] - c['s_o'] for c in campioni]):+.3f}")
    print()

    print("  per foto di riferimento:")
    for n in nomi:
        sub = [c for c in campioni if c["foto_o"] == n]
        if not sub:
            continue
        d = np.mean([c["s_n"] - c["s_o"] for c in sub])
        stessa = sum(1 for c in sub if c["foto_n"] == c["foto_o"]) / len(sub)
        stato = "ok" if d > 0.02 else ("INVARIATO" if d > -0.02 else "PEGGIORATO")
        print(f"    {n:<16} volti={len(sub):<4} delta={d:+.3f}  "
              f"stessa foto {stessa:.0%}  -> {stato}")

    peggiori = [c for c in campioni if c["s_n"] - c["s_o"] <= 0.0]
    print()
    if peggiori:
        print(f"  ! {len(peggiori)}/{len(campioni)} volti NON sono diventati piu' "
              f"simili alla foto")
        for c in peggiori[:5]:
            print(f"      {c['foto_o']} -> {c['foto_n']}  "
                  f"{c['s_o']:.3f} -> {c['s_n']:.3f}  (oculare {c['distanza']:.0f}px)")
        print("  -> in questi casi il volto in video e' ancora quello del video:")
        print("     lo swap non e' avvenuto, non e' un accessorio sopravvissuto.")
    else:
        print("  -> tutti i volti sono diventati piu' simili alla foto assegnata")
    return 0


if __name__ == "__main__":
    sys.exit(main())
