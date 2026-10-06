#!/usr/bin/env python3
"""Prova uno swapper su un singolo frame, con misura oggettiva.

Non serve guardare l'immagine: si misura quanto il volto e' diventato
simile alla foto di riferimento. Serve anche a intercettare i "FAIL swap" di
HyperSwap, che su alcuni volti producono output validi ma senza alcuno scambio.
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

MODELLI = "hyperswap,hififace,inswapper"


def embedding_photo(percorso: Path):
    img = cv2.imread(str(percorso))
    if img is None:
        raise SystemExit(f"foto non leggibile: {percorso}")
    fs = get_many_faces(img)
    if not fs:
        raise SystemExit(f"nessun volto nella foto: {percorso.name}")
    f = max(fs, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return f, f.normed_embedding


def main() -> int:
    ap = argparse.ArgumentParser(description="Prova uno swapper su un frame")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--photo", required=True, nargs="+", type=Path)
    ap.add_argument("--frame", type=int, default=66)
    ap.add_argument("--swapper", default="hyperswap")
    ap.add_argument("--modelli", default=MODELLI)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    cap = cv2.VideoCapture(str(a.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"frame {a.frame} non leggibile")

    foto = [embedding_photo(p) for p in a.photo]
    E = np.vstack([e for _, e in foto])
    nomi = [p.name for p in a.photo]

    volti = get_many_faces(img)
    if not volti:
        raise SystemExit("nessun volto nel frame")
    print(f"frame {a.frame}: {len(volti)} volti")
    for n, (f, _) in zip(nomi, foto):
        print(f"  foto {n}: oculare {np.linalg.norm(f.kps[0]-f.kps[1]):.0f}px")

    modelli = [m.strip() for m in a.modelli.split(",") if m.strip()]
    solo = a.swapper if a.swapper else None
    if solo:
        modelli = [solo]

    risultati = {}
    for nome in modelli:
        try:
            if nome == "inswapper":
                from modules.processors.frame.face_swapper import get_frame_swapper
                scambio = get_frame_swapper()
                scambio.load_model()
                fn = lambda s, t, im, _f=scambio: _f.swap_face(s, t, im)
            else:
                import hyper_swap
                sw = hyper_swap.carica(nome, Path("dlfolder/models"))
                fn = lambda s, t, im, _s=sw: _s.swap(s, t, im)
        except Exception as e:
            print(f"\n{nome}: non caricabile ({type(e).__name__}: {e})")
            continue

        risultati[nome] = {}
        for vi, target in enumerate(volti):
            fonte, _ = foto[min(vi, len(foto) - 1)]
            prima = float(np.max(E @ target.normed_embedding))
            try:
                out = fn(fonte, target, img)
            except Exception as e:
                print(f"\n{nome} volto {vi}: errore {type(e).__name__}: {e}")
                continue
            dopo_fs = get_many_faces(out)
            m = max(dopo_fs, key=lambda x: (x.bbox[2] - x.bbox[0])
                    * (x.bbox[3] - x.bbox[1])) if dopo_fs else None
            dopo = float(np.max(E @ m.normed_embedding)) if m else float("nan")
            stessa = int(np.argmax(E @ m.normed_embedding)) if m else -1
            risultati[nome][vi] = out
            stato = "ok" if dopo - prima > 0.05 else "NESSUNO SCAMBIO"
            print(f"\n{nome} volto {vi} -> foto {nomi[min(vi, len(foto)-1)]}")
            print(f"  somiglianza {prima:.3f} -> {dopo:.3f}  ({dopo-prima:+.3f})"
                  f"  ora piu' vicino a: {nomi[stessa] if stessa >= 0 else '?'}"
                  f"   {stato}")

    if a.out and risultati:
        righe = []
        for nome, per_volo in risultati.items():
            riga = [img]
            for vi in sorted(per_volo):
                r = per_volo[vi]
                x1, y1, x2, y2 = [int(v) for v in volti[vi].bbox]
                pad = int(max(x2 - x1, y2 - y1) * 0.6)
                h, w = r.shape[:2]
                t = r[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
                t = cv2.resize(t, (260, 340))
                cv2.putText(t, nome, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .6,
                            (0, 255, 255), 2)
                riga.append(t)
            while len(riga) < 1 + len(volti):
                riga.append(np.zeros_like(riga[0]))
            righe.append(np.hstack(riga[:1 + len(volti)]))
        hmin = min(r.shape[0] for r in righe)
        cv2.imwrite(str(a.out), np.vstack([r[:hmin] for r in righe]))
        print(f"\nscritto {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
