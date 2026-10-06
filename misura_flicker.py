"""Misura il flickering su un tratto di video dove le condizioni sono note.

Serve perche' la diagnostica generica non distingue due cose diverse:

  jitter geometrico  il detector (o l'incolla) sposta/scala il viso di
                      qualche pixel fra frame consecutivi. E' questo il
                      difetto che l'occhio legge come "scattoso"
  oscillazione identitaria  l'identita' del volto cambia fra frame

Entrambe si misurano meglio dopo aver rimosso il movimento reale: si
confronta ogni frame con una mediana a 5 frame, che assorbe i movimenti
veri della testa e lascia solo il tremito. Il confronto fra originale e
output dice quanto flickering ha aggiunto lo swap, non quanto ne aveva gia'
il video.
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

sys.path.insert(0, "dlfolder")


def _mediana_movimento(serie: np.ndarray, k: int = 5) -> np.ndarray:
    """Mediana centrata a finestra k: tiene i movimenti veri, toglie il rumore."""
    n = len(serie)
    out = np.empty_like(serie)
    h = k // 2
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        out[i] = np.median(serie[lo:hi], axis=0)
    return out


def _residuo(serie: np.ndarray, k: int = 5) -> np.ndarray:
    return serie - _mediana_movimento(serie, k)


def misura(video: str, da: int, a: int, campioni: int, scale: int):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(scale, scale))
    app.max_face_crop = False

    cap = cv2.VideoCapture(video)
    idx = 0
    kps, box, emb = [], [], []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if da <= idx < a:
            volti = app.get(img)
            if volti:
                # il volto piu' grande: su un tratto a persona unica e' quello
                # che ci interessa; i frame senza volti vengono saltati
                f = max(volti, key=lambda x: (x.bbox[2] - x.bbox[0]) *
                        (x.bbox[3] - x.bbox[1]))
                if f.kps is not None and f.normed_embedding is not None:
                    kps.append(np.asarray(f.kps, np.float32))
                    box.append(np.asarray(f.bbox, np.float32))
                    emb.append(np.asarray(f.normed_embedding, np.float32))
        idx += 1
        if idx >= a:
            break
    cap.release()

    if len(kps) < 8:
        print(f"[errore] solo {len(kps)} volti misurabili fra i frame {da}-{a}")
        return None

    # scarto di campionamento: se i frame sono radi, il residuo include
    # movimento reale e la misura peggiora da sola
    passo = max(1, len(kps) // max(campioni, 1))
    K = np.array(kps)[::passo]
    B = np.array(box)[::passo]
    E = np.array(emb)[::passo]
    n = len(K)

    # jitter geometrico: quanto dista ogni kps dalla mediana locale
    res_k = np.stack([_residuo(K[:, i, :], 5) for i in range(K.shape[1])], axis=1)
    jitter_kps = float(np.linalg.norm(res_k, axis=2).mean())

    scarto = _residuo(B, 5)
    lato = np.median(B[:, 2] - B[:, 0])
    jitter_pos = float(np.linalg.norm(scarto[:, :2], axis=1).mean())
    jitter_scala = float((np.abs(scarto[:, 2:]).mean() / lato) * 100.0)

    # vibrazione d'identita'
    d = np.abs(np.diff(E, axis=0)).sum(axis=1)
    osc_ident = float(d.mean())

    # salto brusco: quanto spesso l'identita' cambia piu' della norma
    return {
        "frame": n,
        "lato_px": float(lato),
        "jitter_kps_px": jitter_kps,
        "jitter_pos_px": jitter_pos,
        "jitter_scala_pct": jitter_scala,
        "osc_identita": osc_ident,
        "salto_max": float(d.max()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", nargs="+", required=True,
                    help="uno o piu' video, nell'ordine originale/variazioni")
    ap.add_argument("--da", type=int, default=0)
    ap.add_argument("--a", type=int, default=120)
    ap.add_argument("--campioni", type=int, default=400)
    ap.add_argument("--det-size", type=int, default=640)
    args = ap.parse_args()

    risultati = []
    for v in args.video:
        r = misura(v, args.da, args.a, args.campioni, args.det_size)
        if r:
            risultati.append((v, r))

    if not risultati:
        return 1

    righe = ["video                                jitter_kps  jitter_pos  jitter_scala  osc_ident  salto_max"]
    for v, r in risultati:
        righe.append(
            f"{v.split('/')[-1][:34]:<35} "
            f"{r['jitter_kps_px']:>10.3f}  {r['jitter_pos_px']:>10.3f}  "
            f"{r['jitter_scala_pct']:>12.3f}%  {r['osc_identita']:>9.4f}  "
            f"{r['salto_max']:>9.4f}")

    base = risultati[0][1]
    if len(risultati) > 1:
        righe.append("")
        righe.append("eccesso rispetto all'originale (quanto flickering aggiunge lo swap)")
        righe.append("video                                jitter_kps  jitter_pos  jitter_scala  osc_ident")
        for v, r in risultati[1:]:
            righe.append(
                f"{v.split('/')[-1][:34]:<35} "
                f"{r['jitter_kps_px']-base['jitter_kps_px']:>+10.3f}  "
                f"{r['jitter_pos_px']-base['jitter_pos_px']:>+10.3f}  "
                f"{r['jitter_scala_pct']-base['jitter_scala_pct']:>+11.3f}%  "
                f"{r['osc_identita']-base['osc_identita']:>+9.4f}")

    print("\n".join(righe))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
