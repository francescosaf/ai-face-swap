"""Verifica se un output e' un vero HEAD SWAP o solo un FACE SWAP.

La differenza non e' una questione di sfumature: e' che regioni dell'immagine
sono state riscritte.

  face swap    cambia il volto, lascia capelli, collo e sfondo del video
  head swap    cambia volto E capelli, lascia collo, sfondo e movimento

Per saperlo si misura, regione per regione, la frazione di pixel che il
processo ha modificato in modo netto rispetto all'originale. Le regioni sono
quelle di BiSeNet, non quadrati a caso.

Uso:
    python verifica_headswap.py --originale VIDEO --output VIDEO_ALLOCCATO
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dlfolder"))

from face_parsing import FaceParser, FACE_CLASSES, HAIR_CLASSES  # noqa: E402
from modules.face_analyser import get_many_faces  # noqa: E402

# una differenza oltre questa soglia conta come pixel "riscritto": il rumore
# di compressione resta sotto il 12, un volto trapiantato ben sopra il 30
SOGLIA_CAMBIO = 22


def regioni(labels: np.ndarray, capelli, volto, capigliatura):
    return {
        "volto": np.isin(labels, list(volto)),
        "capelli": np.isin(labels, list(capelli)),
        "testa_capelli": np.isin(labels, list(capigliatura)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--originale", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--campioni", type=int, default=12)
    a = ap.parse_args()

    parser = FaceParser(ROOT / "models")
    vo, vs = cv2.VideoCapture(str(a.originale)), cv2.VideoCapture(str(a.output))
    n = min(int(vo.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(vs.get(cv2.CAP_PROP_FRAME_COUNT)))
    vo.release()
    vs.release()
    indici = np.linspace(0, n - 1, a.campioni).astype(int)

    acc = {"volto": [], "capelli": [], "testa_capelli": [], "esterno": []}

    for i in indici:
        vo = cv2.VideoCapture(str(a.originale))
        vo.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok_o, f_o = vo.read()
        vo.release()
        vs = cv2.VideoCapture(str(a.output))
        vs.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok_s, f_s = vs.read()
        vs.release()
        if not (ok_o and ok_s):
            continue
        volti = get_many_faces(f_o)
        if not volti:
            continue
        face = max(volti, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        labels, _ = parser.parse(f_o, face.bbox)

        cambiato = (cv2.absdiff(f_o, f_s).max(axis=2) > SOGLIA_CAMBIO)
        r = regioni(labels, HAIR_CLASSES, FACE_CLASSES, HAIR_CLASSES | FACE_CLASSES)

        def quota(m):
            n = int(m.sum())
            return float(cambiato[m].sum()) / n if n else float("nan")

        acc["volto"].append(quota(r["volto"]))
        acc["capelli"].append(quota(r["capelli"]))
        acc["testa_capelli"].append(quota(r["testa_capelli"]))
        # tutto quello che sta fuori dalla testa: non deve essere toccato,
        # altrimenti perdo movimento, sfondo e audio di scena
        testa = np.isin(labels, list(HAIR_CLASSES) + list(FACE_CLASSES))
        acc["esterno"].append(quota(~testa))

    def m(x):
        return float(np.nanmean(x)) if len(x) else float("nan")

    v, c, t, e = m(acc["volto"]), m(acc["capelli"]), m(acc["testa_capelli"]), m(acc["esterno"])

    print("=" * 68)
    print(f"  Cosa e' cambiato davvero   {a.output.name}")
    print("=" * 68)
    print(f"  volto       riscritto     : {v:6.1%}")
    print(f"  capelli     riscritti     : {c:6.1%}")
    print(f"  testa intera riscritta    : {t:6.1%}")
    print(f"  esterno     toccato       : {e:6.1%}")
    print("  " + "-" * 66)
    if c < 0.15:
        print("  -> FACE SWAP: i capelli restano quelli del video")
    else:
        print("  -> HEAD SWAP: anche la capigliatura viene dalla foto")
    if e > 0.08:
        print(f"  ! ATTENZIONE: {e:.0%} di tutto il resto e' stato toccato")
    else:
        print(f"  movimento, collo e sfondo intatti ({e:.0%})")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())