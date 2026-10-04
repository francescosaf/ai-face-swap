"""Misura gli aloni del trapianto dei capelli.

Il difetto: incollando la capigliatura della foto sopra quella del video con
una maschera morbida, si ottiene una "maschera semitrasparente sopra". Lo si
vede come un alone chiaro o scuro lungo il bordo della capigliatura.

La causa misurabile non e' la sfumatura, e' il fatto che il trapianto venga
incollato dove nel video **non c'erano capelli**: sulla fronte, sulla pelle,
sullo sfondo. Ogni pixel incollato fuori dalla silhuette dei capelli
originali introduce un bordo nuovo, e un bordo nuovo contro lo sfondo e'
esattamente un alone.

Tre misure:

  fuori_silhouette  quanto e' stato scritto dove non c'erano capelli.
                    Questa e' la causa diretta dell'alone: piu' e' alto,
                    piu' l'alone e' garantito.
  alone             eccesso di gradiente sulla fascia che contorna la
                    capigliatura, rispetto all'originale.
  residuo           quanto del video originale resta visibile dentro i capelli.
                    Un trapianto morbido lascia trasparire l'originale:
                    residuo alto = effetto "velo" sopra.

Uso:
    python verifica_aloni.py --originale VIDEO --output OUT.mp4
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

from face_parsing import FaceParser, HAIR_CLASSES, inner_face_mask  # noqa: E402
from modules.face_analyser import get_many_faces  # noqa: E402

SOGLIA_CAMBIO = 22
K = np.ones((7, 7), np.uint8)

# oltre questi valori l'alone e' presente
SOGLIE = {"fuori_silhouette": 0.06, "alone": 8.0, "residuo": 0.45}


def pendenza(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--originale", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--campioni", type=int, default=16)
    a = ap.parse_args()

    parser = FaceParser(ROOT / "models", ["CPUExecutionProvider"])
    vo, vs = cv2.VideoCapture(str(a.originale)), cv2.VideoCapture(str(a.output))
    n = min(int(vo.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(vs.get(cv2.CAP_PROP_FRAME_COUNT)))
    vo.release()
    vs.release()
    indici = np.linspace(0, n - 1, a.campioni).astype(int)

    fuori, alone_v, residuo_v, capelli_v = [], [], [], []

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

        capelli = parser.mask_for(labels, HAIR_CLASSES)
        if not capelli.any():
            continue

        # fascia che contorna la capigliatura, dentro e fuori
        esterno = cv2.dilate(capelli, K)
        fascia = cv2.bitwise_and(esterno, cv2.bitwise_not(cv2.erode(capelli, K)))
        interno = cv2.erode(capelli, K)
        if not interno.any():
            continue

        cambiato = cv2.absdiff(f_o, f_s).max(axis=2) > SOGLIA_CAMBIO

        # quanto e' stato scritto FUORI dalla capigliatura, escludendo il
        # volto (che inswapper cambia legittimamente) e la pelle
        fuori_capelli = cv2.bitwise_and(esterno,
                                        cv2.bitwise_not(
                                            parser.mask_for(labels, (6, 7, 8, 9, 10))))
        if int(fuori_capelli.sum()) > 200:
            fuori.append(float(cambiato[fuori_capelli > 0].mean()))

        if int(capelli.sum()) > 200:
            capelli_v.append(float(cambiato[capelli > 0].mean()))

        # alone: gradiente in eccesso sulla fascia, rispetto all'originale
        if fascia.any():
            po = float(pendenza(f_o)[fascia > 0].mean())
            ps = float(pendenza(f_s)[fascia > 0].mean())
            alone_v.append(ps - max(po, 1.0))

        # residuo: quanta struttura del video resta visibile dentro i capelli
        po = cv2.cvtColor(f_o, cv2.COLOR_BGR2GRAY).astype(np.float32)[interno > 0]
        ps = cv2.cvtColor(f_s, cv2.COLOR_BGR2GRAY).astype(np.float32)[interno > 0]
        if po.std() > 1e-6 and ps.std() > 1e-6:
            residuo_v.append(float(np.corrcoef(po, ps)[0, 1]))

    def m(x):
        return float(np.mean(x)) if x else float("nan")

    fs_, al, rs, cp = m(fuori), m(alone_v), m(residuo_v), m(capelli_v)

    print("=" * 70)
    print(f"  A L O N I   {a.output.name}")
    print("=" * 70)
    print(f"  capelli cambiati            : {cp:6.1%}")
    print(f"  scritto FUORI dalla capigliatura : {fs_:6.1%}   <- causa dell'alone")
    print(f"  alone in eccesso sul bordo  : {al:+6.1f}")
    print(f"  residuo del video nei capelli: {rs:6.2f}")
    print("  " + "-" * 68)
    problemi = []
    if fs_ > SOGLIE["fuori_silhouette"]:
        problemi.append(f"ALONE: scritto il {fs_:.0%} fuori dalla capigliatura "
                       f"(soglia {SOGLIE['fuori_silhouette']:.0%})")
    if al > SOGLIE["alone"]:
        problemi.append(f"ALONE: bordo {al:+.1f} piu' tagliente "
                       f"(soglia {SOGLIE['alone']:.0f})")
    if rs > SOGLIE["residuo"]:
        problemi.append(f"VELO: resta il {rs:.0%} del video originale dentro i capelli "
                       f"(soglia {SOGLIE['residuo']:.0%})")
    for p in problemi:
        print(f"  ! {p}")
    if not problemi:
        print("  nessun alone rilevato")
    print("=" * 70)
    return 1 if problemi else 0


if __name__ == "__main__":
    sys.exit(main())