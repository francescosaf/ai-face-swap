"""Individua chi ha i capelli lunghi e se la maglietta sotto e' recuperabile.

Due domande, una per frame:

1. Chi ha i capelli lunghi? Non serve guardare: si misura quanta capigliatura
   sta *sotto il mento*. I capelli lunghi invadono spalle e maglietta, quelli
   corti no.
2. La maglietta coperta e' mai visibile in un altro frame? Se la persona gira
   la testa, la porzione di maglietta che i capelli coprono in un frame
   appare scoperta in un altro. In quel caso una mediana temporale la
   ricostruisce senza bisogno di un modello di inpainting.

Il confronto e' fatto in un sistema di riferimento allineato sui landmark del
volto: le maschere dei capelli di frame diversi non sono confrontabili cosi'
come sono, perche' la persona si muove.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dlfolder"))

from face_parsing import FaceParser, HAIR_CLASSES  # noqa: E402
from modules.face_analyser import get_many_faces  # noqa: E402

SOGLIA = 0.55          # distanza coseno per considerare la stessa persona
PASSO = 3              # campionamento frame
ROI = 384              # lato del sistema di riferimento: deve contenere
                       # capelli sopra la testa e sulle spalle


def main() -> int:
    video = Path(sys.argv[1])
    parser = FaceParser(ROOT / "models", ["CPUExecutionProvider"])
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    gruppi: list[dict] = []

    for i in range(0, n, PASSO):
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        volti = get_many_faces(img)
        if not volti:
            continue
        f = max(volti, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        labels, _ = parser.parse(img, f.bbox)
        capelli = parser.mask_for(labels, HAIR_CLASSES)
        if not capelli.any():
            continue

        kp = np.asarray(f.kps, np.float32)
        mento = float(kp[4][1])
        sotto = capelli[int(mento):, :]           # capelli sotto il mento
        area_volto = max(1.0, float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
        capelli_sotto = float((sotto > 0).sum()) / area_volto
        capelli_tot = float((capelli > 0).sum()) / area_volto

        # sistema di riferimento allineato: centro = mezzo interoculare,
        # scala = distanza interoculare, cosi' frame diversi sono confrontabili
        # La finestra deve contenere i capelli SOPRA la testa e, per le
        # persone con capelli lunghi, anche quelli sulle spalle: quindi e'
        # piu' alta di una faccia e centrata fra gli occhi e il mento, non
        # sugli occhi (centrata li' la finestra taglia via tutta la
        # capigliatura e il confronto risulta vuoto).
        fh = max(1.0, float(f.bbox[3] - f.bbox[1]))
        oc = float(np.linalg.norm(kp[0] - kp[1]))
        cx = float((kp[0][0] + kp[1][0]) / 2)
        cy = float((kp[0][1] + kp[1][1]) / 2) + 0.30 * fh
        sc = ROI / (2.6 * fh)
        M = np.float32([[sc, 0, ROI / 2 - cx * sc],
                        [0, sc, ROI / 2 - cy * sc]])
        capelli_r = cv2.warpAffine(capelli, M, (ROI, ROI), flags=cv2.INTER_NEAREST)

        emb = f.normed_embedding
        g = next((g for g in gruppi
                  if 1 - float(emb @ g["emb"]) < SOGLIA), None)
        if g is None:
            gruppi.append({"emb": emb, "frame": [], "sotto": [],
                           "tot": [], "capelli_r": [capelli_r > 0]})
            g = gruppi[-1]
        g["frame"].append(i)
        g["sotto"].append(capelli_sotto)
        g["tot"].append(capelli_tot)
        g["capelli_r"].append(capelli_r > 0)

    print("=" * 78)
    print("  CHI HA I CAPELLI LUNGHI")
    print("=" * 78)
    print("  gruppo  frame  capelli/volo  capelli sotto il mento/volo")
    risultati = []
    for n_, g in enumerate(gruppi, 1):
        sotto, tot = float(np.mean(g["sotto"])), float(np.mean(g["tot"]))
        risultati.append((n_, sotto, tot, len(g["frame"]), g))
        print(f"  persona {n_}   {len(g['frame']):4d}     {tot:6.2f}      {sotto:6.2f}")

    if not risultati:
        print("  nessun volto trovato")
        return 1

    # la piu' "capelluta sotto il mento" e' quella con i capelli lunghi
    n_lunghi, sotto_l, tot_l, cnt_l, g_l = max(risultati, key=lambda r: r[1])
    print("  " + "-" * 78)
    print(f"  -> persona {n_lunghi} ha i capelli lunghi "
          f"({sotto_l:.2f} di capigliatura sotto il mento)")

    # la maglietta coperta e' visibile in qualche altro frame?
    maschere = np.array(g_l["capelli_r"])
    if len(maschere) < 3:
        print(f"  -> solo {len(maschere)} frame: troppo pochi per la mediana temporale")
        return 0

    rivelato = np.zeros(maschere[0].shape, bool)
    for m in maschere:
        rivelato |= ~m
    coperto = maschere.any(axis=0)
    if not coperto.any():
        print("  -> nessuna area di capelli da valutare")
        return 0

    recuperabile = float((coperto & rivelato).sum()) / float(coperto.sum())
    print("  " + "-" * 78)
    print(f"  frame della persona: {len(maschere)}")
    print(f"  area di capelli mai scoperta in nessun frame : "
          f"{1 - recuperabile:.1%}")
    print(f"  area recuperabile dagli altri frame           : {recuperabile:.1%}")
    print("  " + "-" * 78)
    if recuperabile > 0.6:
        print("  -> La maglietta e' visibile altrove: la mediana temporale basta,")
        print("     NON serve un modello di inpainting.")
    else:
        print("  -> La maglietta non e' mai visibile: serve un inpainting")
        print("     (LaMa ONNX) per ricostruirla.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())