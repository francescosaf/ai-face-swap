"""Diagnosi dei difetti tipici del face/head swap su un video generato.

Non serve guardare i frame a occhio: i tre difetti che si vedono in un
output sbagliato hanno una firma numerica.

  face melting       i dettagli del volto vengono persi: la varianza del
                     Laplaciano nella regione del volto crolla rispetto
                     all'originale, perche' il volto e' stato ammorbidito
                     o stirato
  warped face        le proporzioni si deformano: distanza inter-oculare e
                     aspect ratio del bounding box divergono dall'originale
  alignment failure  la regione incollata non coincide col volto: IoU dei
                     box e scarto del centroide peggiorano
  bordo visibile     discontinuita' netta lungo il contorno della maschera
  flickering         l'identita' o il contenuto oscillano fra frame

Uso:
    python diagnosi_difetti.py --originale VIDEO --output VIDEO_ALLOCCATO
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

from face_parsing import inner_face_mask  # noqa: E402
from modules.face_analyser import get_many_faces  # noqa: E402

# soglie: oltre questi valori il difetto e' considerato presente
MIN_INTEROCULARE = 25.0   # sotto questa soglia i landmark non sono affidabili

SOGLIE = {
    "melting": 0.55,       # rapporto dettagli output/originale
    "warping_oculare": 0.08,  # scarto relativo della distanza inter-oculare
    "warping_aspect": 0.06,   # scasso relativo dell'aspect ratio del box
    "allineamento_iou": 0.80,  # IoU minima fra i box
    "allineamento_px": 6.0,    # spostamento massimo del baricentro dei landmark
    "bordo": 6.0,         # eccesso di gradiente sul contorno vs originale
    "flicker": 0.08,      # deviazione std dell'identita' fra frame
}


def dettagli(img: np.ndarray, mask: np.ndarray) -> float:
    """Varianza del Laplaciano dentro la maschera: misura la grana."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    m = mask > 0
    if m.sum() < 50:
        return 0.0
    return float(lap[m].var())


def piu_vicino(faces, box):
    """Il volto piu' vicino al box originale: altrimenti si misura un'altra
    persona e il confronto diventa privo di significato."""
    if not faces:
        return None
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return min(faces, key=lambda f: float(np.hypot(
        (f.bbox[0] + f.bbox[2]) / 2 - cx, (f.bbox[1] + f.bbox[3]) / 2 - cy)))


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / float((a[2] - a[0]) * (a[3] - a[1])
                         + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def diagnostica(percorso_orig: Path, percorso_out: Path, campioni: int = 20):
    vo = cv2.VideoCapture(str(percorso_orig))
    n_orig = int(vo.get(cv2.CAP_PROP_FRAME_COUNT))
    vo.release()
    vo = cv2.VideoCapture(str(percorso_out))
    n_out = int(vo.get(cv2.CAP_PROP_FRAME_COUNT))
    vo.release()
    if not n_out:
        raise SystemExit("[errore] output non leggibile")

    indici = np.linspace(0, min(n_orig, n_out) - 1, campioni).astype(int)

    def frame_at(percorso, i):
        cap = cv2.VideoCapture(str(percorso))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        cap.release()
        return f if ok else None

    dettagli_r, dist_oc, dist_centro, iou_b, grad_bordo = [], [], [], [], []
    spostamenti, aspect, identita = [], [], []

    for i in indici:
        fo, fs = frame_at(percorso_orig, i), frame_at(percorso_out, i)
        if fo is None or fs is None:
            continue
        volti_o = get_many_faces(fo)
        if not volti_o:
            continue
        f_o = max(volti_o, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        f_s = piu_vicino(get_many_faces(fs), f_o.bbox)
        if f_s is None:
            continue

        kpo, kps = np.asarray(f_o.kps, np.float32), np.asarray(f_s.kps, np.float32)
        m_o = inner_face_mask(fo.shape, kpo, 1.1)
        m_s = inner_face_mask(fs.shape, kps, 1.1)

        # Su un volto di 11 px la regressione dei landmark ha un errore
        # enorme: misurare li' dà scarti del 100-160% che non sono difetti
        # ma rumore. La geometria si valuta solo dove il volto è grande
        # abbastanza da essere misurabile.
        oc_o = float(np.linalg.norm(kpo[0] - kpo[1]))
        if oc_o < MIN_INTEROCULARE:
            continue

        d_o, d_s = dettagli(fo, m_o), dettagli(fs, m_s)
        if d_o > 0:
            dettagli_r.append(d_s / d_o)

        oc_s = float(np.linalg.norm(kps[0] - kps[1]))
        if oc_o > 1:
            dist_oc.append(abs(oc_s / oc_o - 1.0))

        # Proporzione del volto dai soli landmark: larghezza della bocca su
        # distanza inter-oculare. L'aspect ratio del bounding box non va bene
        # per un head swap, perche' il detector dimensiona il box anche sui
        # capelli e il suo valore cambia anche quando il volto e' identico.
        def proporzione(k):
            oc = float(np.linalg.norm(k[0] - k[1]))
            bo = float(np.linalg.norm(k[3] - k[4]))
            return bo / oc if oc > 1 else float("nan")
        po, ps = proporzione(kpo), proporzione(kps)
        if po == po and ps == ps:      # NaN-safe
            aspect.append(abs(ps / po - 1.0))

        iou_b.append(iou(f_o.bbox, f_s.bbox))
        # Lo spostamento si misura sul baricentro dei landmark (occhi, naso,
        # bocca), non sul centro del bounding box: il detector dimensiona il
        # box su tutta la testa, capelli compresi, quindi in un head swap il
        # box si sposta anche quando il volto e' perfettamente allineato e
        # misurarlo darebbe un falso allineamento rotto.
        spostamenti.append(float(np.linalg.norm(kps[:4].mean(axis=0)
                                                - kpo[:4].mean(axis=0))))

        # Eccesso di discontinuita' lungo il bordo della maschera. Il
        # contorno del visto ha gia' gradiente naturale (zigomi, mascella):
        # confrontare con una soglia assoluta inseguirebbe un fantasma, quindi
        # si misura quanto il bordo e' piu' tagliente rispetto all'originale
        # nello stesso punto. Lo stesso contorno (landmark originali) viene
        # usato sui due frame, cosi' il confronto e' posizionato.
        bordo = cv2.morphologyEx(m_o, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        if bordo.any():
            def pendenza(img):
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                                cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
            e = bordo > 0
            g_o = float(pendenza(fo)[e].mean())
            g_s = float(pendenza(fs)[e].mean())
            grad_bordo.append(max(0.0, g_s - max(g_o, 1.0)))

        identita.append(1.0 - float(f_s.normed_embedding @ f_o.normed_embedding))

    def media(x):
        return float(np.mean(x)) if x else float("nan")

    return {
        "dettagli": media(dettagli_r),
        "oculare": media(dist_oc),
        "aspect": media(aspect),
        "iou": media(iou_b),
        "spostamento": media(spostamenti),
        "bordo": media(grad_bordo),
        "flicker": float(np.std(identita)) if len(identita) > 1 else float("nan"),
        "campioni": len(identita),
    }


def esito(m: dict) -> list[str]:
    out = []
    if m["dettagli"] < SOGLIE["melting"]:
        out.append(f"FACE MELTING: i dettagli del volto sono al {m['dettagli']:.0%} "
                   f"dell'originale (soglia {SOGLIE['melting']:.0%})")
    if m["oculare"] > SOGLIE["warping_oculare"]:
        out.append(f"WARPED FACE: distanza inter-oculare cambiata del "
                   f"{m['oculare']:.1%} (soglia {SOGLIE['warping_oculare']:.0%})")
    if m["aspect"] > SOGLIE["warping_aspect"]:
        out.append(f"WARPED FACE: proporzioni del volto cambiate del "
                   f"{m['aspect']:.1%} (soglia {SOGLIE['warping_aspect']:.0%})")
    if m["iou"] < SOGLIE["allineamento_iou"]:
        out.append(f"ALIGNMENT FAILURE: i box originale e output si sovrappongono "
                   f"solo al {m['iou']:.0%} (soglia {SOGLIE['allineamento_iou']:.0%})")
    if m["spostamento"] > SOGLIE["allineamento_px"]:
        out.append(f"ALIGNMENT FAILURE: il volto si sposta di "
                   f"{m['spostamento']:.1f} px (soglia {SOGLIE['allineamento_px']:.0f})")
    if m["bordo"] > SOGLIE["bordo"]:
        out.append(f"BORDO VISIBILE: il contorno e' {m['bordo']:.1f} piu' "
                   f"tagliente dell'originale (soglia {SOGLIE['bordo']:.0f})")
    if m["flicker"] > SOGLIE["flicker"]:
        out.append(f"FLICKER: l'identita' oscilla di {m['flicker']:.3f} fra frame")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--originale", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--campioni", type=int, default=20)
    a = ap.parse_args()

    m = diagnostica(a.originale, a.output, a.campioni)
    print("=" * 68)
    print(f"  DIAGNOSI DIFETTI   {a.output.name}")
    print("=" * 68)
    print(f"  frame misurabili (viso >= {MIN_INTEROCULARE:.0f}px) : {m['campioni']}")
    print(f"  dettagli volto (orig=100%)  : {m['dettagli']:.0%}")
    print(f"  distanza inter-oculare      : {m['oculare']:.2%} di scarto")
    print(f"  proporzioni bocca/occhi     : {m['aspect']:.2%} di scarto")
    print(f"  sovrapposizione box         : {m['iou']:.1%}")
    print(f"  spostamento del volto       : {m['spostamento']:.2f} px")
    print(f"  bordo piu' tagliente        : +{m['bordo']:.1f}")
    print(f"  oscillazione identita'      : {m['flicker']:.3f}")
    print("  " + "-" * 66)
    problemi = esito(m)
    if problemi:
        for p in problemi:
            print(f"  ! {p}")
    else:
        print("  nessun difetto rilevato")
    print("=" * 68)
    return 1 if problemi else 0


if __name__ == "__main__":
    sys.exit(main())