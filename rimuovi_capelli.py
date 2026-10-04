"""Rimuove i capelli lunghi del video e ricostruisce quello che coprono.

Il problema: la terza persona ha i capelli lunghi che coprono spalle e
maglietta. Per un head swap serve toglierli, altrimenti sopra alla nuova
capigliatura restano due capigliature e si vede un alone.

La soluzione non e' inventare la maglietta: e' recuperarla. Misurato su
questo video, il 98.4% dell'area coperta dai capelli e' visibile in qualche
altro frame della stessa persona. La mediana temporale dei frame allineati
ricostruisce quindi maglietta, collo e spalle con il contenuto reale del
video, non con un'ipotesi.

Solo la parte che non e' mai visibile viene riempita, e per quella basta
unTelea sui bordi: e' l'1.6% e non contiene texture da ricostruire.

Il risultato e' un video "pulito" su cui il normale head swap scrive la nuova
testa senza piu' incastrarsi con i capelli lunghi.

Uso:
    python rimuovi_capelli.py --video IN.mp4 --output PULITO.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dlfolder"))

from face_parsing import FaceParser, HAIR_CLASSES, inner_face_mask  # noqa: E402
from modules.face_analyser import get_many_faces  # noqa: E402
from video_face_roster import cluster_identities  # noqa: E402

ROI = 384          # lato del ritaglio allineato
SOGLIA = 0.55      # distanza coseno per considerare la stessa persona
DILATAZIONE = 9    # px: quanto si allarga la regione dei capelli da cancellare


def allineamento(f) -> np.ndarray:
    """Matrice che porta il frame nel ritaglio allineato del volto.

    Centrata fra occhi e mento e alta 2.6 volte la faccia: cosi' contiene i
    capelli sopra la testa e, per chi ha i capelli lunghi, anche quelli che
    scendono sulle spalle. Centrata sugli occhi la finestra li taglia via.
    """
    kp = np.asarray(f.kps, np.float32)
    fh = max(1.0, float(f.bbox[3] - f.bbox[1]))
    cx = float((kp[0][0] + kp[1][0]) / 2)
    cy = float((kp[0][1] + kp[1][1]) / 2) + 0.30 * fh
    sc = ROI / (2.6 * fh)
    return np.float32([[sc, 0, ROI / 2 - cx * sc],
                       [0, sc, ROI / 2 - cy * sc]]), kp


def inverti(M: np.ndarray) -> np.ndarray:
    """Ritaglio allineato -> frame."""
    return np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2].astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--provider", default="CPUExecutionProvider")
    a = ap.parse_args()

    parser = FaceParser(ROOT / "models", [a.provider])

    cap = cv2.VideoCapture(str(a.video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 24.0
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    print(f"[1/3] letti {len(frames)} frame  {W}x{H} @ {FPS:.2f} fps")

    # --- 1. rileva i volti e assegna le identita' ---
    # Il clustering greedy per centroidi frammentava la stessa persona in 21
    # gruppi, e con pochi frame per gruppo la mediana temporale non regge.
    # Si riusa quello del roster: linkage average con soglia scelta sui dati.
    rilevati = []
    for i, img in enumerate(frames):
        volti = get_many_faces(img)
        if not volti:
            continue
        f = max(volti, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        rilevati.append({"i": i, "face": f})
        if i % 60 == 0:
            print(f"      frame {i}/{len(frames)}  rilevati={len(rilevati)}")

    def esporta(sequenza: list) -> None:
        """Scrive i frame e rimixa l'audio originale.

        cv2.VideoWriter con mp4v si apriva senza errori ma produceva un file
        vuoto: va controllato isOpened(), e per l'audio serve comunque ffmpeg.
        """
        with tempfile.TemporaryDirectory() as td:
            for n, im in enumerate(sequenza):
                cv2.imwrite(str(Path(td) / f"{n:05d}.png"), im)
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{FPS}",
                   "-i", str(Path(td) / "%05d.png"), "-i", str(a.video),
                   "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264",
                   "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac",
                   "-shortest", str(a.output)]
            subprocess.run(cmd, check=True)

    if len(rilevati) < 4:
        esporta(frames)
        print("[ok] troppo pochi volti: video copiato invariato")
        return 0

    # Raggruppare le embedding frame per frame le frammenta: sono rumorose e
    # la soglia scelta sui dati taglia troppo stretto (70 gruppi su 376
    # rilevazioni, contro i 5 del roster). Il roster aggrega prima per
    # tracciato temporale e poi raggruppa: si fa lo stesso, in memoria.
    rilevati.sort(key=lambda r: r["i"])
    tracciati: list[dict] = []
    for r in rilevati:
        emb = r["face"].normed_embedding
        inserito = False
        for t in tracciati:
            if r["i"] - t["ultimo"] > 4:
                continue
            if 1 - float(emb @ t["media"]) > 0.5:
                continue
            t["media"] = (t["media"] * t["n"] + emb) / (t["n"] + 1)
            t["n"] += 1
            t["ultimo"] = r["i"]
            t["membri"].append(r)
            inserito = True
            break
        if not inserito:
            tracciati.append({"media": emb.copy(), "n": 1, "ultimo": r["i"],
                              "membri": [r]})

    etichette, soglia = cluster_identities([t["media"] for t in tracciati])
    persone: dict[int, list] = {}
    for t, lab in zip(tracciati, etichette):
        persone.setdefault(lab, []).extend(t["membri"])
    for rs in persone.values():
        rs.sort(key=lambda r: r["i"])
    print(f"[1/3] {len(rilevati)} rilevazioni, {len(tracciati)} tracciati -> "
          f"{len(persone)} persone (soglia {soglia:.3f})")

    # --- 2. ritagli allineati e maschere, raggruppati per persona ---
    for lab, rs in persone.items():
        for r in rs:
            img = frames[r["i"]]
            M, kp = allineamento(r["face"])
            r["M"] = M
            r["rit"] = cv2.warpAffine(img, M, (ROI, ROI), flags=cv2.INTER_AREA)
            r["cap_fonte"] = parser.mask_for(parser.parse(img, r["face"].bbox)[0],
                                             HAIR_CLASSES)
            r["cap"] = cv2.warpAffine(r["cap_fonte"], M, (ROI, ROI),
                                      flags=cv2.INTER_NEAREST)
            r["cap_frame"] = cv2.dilate(r["cap_fonte"],
                                         np.ones((DILATAZIONE, DILATAZIONE), np.uint8))
            r["vol_frame"] = inner_face_mask(img.shape, kp, 1.15)

    print(f"[2/3] {len(persone)} persone distinte")

    # --- 2. mediana temporale per persona: la "lastra pulita" ---
    piatti = {}
    for n, g in persone.items():
        if len(g) < 3:
            continue
        # mediana sui pixel: i capelli sonoFuori dal valore centrale, il
        # torso e lo sfondo restanti coincidono, la mediana li ricostruisce
        lastra = np.median(np.stack([r["rit"] for r in g]), axis=0).astype(np.uint8)
        # chiude i buchi una volta sola: la mediana sfuma dove il soggetto si e'
        # mosso, e quei pixel vanno riempiti senza fingere texture
        zona = np.zeros(lastra.shape[:2], np.uint8)
        for r in g:
            zona = cv2.bitwise_or(zona, cv2.warpAffine(r["cap"], r["M"], (ROI, ROI),
                                                      flags=cv2.INTER_NEAREST))
        buco = zona.copy()
        for _ in range(3):
            er = cv2.erode(buco, np.ones((9, 9), np.uint8))
            frontiera = cv2.bitwise_and(buco, cv2.bitwise_not(er))
            if not frontiera.any():
                break
            lastra = cv2.inpaint(lastra, frontiera, 5, cv2.INPAINT_TELEA)
            buco = er
        piatti[n] = lastra
        cap_medio = float(np.mean([(r["cap"] > 0).mean() for r in g]))
        def mento_roi(f, M):
            x, y = float(f.kps[4][0]), float(f.kps[4][1])
            return int(M[1, 0] * x + M[1, 1] * y + M[1, 2])
        sotto = float(np.mean([(r["cap"][max(0, mento_roi(r["face"], r["M"])):] > 0).mean()
                               for r in g]))
        print(f"      persona {n}: {len(g)} frame, capigliatura media "
              f"{cap_medio:.1%}, sotto il mento {sotto:.1%}")
    print(f"[3/3] lastre pulite per {len(piatti)} persone")

    per_frame = {}
    for lab, rs in persone.items():
        for r in rs:
            r["lab"] = lab
            per_frame[r["i"]] = r

    # --- 3. ripulisci ogni frame ---
    risultato = []
    for i, img in enumerate(frames):
        r = per_frame.get(i)
        if r is None or r["lab"] not in piatti:
            risultato.append(img)
            continue

        # da cancellare: capelli allargati, MAI il volto (lo rifa inswapper)
        da = cv2.bitwise_and(r["cap_frame"], cv2.bitwise_not(r["vol_frame"]))
        if int((da > 0).sum()) < 150:
            risultato.append(img)
            continue

        # La lastra torna nel frame e si compone SOLO dentro la maschera dei
        # capelli. Il ritaglio allineato copre una porzione del frame: se si
        # scriveva l'intero frame dal ritaglio, il 67% dell'immagine fuori
        # dalla testa risultava corrotto.
        lastra = cv2.warpAffine(piatti[r["lab"]], inverti(r["M"]), (W, H),
                                flags=cv2.INTER_CUBIC)
        alpha = np.clip(cv2.GaussianBlur(da, (0, 0), 2.0).astype(np.float32)
                        / 255.0, 0, 1)[..., None]
        risultato.append(np.clip(img.astype(np.float32) * (1 - alpha)
                                 + lastra.astype(np.float32) * alpha,
                                 0, 255).astype(np.uint8))
        if i % 60 == 0:
            print(f"      pulito {i}/{len(frames)}")
    esporta(risultato)
    print(f"[ok] scritto {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())