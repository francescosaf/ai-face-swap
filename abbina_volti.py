#!/usr/bin/env python3
"""Censimento dei volti di un video + tabella frame -> persona -> foto.

Serve a decidere la mappa `--map` di headless_faceswap.py. Produce:

    persona_N.png        griglia di volti campione della persona N
    match_riepilogo.md   tabella leggibile con numeri e somiglianze
    match_frame_foto.csv una riga per ogni frame: persona, bbox, foto
    match_config.json    config pronto da passare a headless_faceswap.py

Perche' la mappa va scritta a mano
-----------------------------------
L'abbinamento automatico foto <-> persona funziona solo se il soggetto della
foto e' davvero presente nel video: in quel caso la somiglianza arcface e'
alta (0.4-0.7) e si sa a chi appartiene. Quando le foto sono di persone
assenti dal video la somiglianza e' rumore (0.0-0.2) e la persona "piu'
simile" e' casuale: assegnare in automatico produce abbinamenti senza
senso. Per questo le somiglianze vengono solo riportate, e la colonna
`foto` del CSV resta vuota finche' non la compili con --assegna.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "dlfolder"))

from insightface.app import FaceAnalysis  # noqa: E402

SCHEMA = {  # Ruoli insightface: 0 occhio sx, 1 occhio dx, 2 naso, 3 bocca sx, 4 bocca dx
    "occhi": (0, 1),
}


# I nomi dei provider sono case-sensitive: con il solo nome del provider
# ('cpu') si finiva per chiedere 'cpuExecutionProvider', che insightface non
# riconosce e segnala come non disponibile, restando poi su CPU per caso.
NOMI_PROVIDER = {"cpu": "CPUExecutionProvider",
                 "cuda": "CUDAExecutionProvider",
                 "coreml": "CoreMLExecutionProvider"}


def carica_analyser(provider: str, det_size: int = 640) -> FaceAnalysis:
    nome = NOMI_PROVIDER.get(provider.lower(),
                             provider if provider.endswith("ExecutionProvider")
                             else f"{provider}ExecutionProvider")
    app = FaceAnalysis(name="buffalo_l", providers=[nome])
    app.prepare(ctx_id=0 if provider.lower() == "cuda" else -1,
                det_size=(det_size, det_size))
    app.max_face_crop = False
    return app


def _retry_soglia_bassa(app, img, soglia=0.25, minimo=0.25):
    """Seconda prova con soglia ridotta, come in face_analyser."""
    det = getattr(app, "det_model", None)
    if det is None:
        return []
    precedente = getattr(det, "det_thresh", None)
    try:
        det.det_thresh = soglia
        fs = app.get(img)
    finally:
        if precedente is not None:
            det.det_thresh = precedente
    return [f for f in fs if getattr(f, "det_score", 1.0) >= minimo]


def rileva(app: FaceAnalysis, video: Path, stride: int):
    """Rileva i volti su tutti i frame (o ogni `stride`)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"non riesco ad aprire {video}")
    righe = []
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % stride == 0:
            fs = app.get(img)
            if not fs:
                # stessa soglia di faceswap: altrimenti il passo di swap
                # trova un volto che questo passo non ha mappato, e la
                # persona resta originale per qualche decimo di secondo
                fs = _retry_soglia_bassa(app, img)
            for f in fs:
                x1, y1, x2, y2 = [int(v) for v in f.bbox]
                righe.append({
                    "frame": i, "img": img, "emb": f.normed_embedding,
                    "bbox": (x1, y1, x2, y2), "kps": np.asarray(f.kps),
                })
        i += 1
    cap.release()
    return righe, i


def raggruppa(emb: np.ndarray, soglia: float, n_forzato: int | None):
    """Raggruppa i volti per identita' (cosine agglomerativo)."""
    from sklearn.cluster import AgglomerativeClustering
    modello = AgglomerativeClustering(
        n_clusters=n_forzato, distance_threshold=None if n_forzato else soglia,
        metric="cosine", linkage="average")
    return modello.fit_predict(emb)


def griglia(ritagli: list[np.ndarray], cols: int = 4, lato: int = 128) -> np.ndarray:
    if not ritagli:
        return np.zeros((lato, lato, 3), np.uint8)
    celle = []
    for r in ritagli:
        h, w = r.shape[:2]
        s = lato / max(h, w)
        celle.append(cv2.resize(r, (max(1, int(w * s)), max(1, int(h * s)))))
    righe = []
    for i in range(0, len(celle), cols):
        gruppo = celle[i:i + cols]
        hmax = max(c.shape[0] for c in gruppo)
        riga = []
        for c in gruppo:
            tappo = np.zeros((hmax, c.shape[1], 3), np.uint8)
            tappo[:c.shape[0]] = c
            riga.append(tappo)
        while len(riga) < cols:
            riga.append(np.zeros((hmax, riga[0].shape[1], 3), np.uint8))
        righe.append(np.hstack(riga))
    wmax = max(r.shape[1] for r in righe)
    out = []
    for r in righe:
        if r.shape[1] < wmax:
            tappo = np.zeros((r.shape[0], wmax - r.shape[1], 3), np.uint8)
            r = np.hstack([r, tappo])
        out.append(r)
    return np.vstack(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--photo", required=True, type=Path, nargs="+")
    ap.add_argument("--out", required=True, type=Path,
                    help="cartella in cui scrivere i file")
    ap.add_argument("--stride", type=int, default=1,
                    help="analizza un frame ogni N (1 = tutti)")
    ap.add_argument("--soglia", type=float, default=0.45,
                    help="distanza coseni per unire due volti (0.45 = severo)")
    ap.add_argument("--identities", type=int, default=None,
                    help="forza il numero di persone")
    ap.add_argument("--det-size", type=int, default=640, choices=[160, 320, 640, 1024],
                    help="DEVE coincidere con il --det-size usato nello swap: "
                         "i bounding box del CSV sono prodotti a questa "
                         "dimensione e l'abbinamento frame per bbox non "
                         "combacia se le due corrono diverse.")
    ap.add_argument("--min-rilevamenti", type=int, default=30,
                    help="una persona conta come visibile solo con almeno "
                         "questo numero di volti rilevati")
    ap.add_argument("--min-px", type=int, default=32,
                    help="lato minimo del volto in pixel perche' un "
                         "rilevamento conti")
    ap.add_argument("--assegna", default="",
                    help="mappa persona->foto, es. 1,3,0 (indici 0-based delle "
                         "--photo). Se omesso la colonna foto resta vuota.")
    ap.add_argument("--provider", default="cpu",
                    choices=["cuda", "coreml", "cpu"])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    app = carica_analyser(args.provider, args.det_size)
    print(f"      detector a {args.det_size}px")

    print(f"[1/5] rilevo i volti su {args.video.name} (stride {args.stride})")
    righe, n_frame = rileva(app, args.video, args.stride)
    if not righe:
        print("nessun volto rilevato")
        return 1
    print(f"      {len(righe)} volti su {n_frame} frame")

    print("[2/5] carico i volti delle foto")
    foto_emb = []
    for p in args.photo:
        img = cv2.imread(str(p))
        if img is None:
            raise SystemExit(f"foto non leggibile: {p}")
        fs = app.get(img)
        if not fs:
            print(f"      ATTENZIONE: nessun volto in {p.name}")
            foto_emb.append(None)
            continue
        f = max(fs, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        foto_emb.append(f.normed_embedding)
        print(f"      {p.name}: {len(fs)} volto/i, uso il piu' grande")

    # Filtro di visibilita'. Il clustering non distingue una persona vera
    # dai suoi frammenti: una stessa persona capita spesso in 3-4 cluster,
    # perche' il detector la segmenta quando cambia posa o luce. Il risultato
    # e' un elenco di "persone" che non esistono e che a occhio non si vedono.
    # Qui si tiene solo chi e' visibile: almeno --min-rilevamenti volti, e
    # con il volto abbastanza grande da essere riconoscibile (--min-px).
    if args.min_px > 0:
        grandi = [k for k, r in enumerate(righe)
                  if min(r["bbox"][2] - r["bbox"][0],
                         r["bbox"][3] - r["bbox"][1]) >= args.min_px]
    else:
        grandi = list(range(len(righe)))
    if not grandi:
        grandi = list(range(len(righe)))
        print(f"      ATTENZIONE: nessun volto raggiunge {args.min_px}px, "
              f"il filtro sulla dimensione e' stato ignorato")

    print("[3/5] raggruppamento delle identita'")
    emb = np.array([r["emb"] for r in righe])
    lab = raggruppa(emb, args.soglia, args.identities)
    # le persone sono numerate in ordine di prima apparizione, come nel
    # numerazione di --map: "persona 1" e' la prima che si vede nel video
    gruppi: dict[int, list[int]] = {}
    for k, l in enumerate(lab):
        gruppi.setdefault(int(l), []).append(k)
    ordine = sorted(gruppi, key=lambda l: min(righe[k]["frame"] for k in gruppi[l]))
    persona_di = {k: n for n, l in enumerate(ordine, 1) for k in gruppi[l]}

    # quante volte e per quanti frame e' visibile ogni persona
    n_rilev = {n: 0 for n in range(1, len(ordine) + 1)}
    for k in grandi:
        n_rilev[persona_di[k]] += 1
    visibili = {n for n, c in n_rilev.items() if c >= args.min_rilevamenti}
    scartate = sorted(set(n_rilev) - visibili)

    sims = np.full((len(righe), len(args.photo)), np.nan)
    for k, r in enumerate(righe):
        for j, e in enumerate(foto_emb):
            if e is not None:
                sims[k, j] = float(np.dot(r["emb"], e))

    # -1 (o x/none/originale) = lascia il volto originale, senza scambio.
    # Serve per le persone che non vanno sostituite: senza questo valore
    # finivano comunque con una foto a caso.
    _originali = {"-1", "x", "none", "originale", "original", "no", "n"}
    assegna = []
    if args.assegna:
        for tok in args.assegna.split(","):
            t = tok.strip().lower()
            assegna.append(-1 if t in _originali else int(t))
    if assegna and any(not -1 <= a < len(args.photo) for a in assegna):
        print(f"[errore] --assegna fuori intervallo: foto disponibili 0.."
              f"{len(args.photo)-1}, oppure -1 per lasciare l'originale")
        return 1

    print("[4/5] scrivo CSV e config")
    nomi = [p.name for p in args.photo]
    with open(args.out / "match_frame_foto.csv", "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["frame", "persona", "x1", "y1", "x2", "y2"]
                      + [f"sim_{n}" for n in nomi]
                      + ["foto_assegnata", "foto_indice"])
        for k, r in enumerate(righe):
            pn = persona_di[k]
            # persona non abbastanza visibile: resta originale. Senza questo
            # controllo un frammento di 3 frame finiva con una foto a caso.
            if pn not in visibili:
                idx = -1
            elif assegna:
                idx = assegna[pn - 1] if pn <= len(assegna) else None
            else:
                idx = None
            wcsv.writerow([r["frame"], pn, *r["bbox"]]
                          + [f"{sims[k, j]:.4f}" if not np.isnan(sims[k, j]) else ""
                             for j in range(len(nomi))]
                          + [("originale" if idx == -1
                              else nomi[idx]) if idx is not None else "",
                             idx if idx is not None else ""])

    # config pronto: --map con un indice foto per ogni persona
    mappa = [assegna[n - 1] if n <= len(assegna) else 0 for n in range(1, len(ordine) + 1)]
    (args.out / "match_config.json").write_text(json.dumps({
        "video": str(args.video), "photo": [str(p) for p in args.photo],
        "persone": len(ordine), "mappa_foto": mappa,
        "nota": "mappa_foto usa indici 0-based nelle --photo; ricalcolala con "
                "--assegna se cambi le abbinamenti",
    }, indent=2, ensure_ascii=False))

    print("[5/5] griglie dei volti e riepilogo")
    righe_md = ["| persona | visibile | rilev. validi | rilevamenti | "
                "primo | ultimo | oculare | "
                + " | ".join(f"sim {n}" for n in nomi) + " | foto | thumbnail |",
                "|---|---|---|---|---|---|---|" + "---|" * (len(nomi) + 2)]
    for n, l in enumerate(ordine, 1):
        ks = gruppi[l]
        frames = [righe[k]["frame"] for k in ks]
        oc = [float(np.linalg.norm(righe[k]["kps"][SCHEMA["occhi"][0]]
                                   - righe[k]["kps"][SCHEMA["occhi"][1]])) for k in ks]
        campioni = [ks[j] for j in np.linspace(0, len(ks) - 1, 4).astype(int)]
        ritagli = []
        for k in campioni:
            x1, y1, x2, y2 = righe[k]["bbox"]
            ritagli.append(righe[k]["img"][max(0, y1):y2, max(0, x1):x2].copy())
        nome_png = f"persona_{n}.png"
        cv2.imwrite(str(args.out / nome_png), griglia(ritagli))
        med = [sims[ks, j].mean() if not np.isnan(sims[ks, j]).all() else float("nan")
               for j in range(len(nomi))]
        if n not in visibili:
            foto = "scartata (sotto soglia)"
        else:
            _a = assegna[n - 1] if n <= len(assegna) else None
            foto = ("originale" if _a == -1 else nomi[_a]) if _a is not None \
                else "**da decidere**"
        righe_md.append(
            f"| {n} | {'**si**' if n in visibili else 'no'} | {n_rilev[n]} | "
            f"{len(ks)} | {min(frames)} | {max(frames)} | "
            f"{np.mean(oc):.0f}px | "
            + " | ".join("n/d" if np.isnan(m) else f"{m:+.3f}" for m in med)
            + f" | {foto} | `{nome_png}` |")

    massimi = [(n, l) for n, l in enumerate(ordine, 1)]
    suggerito = []
    for n, l in massimi:
        ks = gruppi[l]
        m = [(np.nanmean(sims[ks, j]) if not np.isnan(sims[ks, j]).all() else -9, j)
             for j in range(len(nomi))]
        suggerito.append(max(m)[1])
    (args.out / "match_riepilogo.md").write_text(
        "# Match video -> foto\n\n"
        f"Video: `{args.video.name}`  ({n_frame} frame, {len(righe)} volti rilevati)\n"
        f"Cluster rilevati: **{len(ordine)}**  \n"
        f"Persone visibili (>= {args.min_rilevamenti} volti validi): "
        f"**{len(visibili)}**  \n"
        f"Cluster scartati: **{len(scartate)}** "
        f"({', '.join(str(n) for n in scartate) if scartate else 'nessuno'})\n\n"
        f"I cluster scartati restano con il volto originale. Per una persona "
        f" vera spezzata in piu' cluster, alza `--min-rilevamenti` solo se il "
        f" filtro scarta anche pezzi visibili.\n\n"
        "## Persone nel video\n\n" + "\n".join(righe_md) + "\n\n"
        "## Cosa fare\n\n"
        "1. Apri le immagini `persona_N.png` e riconosci chi e' chi.\n"
        "2. Scrivi la mappa nel formato `--assegna`, indici 0-based delle foto "
        "nell'ordine in cui le hai passate con `--photo`.\n"
        "3. Rilancia con `--assegna 1,2,0` e usa il `mappa_foto` che esce "
        "in `match_config.json` come `--map`.\n\n"
        "## Avvertenza sulle somiglianze\n\n"
        "Se i soggetti delle foto non sono presenti nel video tutte le "
        "somiglianze sono rumore (0.0-0.2) e la colonna `foto` va decisa a "
        "vista, non dai numeri. La riga \"suggerimento automatico\" serve "
        "solo come controllo:\n\n"
        + f"- suggerimento automatico: {suggerito} "
        + "(attenditi: " + ", ".join(str(s) for s in suggerito) + ")\n",
        encoding="utf-8")

    print(f"\nCluster rilevati: {len(ordine)}")
    print(f"{'persona':>8} {'visib.':>7} {'validi':>7} {'rilev.':>7} "
          f"{'primo':>7} {'ultimo':>7}  "
          + "".join(f"{n.split('.')[0]:>9}" for n in nomi) + "   foto")
    for n, l in enumerate(ordine, 1):
        ks = gruppi[l]
        frames = [righe[k]["frame"] for k in ks]
        med = [sims[ks, j].mean() if not np.isnan(sims[ks, j]).all() else float("nan")
               for j in range(len(nomi))]
        if n not in visibili:
            foto = "SCARTATA (sotto soglia)"
        else:
            _a = assegna[n - 1] if n <= len(assegna) else None
            foto = ("originale" if _a == -1 else nomi[_a]) if _a is not None \
                else "da decidere"
        print(f"{n:>8} {'si' if n in visibili else 'no':>7} {n_rilev[n]:>7} "
              f"{len(ks):>7} {min(frames):>7} {max(frames):>7}  "
              + "".join(f"{m:>9.3f}" if not np.isnan(m) else f"{'n/d':>9}" for m in med)
              + f"   {foto}")
    print(f"\n  visibili: {len(visibili)} su {len(ordine)} cluster "
          f"(soglia {args.min_rilevamenti} volti validi, volto >= {args.min_px}px)")
    if scartate:
        print(f"  scartate: {scartate} -> restano col volto originale")
    print(f"\nfile scritti in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())