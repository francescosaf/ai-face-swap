#!/usr/bin/env python3
"""Analizza un video eProduce un censimento delle persone distinte che vi compaiono.

Serve per due motivi concreti:

1. Dirti quante persone ci sono, cosi' sai quante foto procurarti.
2. Produrre un file di configurazione gia' scritto, dove ogni riga dice
   "persona N -> mettici la foto X", con il prompt della modalita' scelta.

L'assegnazione automatica foto/persona NON e' affidabile quando i soggetti delle
foto non sono gia' nel video: in quel caso la similarita' degli embedding non
porta informazione. Per questo lo script salva anche un provino (contact sheet)
per ogni persona: serve a riconoscerla a occhio e riempire il campo a mano.

Uso:
    python video_face_roster.py video.mp4 -o TestFaceSwap/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DL_DIR = HERE / "dlfolder"

PROMPTS = {
    "headswap": (
        "Photorealistic full head replacement from the neck up. Seamlessly swap the "
        "face, entire hairstyle, hair volume, hairline, and ears from the reference "
        "image onto the target subject. Replace everything above the collarbone line. "
        "Perfectly match the original neck skin tone, lighting, and shadow transitions. "
        "8k resolution, ultra-detailed hair strands, highly realistic skin texture."
    ),
    "faceswap": (
        "Photorealistic face replacement only. Swap the eyes, nose, mouth, and inner "
        "facial structure from the reference photo onto the target subject. Strictly "
        "preserve the original subject's hair, hairstyle, head shape, and background. "
        "Seamless facial blend, perfect skin color matching, 8k resolution."
    ),
}


def setup_modules(provider: str | None):
    sys.path.insert(0, str(DL_DIR))
    import onnxruntime as ort
    import modules.globals as g
    avail = ort.get_available_providers()
    chosen = provider or next(
        (p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider",
                     "CPUExecutionProvider") if p in avail),
        "CPUExecutionProvider")
    g.execution_providers = [chosen, "CPUExecutionProvider"]
    g.det_size = 640
    from modules.face_analyser import get_many_faces
    return get_many_faces, chosen


def probe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
         "-show_entries", "format=duration", "-of", "json", str(video)],
        capture_output=True, text=True, check=True)
    d = json.loads(out.stdout)
    st = d["streams"][0]
    num, den = st.get("r_frame_rate", "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return {"width": int(st["width"]), "height": int(st["height"]), "fps": fps,
            "frames": int(st.get("nb_frames") or 0),
            "duration": float(d.get("format", {}).get("duration") or 0)}


def extract_samples(video: Path, n: int) -> list[Path]:
    tmp = Path(tempfile.mkdtemp(prefix="roster_"))
    step = max(1, n)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(video),
                    "-vf", f"select='not(mod(n\\,{step}))'",
                    "-fps_mode", "passthrough", "-qscale:v", "0",
                    str(tmp / "%05d.png")], check=True)
    return sorted(tmp.glob("*.png"))


def pick_cut(height: np.ndarray) -> float:
    """Sceglie dove tagliare l'albero gerarchico con il metodo di Otsu.

    Le distanze fra volti della stessa persona sono basse, quelle fra persone
    diverse alte: la soglia va nel 'valle' fra i due gruppi. Otsu massimizza
    la varianza fra le due classi, quindi trova quella valle dai dati invece
    di una costante arbitraria.
    """
    if height.size < 2:
        return float(height[0]) if height.size else 0.5
    h = np.sort(height)
    n = len(h)
    idx = np.arange(1, n)
    w0 = idx / n
    w1 = 1 - w0
    csum = np.cumsum(h)
    total = csum[-1]
    mu0 = csum[:-1] / idx
    mu1 = (total - csum[:-1]) / (n - idx)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(h[int(np.argmax(between))])


def cluster_identities(embeddings: list, force_k: int = 0) -> tuple[list[int], float]:
    """Raggruppa gli embedding in identita'. Ritorna (etichette, soglia usata)."""
    if not embeddings:
        return [], 0.0
    S = np.vstack(embeddings)
    if len(S) == 1:
        return [0], 0.0
    if force_k > 1:
        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters=min(force_k, len(S)),
                                        linkage="average", metric="cosine")
        return model.fit_predict(S).tolist(), float("nan")
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    Z = linkage(pdist(S, metric="cosine"), method="average")
    cut = pick_cut(Z[:, 2])
    return (fcluster(Z, t=cut, criterion="distance") - 1).tolist(), cut


def build_tracks(samples: list[Path], get_many_faces) -> list[dict]:
    """Raggruppa i volti in tracciati temporali e media le embedding.

    L'embedding di uno stesso volto cambia molto fra frame consecutivi (posa,
    espressione, motion blur, compressione): da 0 a ~0.56 sul video di prova.
    Senza questa media il clustering frammenta la stessa persona in 26
    'identita''. Mediando prima su ogni tracciato il segnale diventa stabile.
    """
    tracks: list[dict] = []
    prev = []
    for si, sp in enumerate(samples):
        frame = cv2.imread(str(sp))
        if frame is None:
            continue
        faces = get_many_faces(frame) or []
        current = []
        for f in faces:
            if f.normed_embedding is None:
                continue
            box = np.asarray(f.bbox[:4], np.float32)
            best, best_iou = None, 0.0
            for t in tracks:
                if t["last"] != si - 1:
                    continue
                tb = t["box"]
                ix1, iy1 = max(box[0], tb[0]), max(box[1], tb[1])
                ix2, iy2 = min(box[2], tb[2]), min(box[3], tb[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = float(np.prod(box[2:] - box[:2]) + np.prod(tb[2:] - tb[:2]) - inter)
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best, best_iou = t, iou
            if best is not None and best_iou > 0.25:
                best["embs"].append(f.normed_embedding)
                best["members"].append({"frame": sp, "bbox": f.bbox, "face": f})
                best["last"] = si
                best["box"] = box
                current.append(best)
            else:
                tracks.append({"embs": [f.normed_embedding], "last": si, "box": box,
                               "members": [{"frame": sp, "bbox": f.bbox, "face": f}]})
                current.append(tracks[-1])
        prev = current
    out = []
    for t in tracks:
        out.append({
            "embedding": np.mean(np.vstack(t["embs"]), axis=0),
            "members": sorted(t["members"], key=lambda m: str(m["frame"])),
            "count": len(t["members"]),
        })
    return [t for t in out if t["count"] >= 2]


def build_roster(samples: list[Path], get_many_faces, force_k: int = 0) -> tuple[list[dict], float]:
    """Raggruppa i volti per identita' e annota dove e quando compaiono."""
    tracks = build_tracks(samples, get_many_faces)
    if not tracks:
        return [], 0.0
    labels, cut = cluster_identities([t["embedding"] for t in tracks], force_k)
    groups: dict[int, dict] = {}
    for lbl, t in zip(labels, tracks):
        g = groups.setdefault(int(lbl), {"id": 0, "samples": [], "count": 0})
        g["samples"].extend(t["members"])
        g["count"] += t["count"]
    out = []
    for g in sorted(groups.values(), key=lambda x: -x["count"]):
        g["samples"].sort(key=lambda m: str(m["frame"]))
        g["id"] = len(out)
        out.append(g)
    return out, cut


def save_provini(roster: list[dict], out_dir: Path, target_fps: float) -> list[list[str]]:
    """Un contact sheet per persona: serve a capire chi e' chi."""
    paths = []
    for p in roster:
        picks = p["samples"]
        if len(picks) > 6:
            step = len(picks) / 6
            picks = [picks[int(i * step)] for i in range(6)]
        tiles = []
        for m in picks:
            img = cv2.imread(str(m["frame"]))
            if img is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in m["bbox"]]
            pad = int((x2 - x1) * 0.5)
            H, W = img.shape[:2]
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
            crop = img[y1:y2, x1:x2]
            if crop.size:
                crop = cv2.resize(crop, (160, 200))
                cv2.rectangle(crop, (0, 0), (159, 199), (0, 255, 0), 2)
                tiles.append(crop)
        if not tiles:
            paths.append([])
            continue
        while len(tiles) < 6:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.hstack(tiles[:6])
        n = int(p["samples"][0]["frame"].stem)
        dest = out_dir / f"persona_{p['id']+1}_provino.png"
        cv2.imwrite(str(dest), sheet)
        paths.append([str(dest)])
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Censimento delle persone distinte in un video + config pronta.")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path.home() / "Downloads" / "TestFaceSwap")
    ap.add_argument("--mode", choices=["headswap", "faceswap"], default="headswap")
    ap.add_argument("--samples", type=int, default=120, help="frame campionati")
    ap.add_argument("--identities", type=int, default=0, help="forza il numero di persone")
    ap.add_argument("--provider", choices=["cuda", "coreml", "cpu"])
    ap.add_argument("--prompt", help="prompt di riferimento da registrare nel config; "
                                     "i modelli ONNX non lo usano come ingresso")
    ap.add_argument("--photo", nargs="+", type=Path,
                    help="N foto di riferimento da registrare nel config")
    args = ap.parse_args()

    if not args.video.exists():
        print(f"video non trovato: {args.video}")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    get_many_faces, provider = setup_modules(args.provider)
    info = probe(args.video)
    print(f"Video      : {info['width']}x{info['height']} @ {info['fps']:.2f} fps, "
          f"{info['duration']:.2f}s, {info['frames']} frame")
    print(f"Provider   : {provider}")
    print(f"Modalita'  : {args.mode}")

    samples = extract_samples(args.video, max(1, info["frames"] // args.samples))
    print(f"Campionati : {len(samples)} frame")

    roster, cut = build_roster(samples, get_many_faces, args.identities)
    if not roster:
        print("\nNESSUN VOLTO RILEVATO. Il video non e' adatto a un face swap.")
        return 2
    if cut == cut:
        print(f"Soglia clustering: {cut:.3f} (scelta dai dati)")
    else:
        print(f"Persone forzate a: {args.identities}")
    provini = save_provini(roster, args.out, info["fps"])

    total = sum(p["count"] for p in roster)
    report = args.out / "persone_trovate.txt"
    lines = [
        "CENSIMENTO VOLTI",
        "=" * 58,
        f"Video            : {args.video.name}",
        f"Durata           : {info['duration']:.2f}s  ({info['frames']} frame @ "
        f"{info['fps']:.2f} fps)",
        f"Persone distinte : {len(roster)}",
        f"Volti rilevati   : {total} (su {len(samples)} frame campionati)",
        "",
        "PERSONA   APPEARIZIONI   PRIMO VISTO   PROVINO",
        "-" * 58,
    ]
    people = []
    for i, p in enumerate(roster):
        first = p["samples"][0]["frame"].stem
        secs = int(first) / max(info["fps"], 1e-6) * max(
            1, info["frames"] // max(len(samples), 1)) / max(info["fps"], 1e-6)
        lines.append(f"persona {i+1}   {p['count']:>6}        frame {first}   "
                     f"{Path(provini[i][0]).name if provini[i] else '-'}")
        people.append({
            "person_id": i + 1,
            "appearances": p["count"],
            "first_seen_frame": first,
            "contact_sheet": provini[i][0] if provini[i] else None,
        })
    lines += [
        "",
        "COME USARE QUESTO FILE",
        "-" * 58,
        "1. Guarda i provini e riconosci chi e' chi.",
        "2. Se le foto sono gia' state indicate con --photo, il config e' pronto:",
        "     python headless_faceswap.py --config swap_config.json",
        "3. Altrimenti aggiungi le foto con --photo e rilancia l'analizzatore,",
        "   oppure passale a mano:",
        "     python headless_faceswap.py --video V.mp4 --photo a.jpg b.jpg --map 0,1",
        "",
        "Attenzione: se i soggetti delle foto NON sono gia' nel video, l'abbinamento",
        "foto/persona non puo' essere dedotto automaticamente. Guardali e compilalo a mano.",
        "",
        "PROMPT DELLA MODALITA' " + args.mode.upper(),
        "-" * 58,
        args.prompt or PROMPTS[args.mode],
        "",
        "Nota: i modelli ONNX usati (inswapper, GFPGAN, BiSeNet) non accettano prompt",
        "di testo. Il prompt e' documentazione di riferimento, non un ingresso del modello.",
        "Userebbe un modello generativo diverso (diffusion + ControlNet/IP-Adapter).",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    config = {
        "source_video": str(args.video.resolve()),
        "mode": args.mode,
        "provider": provider,
        "video": info,
        "prompt": args.prompt or PROMPTS[args.mode],
        "prompt_is_used_by_model": False,
        "reference_photos": [str(p.resolve()) for p in (args.photo or [])],
        "people": people,
    }
    cfg_path = args.out / "swap_config.json"
    cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("\n".join(lines[:11 + len(roster) + 1]))
    print(f"\nRapporto : {report}")
    print(f"Config   : {cfg_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())