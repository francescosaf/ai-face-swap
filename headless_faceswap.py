#!/usr/bin/env python3
"""Sostituzione di volti (o di intere teste) in un video, da N foto di riferimento.

Un solo punto d'ingresso per tre ambienti: Colab, Kaggle, macOS/Linux locale.
Rileva l'hardware e sceglie da solo l'execution provider ONNX
(CUDA > CoreML > CPU), scarica i modelli mancanti, e non duplica la logica:
la condivide con video_face_roster.py.

Modalita':
  headswap (default)  sostituisce volto, capelli, orecchie, cappello
  faceswap            sostituisce solo il volto interno (inswapper puro)

Esempi:
    # dal config prodotto dall'analizzatore
    python headless_faceswap.py --config swap_config.json

    # esplicito, con N foto qualsiasi
    python headless_faceswap.py --video clip.mp4 --photo a.jpg b.jpg c.jpg

    # solo una foto, modalita' volto
    python headless_faceswap.py --video clip.mp4 --photo a.jpg --mode faceswap
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DL_DIR = HERE / "dlfolder"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DL_DIR))

MODEL_BASE = "https://huggingface.co/hacksider/deep-live-cam/resolve/main"
MODEL_FILES = {
    "inswapper_128.onnx": f"{MODEL_BASE}/inswapper_128.onnx?download=true",
    "inswapper_128_fp16.onnx": f"{MODEL_BASE}/inswapper_128_fp16.onnx?download=true",
    "gfpgan-1024.onnx": f"{MODEL_BASE}/gfpgan-1024.onnx?download=true",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def detect_environment() -> dict:
    """Identifica l'ambiente e la GPU. Non solleva errori se non c'e' nulla."""
    in_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
    in_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/input").is_dir()
    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20)
            gpu = r.stdout.strip().splitlines()[0] if r.stdout.strip() else None
        except Exception:
            gpu = None
    if in_colab:
        name = "Google Colab"
    elif in_kaggle:
        name = "Kaggle Notebook"
    elif sys.platform == "darwin":
        name = f"macOS locale ({os.uname().machine})"
    else:
        name = "Linux locale"
    return {"name": name, "in_colab": in_colab, "in_kaggle": in_kaggle, "gpu": gpu}


def select_providers(override: str | None) -> list[str]:
    """CUDA > CoreML > CPU. Su Colab installa onnxruntime-gpu se manca."""
    import onnxruntime as ort

    if override:
        chosen = [override]
    else:
        avail = ort.get_available_providers()
        chosen = next((p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider",
                                    "CPUExecutionProvider") if p in avail), None)
        if chosen is None:
            raise RuntimeError(f"Nessun provider utilizzabile. Disponibili: {avail}")
    if chosen == "CUDAExecutionProvider" and \
            "CUDAExecutionProvider" not in ort.get_available_providers():
        log("[env] CUDA non presente in onnxruntime: installo onnxruntime-gpu")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                        "onnxruntime-gpu"], check=True)
        import importlib
        importlib.reload(ort)
    return [chosen]


def ensure_models(use_cuda: bool, enhancer: bool, headswap: bool) -> None:
    """Scarica i modelli mancanti. buffalo_l lo scarica insightface da solo."""
    import urllib.request

    mdir = DL_DIR / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    wanted = ["inswapper_128.onnx"]
    if use_cuda:
        wanted.append("inswapper_128_fp16.onnx")
    if enhancer:
        wanted.append("gfpgan-1024.onnx")
    for name in wanted:
        dest = mdir / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            log(f"[modelli] {name} presente ({dest.stat().st_size/1e6:.0f} MB)")
            continue
        log(f"[modelli] scarico {name} ...")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_FILES[name], tmp)
        tmp.rename(dest)
    if headswap:
        from face_parsing import ensure_parsing_model
        ensure_parsing_model(mdir)
        log("[modelli] bisenet_resnet18.onnx pronto per il parsing")


def probe_video(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
                        "-show_entries", "format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    st = d["streams"][0]
    num, den = (st.get("r_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return {"width": int(st["width"]), "height": int(st["height"]), "fps": fps,
            "frames": int(st.get("nb_frames") or 0),
            "duration": float(d.get("format", {}).get("duration") or 0)}


def extract_frames(video: Path, out_dir: Path) -> int:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(video), "-fps_mode", "passthrough",
                    "-start_number", "0", "-qscale:v", "0",
                    str(out_dir / "%06d.png")], check=True)
    return len(list(out_dir.glob("*.png")))


def encode_video(frame_dir: Path, source: Path, output: Path, fps: float,
                 crf: int, shortest: bool = False) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-framerate", f"{fps}", "-i", str(frame_dir / "%06d.png"),
           "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart"]
    if shortest:
        cmd.append("-shortest")
    cmd.append(str(output))
    subprocess.run(cmd, check=True)


def load_modules(providers, enhancer, det_size, mode):
    import modules.globals as g
    g.execution_providers = providers
    g.det_size = det_size
    g.many_faces = False
    g.map_faces = True
    g.opacity = 1.0
    g.mouth_mask = False
    g.sharpness = 0.0
    g.video_encoder = "libx264"
    g.video_quality = 17
    g.fp_ui = {"face_enhancer": enhancer, "face_enhancer_gpen256": False,
               "face_enhancer_gpen512": False}
    from modules import imread_unicode
    from modules.face_analyser import get_many_faces, get_one_face
    from modules.processors.frame import face_swapper, face_enhancer
    face_swapper.get_face_swapper()
    log("[modelli] inswapper caricato")
    if enhancer:
        face_enhancer.get_face_enhancer()
        log("[modelli] GFPGAN caricato")
    return dict(g=g, imread=imread_unicode, many=get_many_faces, one=get_one_face,
                swapper=face_swapper, enhancer_mod=face_enhancer)


def load_photos(paths, one, imread):
    out = []
    for p in paths:
        img = imread(str(p))
        if img is None:
            raise RuntimeError(f"foto non leggibile: {p}")
        face = one(img)
        if face is None or face.normed_embedding is None:
            raise RuntimeError(f"nessun volto rilevato in: {p}")
        out.append((p, img, face))
        log(f"[foto] {p.name}: volto rilevato")
    return out


def report_similarity(photos):
    for i in range(len(photos)):
        for j in range(i + 1, len(photos)):
            d = 1.0 - float(photos[i][2].normed_embedding @ photos[j][2].normed_embedding)
            log(f"    {photos[i][0].name} vs {photos[j][0].name}: {d:.3f}"
                + ("   <- STESSA PERSONA" if d < 0.45 else ""))


def resolve_inputs(args) -> tuple[Path, list[Path], str | None, str, Path | None]:
    """Unifica config JSON e argomenti in linea. Gli argomenti hanno precedenza."""
    cfg = {}
    cfg_path = None
    if args.config:
        cfg_path = Path(args.config)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    video = args.video or cfg.get("source_video")
    photos = [Path(p) for p in (args.photo or [])]
    prompt = args.prompt
    mode = args.mode

    if not photos and cfg.get("reference_photos"):
        for entry in cfg["reference_photos"]:
            raw = entry if isinstance(entry, str) else \
                (entry.get("path") or entry.get("reference_photo") or "")
            cand = Path(raw)
            if cand.exists():
                photos.append(cand)
            else:
                log(f"[config] foto mancante, ignorata: {cand}")
        if photos:
            log(f"[config] {len(photos)} foto lette dal config")
    if mode is None:
        mode = cfg.get("mode")
    if prompt is None:
        prompt = cfg.get("prompt")

    if not video:
        raise SystemExit("manca il video: usa --video oppure --config")
    if not photos:
        raise SystemExit("mancano le foto: usa --photo N path oppure un --config valido")
    return Path(video), photos, prompt, (mode or "headswap"), cfg_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sostituzione di volti/teste in un video, da N foto.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, help="video da processare")
    ap.add_argument("--photo", nargs="+", type=Path, help="N foto di riferimento")
    ap.add_argument("--config", type=Path, help="config JSON prodotto dall'analizzatore")
    ap.add_argument("--output", type=Path, default=Path.home() / "Downloads" / "faceswap_out.mp4")
    ap.add_argument("--mode", choices=["headswap", "faceswap"],
                    help="default: headswap")
    ap.add_argument("--prompt", help="prompt di riferimento (documentale: i modelli ONNX "
                                     "non accettano testo)")
    ap.add_argument("--provider", choices=["cuda", "coreml", "cpu"])
    ap.add_argument("--no-enhancer", action="store_true", help="disattiva GFPGAN")
    ap.add_argument("--map", dest="mapping",
                    help="quale foto per ogni persona: es. 0,1,0 (indice foto)")
    ap.add_argument("--identities", type=int, default=0,
                    help="forza il numero di persone distinte nel video")
    ap.add_argument("--det-size", type=int, default=640, choices=[160, 320, 640])
    ap.add_argument("--blend", type=int, default=9, help="sfumatura maschera (headswap)")
    ap.add_argument("--harmonize", type=float, default=1.0,
                    help="0 disattiva l'armonizzazione del tono pelle (headswap)")
    ap.add_argument("--temporal", type=float, default=0.35,
                    help="0 disattiva lo smorzamento temporale (0-1, headswap)")
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--limit-frames", type=int, default=0, help="test: procesa N frame")
    ap.add_argument("--save-masks", type=Path, help="salva le maschere del primo frame")
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()

    if not DL_DIR.exists():
        log(f"[errore] Deep-Live-Cam manca in {DL_DIR}")
        log("  git clone https://github.com/hacksider/Deep-Live-Cam.git dlfolder")
        return 1

    video, photos, prompt, mode, cfg_path = resolve_inputs(args)
    env = detect_environment()
    log("=" * 64)
    log(f"  Ambiente : {env['name']}")
    log(f"  GPU      : {env['gpu'] or 'nessuna'}")
    if cfg_path:
        log(f"  Config   : {cfg_path}")

    for p in [video, *photos]:
        if not p.exists():
            log(f"[errore] file non trovato: {p}")
            return 1

    override = {"cuda": "CUDAExecutionProvider", "coreml": "CoreMLExecutionProvider",
                "cpu": "CPUExecutionProvider"}.get(args.provider)
    providers = select_providers(override)
    enhancer = not args.no_enhancer
    log(f"  Modalita': {mode}")
    log(f"  Provider : {providers[0]}")
    if prompt:
        log(f"  Prompt   : registrato (documentale, non usato dai modelli ONNX)")
    if mode == "headswap":
        log("             parser: BiSeNet 19 classi -> maschera capelli/orecchie/cappello")
    log("=" * 64)

    ensure_models(providers[0] == "CUDAExecutionProvider", enhancer, mode == "headswap")
    M = load_modules(providers, enhancer, args.det_size, mode)

    info = probe_video(video)
    log(f"  Video    : {info['width']}x{info['height']} @ {info['fps']:.2f} fps, "
        f"{info['duration']:.2f}s, {info['frames']} frame")

    loaded = load_photos(photos, M["one"], M["imread"])
    if len(loaded) > 1:
        log("\n  Coerenza delle foto (stessa persona se < 0.45):")
        report_similarity(loaded)
        log("")

    swapper = None
    if mode == "headswap":
        from head_swap import HeadSwapper
        swapper = HeadSwapper(DL_DIR / "models", providers, M["swapper"].swap_face,
                              mode="headswap", blend=args.blend,
                              harmonize=args.harmonize, temporal=args.temporal)
        log("[headswap] parser BiSeNet pronto\n")
    entries = [swapper.prepare_source(img, face) for _, img, face in loaded] if swapper else None

    from video_face_roster import build_tracks, cluster_identities

    work = Path(tempfile.mkdtemp(prefix="faceswap_"))
    fdir = work / "frames"
    fdir.mkdir()
    try:
        total = extract_frames(video, fdir)
        if args.limit_frames:
            for extra in sorted(fdir.glob("*.png"))[args.limit_frames:]:
                extra.unlink()
            total = args.limit_frames
        frame_paths = sorted(fdir.glob("*.png"))
        log(f"  Frame    : {total} estratti")
        if total == 0:
            log("[errore] nessun frame estratto")
            return 1

        step = max(1, total // 100)
        tracks = build_tracks(frame_paths[::step], M["many"])
        tracks = [t for t in tracks if t["count"] >= 2]
        if not tracks:
            log("[errore] nessun volto stabile rilevato: il video non e' adatto")
            return 2
        labels, cut = cluster_identities([t["embedding"] for t in tracks], args.identities)
        groups: dict[int, dict] = {}
        for lbl, t in zip(labels, tracks):
            gr = groups.setdefault(int(lbl), {"embedding": [], "count": 0, "first": None})
            gr["embedding"].append(t["embedding"])
            gr["count"] += t["count"]
            f = min(str(m["frame"]) for m in t["members"])
            if gr["first"] is None or f < gr["first"]:
                gr["first"] = f
        people = []
        for gr in groups.values():
            people.append({"centroid": np.mean(np.vstack(gr["embedding"]), axis=0),
                           "count": gr["count"], "first": gr["first"]})
        # come nel referto, le persone sono numerate in ordine di prima
        # apparizione: "persona 1" e' la prima che si vede nel video
        people.sort(key=lambda x: x["first"])

        if args.mapping:
            sel = [int(x) for x in args.mapping.split(",")]
            bad = [s for s in sel if not 0 <= s < len(loaded)]
            if bad:
                log(f"[errore] --map fuori intervallo: {bad} (foto disponibili 0..{len(loaded)-1})")
                return 1
            if len(sel) < len(people):
                sel += [sel[-1]] * (len(people) - len(sel))
        else:
            sel = [min(i, len(loaded) - 1) for i in range(len(people))]

        log(f"  Persone  : {len(people)} distinte")
        if cut == cut:
            log(f"  Soglia   : {cut:.3f} (dalle distanze fra volti)")
        for i, (p, s) in enumerate(zip(people, sel)):
            log(f"    persona {i+1} -> {photos[s].name}   ({p['count']} rilevamenti)")
        if not args.mapping:
            log("\n  NOTA: le persone sono in ordine di apparizione e le foto vengono")
            log("  abbinate in quell'ordine. Se i soggetti delle foto non sono gia'")
            log("  nel video l'abbinamento non e' verificabile da solo: usa")
            log("  video_face_roster.py e poi --map per fissarlo a mano.\n")

        centroids = np.vstack([p["centroid"] for p in people])
        t0 = time.time()
        swapped = 0
        saved = False
        for i, fp in enumerate(frame_paths):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            faces = M["many"](frame)
            if faces:
                for f in faces:
                    if f.normed_embedding is None:
                        continue
                    who = int(np.argmax(centroids @ f.normed_embedding))
                    src = sel[who]
                    if swapper is not None:
                        frame = swapper.swap(frame, entries[src], f)
                    else:
                        frame = M["swapper"].swap_face(loaded[src][2], f, frame)
                    swapped += 1
                    if args.save_masks and not saved:
                        saved = True
                        dump_masks(args.save_masks, swapper, frame, f,
                                   entries[src] if swapper else None)
            if enhancer and faces:
                frame = M["enhancer_mod"].enhance_face(frame, detected_faces=faces)
            cv2.imwrite(str(fp), frame)
            if i % 10 == 0 or i == total - 1:
                done = i + 1
                rate = done / max(time.time() - t0, 1e-6)
                log(f"  [{done:>4}/{total}] {rate:.2f} fps  "
                    f"ETA {(total-done)/max(rate,1e-6)/60:5.1f} min  swap: {swapped}")

        if swapped == 0:
            log("\n[errore] NESSUN volto sostituito. L'output sarebbe identico all'originale.")
            log("         Cause probabili: volti troppo picchi, di profilo, o coperti.")
            return 3

        args.output.parent.mkdir(parents=True, exist_ok=True)
        encode_video(fdir, video, args.output, info["fps"], args.crf,
                     shortest=bool(args.limit_frames))
        el = time.time() - t0
        log("=" * 64)
        log(f"  Output   : {args.output}")
        log(f"  Scambi   : {swapped} su {total} frame in {el/60:.1f} min "
            f"({total/el:.2f} fps)")
        log(f"  Video    : {total/info['fps']:.2f}s")
        log("=" * 64)
        return 0
    finally:
        if args.keep_frames:
            log(f"[frame] conservati in {fdir}")
        else:
            shutil.rmtree(work, ignore_errors=True)


def dump_masks(out: Path, swapper, frame, face, entry):
    """Salva maschere e viste di debug del primo volto processato."""
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "dopo_swap.png"), frame)
    if swapper is None:
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(str(out / "originale.png"), frame)
        cv2.imwrite(str(out / "bbox_volto.png"), frame)
        return
    from face_parsing import (FACE_CLASSES, HAIR_CLASSES, HEAD_CLASSES,
                              FaceParser)
    labels, box = swapper.parser.parse(frame, face.bbox)
    cv2.imwrite(str(out / "originale.png"), frame)
    cv2.imwrite(str(out / "mappa_classi.png"), labels)
    for name, cls in (("maschera_capelli", HAIR_CLASSES), ("maschera_testa", HEAD_CLASSES),
                      ("maschera_viso", FACE_CLASSES)):
        cv2.imwrite(str(out / f"{name}.png"), FaceParser.mask_for(labels, cls))
    vis = frame.copy()
    for cls, col in ((HAIR_CLASSES, (0, 0, 255)), (HEAD_CLASSES, (0, 255, 0))):
        cs, _ = cv2.findContours(FaceParser.mask_for(labels, cls),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, col, 2)
    cv2.imwrite(str(out / "contorni.png"), vis)


if __name__ == "__main__":
    sys.exit(main())