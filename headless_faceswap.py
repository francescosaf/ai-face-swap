#!/usr/bin/env python3
"""Driver headless per face-swap multi-sorgente (3+ foto -> 1 video).

Rileva automaticamente l'ambiente di esecuzione (Colab, Kaggle, macOS/Linux
locale) e seleziona il migliore execution provider ONNX disponibile:
CUDA > CoreML > CPU.

Sostituisce la GUI Tkinter di Deep-Live-Cam, che non e' utilizzabile su
Colab, e risolve la mappatura N sorgenti -> N identita' distinte nel video,
che nessun tool gestisce in modo nativo.

Uso:
    python headless_faceswap.py \
        --target video.mp4 \
        --source foto1.jpg foto2.jpg foto3.jpg \
        --output risultato.mp4

    # Colab/Kaggle: lo script si adatta da solo, nessun flag necessario.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DL_DIR = HERE / "dlfolder"

MODEL_BASE = "https://huggingface.co/hacksider/deep-live-cam/resolve/main"
MODEL_FILES = {
    "inswapper_128.onnx": f"{MODEL_BASE}/inswapper_128.onnx?download=true",
    "inswapper_128_fp16.onnx": f"{MODEL_BASE}/inswapper_128_fp16.onnx?download=true",
    "gfpgan-1024.onnx": f"{MODEL_BASE}/gfpgan-1024.onnx?download=true",
}



def log(msg: str) -> None:
    print(msg, flush=True)


def detect_environment() -> dict:
    """Identifica l'ambiente di esecuzione e la GPU, senza errori se non presente."""
    in_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
    in_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/input").is_dir()
    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20,
            )
            gpu = out.stdout.strip().splitlines()[0] if out.stdout.strip() else None
        except Exception:
            gpu = None
    if in_colab:
        name = "Google Colab"
    elif in_kaggle:
        name = "Kaggle Notebook"
    elif sys.platform == "darwin":
        name = f"macOS locale ({os.uname().machine})"
    else:
        name = f"Linux locale"
    return {"name": name, "in_colab": in_colab, "in_kaggle": in_kaggle, "gpu": gpu}


def select_providers(env: dict, override: str | None) -> list[str]:
    """Sceglie l'execution provider e, su Colab, installa onnxruntime-gpu se manca."""
    import onnxruntime as ort

    if override:
        chosen = [override]
    else:
        avail = ort.get_available_providers()
        for pref in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"):
            if pref in avail:
                chosen = [pref]
                break
        else:
            raise RuntimeError(f"Nessun execution provider utilizzabile. Disponibili: {avail}")

    wants_cuda = chosen[0] == "CUDAExecutionProvider"
    if wants_cuda and "CUDAExecutionProvider" not in ort.get_available_providers():
        log("[env] CUDA richiesto ma non presente in onnxruntime: installo onnxruntime-gpu")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-U", "onnxruntime-gpu"],
            check=True,
        )
        import importlib
        importlib.reload(ort)
    return [p for p in chosen]


def ensure_models(use_cuda: bool, with_enhancer: bool) -> None:
    """Scarica i modelli mancanti. buffalo_l lo scarica insightface da solo."""
    import urllib.request

    models_dir = DL_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    wanted = ["inswapper_128.onnx"]
    if use_cuda:
        wanted.append("inswapper_128_fp16.onnx")
    if with_enhancer:
        wanted.append("gfpgan-1024.onnx")

    for name in wanted:
        dest = models_dir / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            log(f"[modelli] {name} gia' presente ({dest.stat().st_size/1e6:.0f} MB)")
            continue
        url = MODEL_FILES[name]
        log(f"[modelli] scarico {name} ... ({url})")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        log(f"[modelli] {name} scaricato ({dest.stat().st_size/1e6:.0f} MB)")


def probe_video(path: Path) -> dict:
    exe = shutil.which("ffprobe")
    if not exe:
        raise RuntimeError("ffprobe non trovato nel PATH")
    out = subprocess.run(
        [exe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    import json
    data = json.loads(out.stdout)
    st = data["streams"][0]
    num, den = (st.get("r_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    dur = st.get("duration") or data.get("format", {}).get("duration") or "0"
    return {
        "width": int(st["width"]), "height": int(st["height"]),
        "fps": fps, "duration": float(dur),
        "frames": int(st.get("nb_frames") or 0),
    }


def extract_frames(video: Path, out_dir: Path, fps: float) -> int:
    exe = shutil.which("ffmpeg")
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
           "-fps_mode", "passthrough", "-start_number", "0",
           "-qscale:v", "0", str(out_dir / "%06d.png")]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("*.png")))


def encode_video(frame_dir: Path, source: Path, output: Path, fps: float, crf: int,
                 shortest: bool = False) -> None:
    exe = shutil.which("ffmpeg")
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y",
           "-framerate", f"{fps}", "-i", str(frame_dir / "%06d.png"),
           "-i", str(source),
           "-map", "0:v:0", "-map", "1:a:0?",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart"]
    if shortest:
        cmd.append("-shortest")
    cmd.append(str(output))
    subprocess.run(cmd, check=True)


def load_dl_modules(providers: list[str], enhancer: bool, det_size: int):
    sys.path.insert(0, str(DL_DIR))
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
    return g, get_many_faces, get_one_face, imread_unicode, face_swapper, face_enhancer


def load_sources(paths: list[Path], get_one_face, imread_unicode):
    sources = []
    for p in paths:
        img = imread_unicode(str(p))
        if img is None:
            raise RuntimeError(f"Impossibile leggere la foto sorgente: {p}")
        face = get_one_face(img)
        if face is None or face.normed_embedding is None:
            raise RuntimeError(f"Nessun volto rilevato in: {p}")
        sources.append((p, face, face.normed_embedding))
        log(f"[sorgenti] {p.name}: volto rilevato")
    return sources


def report_source_similarity(sources) -> None:
    """Mostra la matrice di distanza fra le foto sorgente.

    Due foto della stessa persona danno una distanza bassa (<~0.45).
    Serve a capire subito se le 'N foto' sono davvero 'N persone'.
    """
    n = len(sources)
    if n < 2:
        return
    log("\n  Coerenza delle foto sorgente (distanza embedding):")
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - float(sources[i][2] @ sources[j][2])
            tag = "  <- STESSA PERSONA" if d < 0.45 else ""
            log(f"    {sources[i][0].name} vs {sources[j][0].name}: {d:.3f}{tag}")
    log("")


def discover_identities(frame_paths, get_many_faces, sample_every: int, force_k: int = 0):
    """Raggruppa le identita' distinte nel video, come fa la GUI di Deep-Live-Cam."""
    import cv2
    from modules.cluster_analysis import find_cluster_centroids

    embs, locations = [], []
    for i, fp in enumerate(frame_paths):
        if i % sample_every:
            continue
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        faces = get_many_faces(frame)
        if not faces:
            continue
        for f in faces:
            if f.normed_embedding is not None:
                embs.append(f.normed_embedding)
                locations.append(fp.name)
    if not embs:
        return [], []
    stack = np.vstack(embs)
    try:
        if force_k > 1:
            from sklearn.cluster import KMeans
            centroids = KMeans(n_clusters=min(force_k, len(embs)),
                               random_state=0, n_init=10).fit(stack).cluster_centers_
        else:
            centroids = find_cluster_centroids(embs)
    except Exception:
        return embs[:1], [locations[0]]
    first_seen = [locations[int(np.argmax(c @ stack.T))] for c in centroids]
    return list(centroids), first_seen


def pick_assignment(n_ids: int, n_src: int, override: str | None):
    """Restituisce, per ogni identita' del video, l'indice della foto da usare.

    --map accetta una lista piu' corta delle fonti: con 3 identita' e 2 foto
    utili, '--map 0,1,0' e' valido. Le identita' non elencate usano la foto
    omonima, o l'ultima disponibile.
    """
    if not override:
        return [min(i, n_src - 1) for i in range(n_ids)]
    parts = [int(x) for x in override.split(",")]
    bad = [p for p in parts if not 0 <= p < n_src]
    if bad:
        raise RuntimeError(
            f"--map contiene indici fuori intervallo {bad}: le foto sono 0..{n_src-1}")
    if len(parts) < n_ids:
        parts += [parts[-1]] * (n_ids - len(parts))
    return parts[:n_ids]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Face swap multi-sorgente headless, auto-adattivo (Colab/Kaggle/locale).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--target", required=True, type=Path, help="video sorgente (i volti da sostituire)")
    ap.add_argument("--source", required=True, nargs="+", type=Path, help="una o piu foto sorgente")
    ap.add_argument("--output", type=Path, default=Path.home() / "Downloads" / "deep_ai.mp4")
    ap.add_argument("--provider", choices=["cuda", "coreml", "cpu"], help="forza il provider (default: auto)")
    ap.add_argument("--no-enhancer", action="store_true", help="disattiva GFPGAN (piu veloce, peggior qualita')")
    ap.add_argument("--map", dest="mapping", help="quale foto per identita': es. 0,1,0")
    ap.add_argument("--identities", type=int, default=0,
                    help="forza il numero di persone distinte nel video (default: automatico)")
    ap.add_argument("--det-size", type=int, default=640, choices=[160, 320, 640])
    ap.add_argument("--crf", type=int, default=17, help="qualita' H.264 (17=alta, 23=default)")
    ap.add_argument("--limit-frames", type=int, default=0, help="processa solo N frame (test)")
    ap.add_argument("--keep-frames", action="store_true", help="non cancellare i frame intermedi")
    args = ap.parse_args()

    if not DL_DIR.exists():
        log(f"[errore] Deep-Live-Cam non trovato in {DL_DIR}")
        log("        git clone https://github.com/hacksider/Deep-Live-Cam.git dlfolder")
        return 1

    env = detect_environment()
    log("=" * 62)
    log(f"  Ambiente : {env['name']}")
    log(f"  GPU      : {env['gpu'] or 'nessuna rilevata'}")
    log("=" * 62)

    for p in [*args.source, args.target]:
        if not p.exists():
            log(f"[errore] file non trovato: {p}")
            return 1

    override = {"cuda": "CUDAExecutionProvider", "coreml": "CoreMLExecutionProvider",
                "cpu": "CPUExecutionProvider"}.get(args.provider)
    providers = select_providers(env, override)
    use_cuda = providers[0] == "CUDAExecutionProvider"
    log(f"  Provider : {providers[0]}")

    ensure_models(use_cuda, not args.no_enhancer)
    g, get_many_faces, get_one_face, imread_unicode, face_swapper, face_enhancer = \
        load_dl_modules(providers, not args.no_enhancer, args.det_size)

    info = probe_video(args.target)
    log(f"  Video    : {info['width']}x{info['height']} @ {info['fps']:.2f} fps, "
        f"{info['duration']:.2f}s")
    if info["fps"] <= 0:
        log("[errore] fps non determinabile dal video")
        return 1

    sources = load_sources(args.source, get_one_face, imread_unicode)
    report_source_similarity(sources)

    work = Path(tempfile.mkdtemp(prefix="faceswap_"))
    frames_dir = work / "frames"
    frames_dir.mkdir()
    try:
        total = extract_frames(args.target, frames_dir, info["fps"])
        log(f"  Frame    : {total} estratti")
        if args.limit_frames:
            for extra in sorted(frames_dir.glob("*.png"))[args.limit_frames:]:
                extra.unlink()
            total = args.limit_frames
            log(f"            limitati a {total} per il test")

        frame_paths = sorted(frames_dir.glob("*.png"))
        sample = max(1, total // 120)
        ids, first_seen = discover_identities(frame_paths, get_many_faces, sample, args.identities)
        log(f"\n  Identita' distinte nel video: {len(ids)}")
        for i, fn in enumerate(first_seen):
            log(f"    identita' {i}: emersa intorno a {fn}")

        assignment = pick_assignment(len(ids), len(sources), args.mapping)
        log("\n  Assegnazione (identita' del video -> foto):")
        for i, src_idx in enumerate(assignment):
            log(f"    identita' {i} -> {args.source[src_idx].name}")
        if args.mapping is None:
            log("\n  ATTENZIONE: assegnazione automatica per ordine di apparizione.")
            log("  Non e' affidabile: le foto sono di persone diverse da quelle del video,")
            log("  quindi la similarita' degli embedding NON dice quale foto va su quale")
            log("  volto. Controlla il risultato e, se serve, forza con --map 0,1,2")
        log("")

        import cv2
        import numpy as np
        centroid_matrix = np.vstack(ids) if ids else None
        swap_face = face_swapper.swap_face
        enhance_face = face_enhancer.enhance_face if not args.no_enhancer else None
        t0 = time.time()
        swapped = 0

        for i, fp in enumerate(frame_paths):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            faces = get_many_faces(frame)
            if faces and centroid_matrix is not None:
                for f in faces:
                    if f.normed_embedding is None:
                        continue
                    sims = centroid_matrix @ f.normed_embedding
                    ident = int(np.argmax(sims))
                    src = sources[assignment[ident]]
                    frame = swap_face(src[1], f, frame)
                    swapped += 1
            if enhance_face is not None and faces:
                frame = enhance_face(frame, detected_faces=faces)
            cv2.imwrite(str(fp), frame)

            if i % 10 == 0 or i == total - 1:
                done = i + 1
                rate = done / max(time.time() - t0, 1e-6)
                eta = (total - done) / max(rate, 1e-6)
                log(f"  [{done:>4}/{total}] {rate:.2f} fps  ETA {eta/60:5.1f} min  "
                    f"volti scambiati: {swapped}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        encode_video(frames_dir, args.target, args.output, info["fps"], args.crf,
                     shortest=bool(args.limit_frames))
        elapsed = time.time() - t0
        log("=" * 62)
        log(f"  Output : {args.output}")
        log(f"  Tempo  : {elapsed/60:.1f} min  ({total/elapsed:.2f} fps, "
            f"{swapped} swap)")
        log(f"  Atteso : {total/info['fps']:.2f}s di video")
        log("=" * 62)
        return 0
    finally:
        if args.keep_frames:
            log(f"[frame] conservati in {frames_dir}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())