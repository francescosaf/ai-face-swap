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


def _iou_iou(a, b) -> float:
    """IoU fra due bbox in formato (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    unione = area_a + area_b - inter
    return inter / unione if unione > 0 else 0.0


def _righe_vicine(mappa_frame, i: int, _voxel: int = 6):
    """Righe dei frame adiacenti, se concordano sulla foto.

    Ritorna [(bbox, foto)] del punto in cui i vicini si incontrano, o None
    se non concordano (due persone diverse in campo, o la scena cambia):
    in quel caso non si indovina e il volto resta com'e'.
    """
    for d in range(1, _voxel + 1):
        raccolte = []
        for j in (i - d, i + d):
            raccolte.extend(mappa_frame.get(j, ()))
        if raccolte:
            if len({idx for _, idx in raccolte}) == 1:
                return raccolte
            return None
    return None


def _scegli_riga(righe, bbox, soglia=1.0):
    """Foto della riga il cui volto e' piu' vicino a `bbox`.

    Ritorna l'indice foto, oppure None se nessuna riga e' abbastanza
    vicina. Serve perche' un frame senza righe puo' contenere una persona
    diversa da quella dei frame adiacenti: darle il volto di quella
    sarebbe uno scambio visibile. La distanza e' normalizzata sulla
    larghezza del volto rilevato: piu' di una larghezza di distanza e'
    un altro soggetto. Un volto che si sposta davvero non supera mai
    quella soglia, uno lontano si.
    """
    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    # normalizzazione sulla larghezza del volto rilevato: due persone in
    # campo stanno a meta' dell'uno dall'altro al massimo, mentre il volto
    # stesso si sposta di molto meno. Con il lato maggiore invece la
    # soglia scattava appena (un volto alto 600px assorbiva 450px di
    # distanza e il secondo soggetto veniva scambiato).
    px = max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1.0)
    migliore, migliore_d = None, None
    for rb, idx in righe:
        rx = (rb[0] + rb[2]) / 2.0
        ry = (rb[1] + rb[3]) / 2.0
        d = ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5 / px
        if migliore_d is None or d < migliore_d:
            migliore, migliore_d = idx, d
    return migliore if migliore_d is not None and migliore_d <= soglia else None


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


SOGLIA_DETECTOR_RIDOTTA = 0.25


def _soglia_ridotta(many):
    """Detector a 0.25 invece che 0.5, avvolgendo get_many_faces.

    A 0.5 i volti appena fuori dal bordo o in moto non escono, e quando un
    volto marginale viene trovato lo scambiato resta quello sbagliato: nel
    video con IG il frame 348 perdeva il soggetto principale. Qui sotto le
    soglie costano 3 volti in piu' su 38, quasi tutti reali, e li scarta
    comunque la mappa. La soglia va rimessa a posto a ogni chiamata: e'
    una proprieta' del modello, condivisa con gli altri passaggi.
    """
    def avvolto(frame):
        from modules.face_analyser import get_face_analyser
        det = get_face_analyser().det_model
        precedente = getattr(det, "det_thresh", None)
        try:
            det.det_thresh = SOGLIA_DETECTOR_RIDOTTA
            return many(frame)
        finally:
            if precedente is not None:
                det.det_thresh = precedente
    return avvolto


def load_modules(providers, enhancer, det_size, mode):
    import modules.globals as g
    g.execution_providers = providers
    g.det_size = det_size
    g.many_faces = False
    g.map_faces = True
    # il detector a 0.5 perde i volti ai margini e in movimento: in
    # faceswap la mappa decide comunque chi scambiare, quindi si puo'
    # abbassare. In headswap cambierebbe il tracciamento, resta 0.5.
    # Va fatto qui e non dentro face_analyser perche' dlfolder e' una
    # repo separata (upstream) esclusa dal versionamento: una modifica
    # li' sparisce al prossimo rebuild.
    if mode == "faceswap":
        many = _soglia_ridotta(many)
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
    ap.add_argument("--no-enhancer", action="store_true",
                    help="disattiva GFPGAN. Sul questo materiale spingeva i "
                         "dettagli della pelle al 400%% dell'originale, e "
                         "quel contrasto in piu' sul video originale si legge "
                         "come sfarfallio.")
    ap.add_argument("--mappa-file", type=Path, default=None,
                    help="CSV frame->foto prodotto da abbina_volti.py. Se "
                         "passato, la foto da usare e' presa dal file per ogni "
                         "frame e il clustering interno viene ignorato: e' il "
                         "modo sicuro perche' i due algoritmi di "
                         "raggruppamento possono dare numeri di persone "
                         "diversi e --map finirebbe sugli indici sbagliati.")
    ap.add_argument("--map", dest="mapping",
                    help="quale foto per ogni persona: es. 0,1,0 (indice foto)")
    ap.add_argument("--identities", type=int, default=0,
                    help="forza il numero di persone distinte nel video")
    ap.add_argument("--det-size", type=int, default=640, choices=[160, 320, 640, 1024],
                    help="dimensione di ingresso del detector. Piu' alta "
                         "trova volti piccoli con keypoint piu' precisi: su "
                         "riprese in movimento e' la leva che riduce davvero "
                         "il tremito, molto piu' del blending temporale.")
    ap.add_argument("--quality", choices=["fast", "high", "maximum"], default="high",
                    help="profilo qualità: fast | high (default) | maximum")
    ap.add_argument("--blend", type=int, default=None, help="sfumatura maschera (override quality)")
    ap.add_argument("--swapper", choices=["inswapper", "hyperswap", "hififace"],
                    default="inswapper",
                    help="modello di sostituzione: inswapper 128px (solo identita', "
                         "conserva gli accessori del video) oppure hyperswap/hififace "
                         "256px, che sostituiscono l'intera testa e restituiscono la "
                         "propria maschera")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="ingrandisce il ritaglio prima dello swap (1.0 = disattivato: "
                         "con HyperSwap abbatte la maschera e lo swap non viene applicato)")
    ap.add_argument("--hair-transfer", type=int, default=None,
                    help="0 disattiva il trapianto geometrico dei capelli "
                         "(override quality): evita l'effetto sticker/cutout")
    ap.add_argument("--harmonize", type=float, default=None,
                    help="0 disattiva l'armonizzazione del tono pelle (headswap)")
    ap.add_argument("--stabilizza", type=float, default=None,
                    help="smorzamento temporale dei keypoint del detector "
                         "(0=off, 1=massimo). E' la leva principale contro "
                         "il flickering: senza, il ritaglio di inswapper "
                         "trema perche' segue il tremito del detector.")
    ap.add_argument("--temporal", type=float, default=None,
                    help="0 disattiva lo smorzamento temporale (override quality)")
    ap.add_argument("--face-scale", type=float, default=None,
                    help="upscale regione facciale (1.0-2.5, override quality)")
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
    log(f"  Quality  : {args.quality}")
    log(f"  Enhancer : {'GFPGAN ATTIVO' if enhancer else 'disattivato (--no-enhancer)'}")
    if args.temporal is not None or args.blend is not None:
        log(f"  Override : temporal={args.temporal}  blend={args.blend}  face_scale={args.face_scale}")
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
    if mode in ("headswap", "faceswap"):
        from head_swap import HeadSwapper
        # Costruisci i kwargs: quality ha priorità, gli override espliciti vincono
        scambiatore = None
        if mode == "faceswap":
            args.harmonize = 0.0
            args.hair_transfer = 0
        head256 = args.swapper != "inswapper"
        # Con uno swapper a 256px la testa viene rigerata: ricolorare i capelli
        # e trapiantarne la geometria dalla foto non serve (ed e' cio' che
        # produceva l'effetto sticker). L'upscale del preset invece resta utile,
        # ma hyper_swap fa gia' da se' il suo zoom sulla sorgente.
        if args.harmonize is None:
            args.harmonize = 0.0 if head256 else 1.0
        if head256 and args.hair_transfer is None:
            args.hair_transfer = 0
        kw = dict(mode=mode, harmonize=args.harmonize, quality=args.quality)
        if args.blend is not None:
            kw["blend"] = args.blend
        if args.temporal is not None:
            kw["temporal"] = args.temporal
        elif mode == "faceswap":
            # Nel faceswap il preset 'high' attivava il blend temporale, che
            # su una testa in movimento trascina l'immagine del frame
            # precedente: risultava piu' sfocato del faceswap diretto, che e'
            # il comportamento migliore. Resta attivabile a mano con
            # --temporal, ma non e' il default.
            kw["temporal"] = 0.0
        if args.stabilizza is not None:
            kw["stabilizza"] = args.stabilizza
        if args.face_scale is not None:
            kw["face_scale"] = args.face_scale
        if args.hair_transfer is not None:
            kw["hair"] = bool(args.hair_transfer)
        swapper = HeadSwapper(DL_DIR / "models", providers, M["swapper"].swap_face, **kw)
        q = HeadSwapper.QUALITY.get(args.quality, {})
        if head256 and mode == "headswap":
            import hyper_swap
            scambiatore = hyper_swap.carica(args.swapper, DL_DIR / "models",
                                            zoom=args.zoom)
            swapper.swapper_fn = scambiatore.swap
            log(f"[headswap] modello: {args.swapper} 256px con maschera propria")
        log(f"[headswap] trapianto capelli = {swapper.hair_transfer}"
            f"   armonizzazione = {swapper.harmonize}"
            f"   face_scale = {swapper.face_scale:.2f}")
        log(f"[headswap] parser BiSeNet pronto  |  semantic={swapper.use_semantic}  "
            f"poisson={swapper.use_poisson}  face_scale={swapper.face_scale:.1f}x  "
            f"temporal={swapper.temporal:.2f}  blend={swapper.blend}")
        log("")
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

        mappa_frame = None
        if args.mappa_file:
            import csv as _csv
            with open(args.mappa_file, newline="") as fh:
                rd = _csv.DictReader(fh)
                if "frame" not in rd.fieldnames or "foto_indice" not in rd.fieldnames:
                    print(f"[errore] {args.mappa_file} non ha le colonne "
                          f"frame e foto_indice")
                    return 1
                mappa_frame = {}
                for riga in rd:
                    if riga["foto_indice"] == "":
                        continue
                    idx = int(riga["foto_indice"])
                    if not -1 <= idx < len(loaded):
                        print(f"[errore] il file punta alla foto {idx}, "
                              f"disponibili 0..{len(loaded)-1}")
                        return 1
                    bbox = (int(riga["x1"]), int(riga["y1"]),
                            int(riga["x2"]), int(riga["y2"]))
                    mappa_frame.setdefault(int(riga["frame"]), []).append((bbox, idx))
            log(f"  Mappa    : {len(mappa_frame)} frame, "
                f"{sum(len(v) for v in mappa_frame.values())} volti da "
                f"{args.mappa_file.name} (clustering interno ignorato)")
            if not mappa_frame:
                log("[errore] la mappa e' vuota: rilassa abbina_volti.py "
                    "con --assegna")
                return 1

        t0 = time.time()
        swapped = 0
        senza_mappa = 0
        lasciati = 0
        frames_persi = []
        ripreso_vicini = 0
        scartati_distanza = 0
        saved = False
        for i, fp in enumerate(frame_paths):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            if swapper is not None:
                swapper.nuovo_frame(i)
            faces = M["many"](frame)
            if faces:
                for f in faces:
                    if f.normed_embedding is None:
                        continue
                    if mappa_frame is not None:
                        # Un frame puo' avere piu' persone: la mappa e' per
                        # frame E per bbox, non solo per frame. Con una mappa
                        # solo per frame i due volti ricevevano la stessa
                        # foto e una delle due persone spariva dal video.
                        righe_frame = mappa_frame.get(i)
                        if not righe_frame:
                            # Il detector del passo di riconoscimento non vede
                            # tutti i frame che il passo di swap vede: capita
                            # su un volto che sfuma o che e' marginale. Il
                            # volto persiste fra frame adiacenti, quindi si
                            # prende l'assegnazione dei vicini invece di
                            # lasciar tornare la persona originale a sorpresa.
                            vicini = _righe_vicine(mappa_frame, i, _voxel=6)
                            if vicini is None:
                                senza_mappa += 1
                                frames_persi.append(i)
                                continue
                            # un frame senza righe puo' contenere un'altra
                            # persona: se la riga vicina non le corrisponde,
                            # il volto resta suo invece di ricevere quello
                            # dei frame adiacenti
                            src = _scegli_riga(vicini, f.bbox)
                            if src is None:
                                scartati_distanza += 1
                                continue
                            ripreso_vicini += 1
                        else:
                            fb = f.bbox.astype(float)
                            src, iou = None, 0.0
                            for bbox, idx in righe_frame:
                                r = _iou_iou(fb, bbox)
                                if r > iou:
                                    src, iou = idx, r
                            if iou < 0.30:
                                # il detector ha riallineato il box fra il
                                # riconoscimento e lo swap. Con una sola riga il
                                # volto e' quasi univoco: usarla invece di
                                # saltare il frame evita che la persona torni
                                # originale, ma solo se il volto rilevato le
                                # corrisponde davvero (altrimenti in scena ce
                                # n'e' un'altra, e la si lascia com'e').
                                if len(righe_frame) == 1:
                                    src = _scegli_riga(righe_frame, f.bbox)
                                    if src is None:
                                        scartati_distanza += 1
                                        continue
                                else:
                                    senza_mappa += 1
                                    frames_persi.append(i)
                                    continue
                        if src == -1:
                            lasciati += 1
                            continue
                    else:
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

        if swapper is not None and swapper._kps_aggiornati:
            log(f"  Stabilizz : {swapper._kps_aggiornati} volti con keypoint "
                f"smorzati, correzione media "
                f"{100 * swapper._kps_correzione / swapper._kps_aggiornati:.2f}% "
                f"del lato del viso")
        if scartati_distanza:
            log(f"  Altre     : {scartati_distanza} volti lasciati com'e' "
                f"(altro soggetto nel frame, non il volto della mappa)")
        if ripreso_vicini:
            log(f"  Vicini    : {ripreso_vicini} volti ripresi dai frame "
                f"adiacenti (il CSV non aveva riga per il frame)")
        if lasciati:
            log(f"  Originali: {lasciati} volti lasciati al volto originale "
                f"su richiesta (mappa = -1)")
        if senza_mappa:
            log(f"  Mappa    : {senza_mappa} volti lasciati intatti, frame "
                f"{frames_persi[:14]}{' ...' if len(frames_persi) > 14 else ''} "
                f"(bbox non combaciante e piu' righe per frame)")
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
        if swapper is not None:
            log(f"  Fallback : {swapper._fallback_usati} frame tenuti dal "
                "frame precedente (il modello non aveva scambiato)")
        if scambiatore is not None:
            log(f"  Modello  : {scambiatore._deblur_usati} frame recuperati "
                f"con la sfocatura inversa, {scambiatore._falliti} senza "
                f"scambio ({scambiatore._riusi} risolti riusando l'ultimo "
                "scambio riuscito)")
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