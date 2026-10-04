#!/usr/bin/env python3
"""
AI Face-Swap Pipeline — Best Practice
=====================================
Due percorsi:

1. LOCALE  → Roop (open-source, Python + ffmpeg, qualità buona)
2. CLOUD   → Segmind / Remark AI API (più realistico, pagato)

Entrambi supportano: video + immagine di riferimento → video con volti sostituiti.
"""
import os, sys, subprocess, shutil, tempfile, json, base64
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

BASE = Path("/Users/francescosaf/Workspace/PersonalPrj/AI face swap")
BASE.mkdir(parents=True, exist_ok=True)

SOURCE_VIDEO = Path("/Users/francescosaf/Downloads/da0a219f-1cda-4ae1-8cbe-6dc5b7ef4538.mp4")
REFERENCE = Path("/Users/francescosaf/Downloads/Foto Unica Collage Semplice.png")
OUTPUT = BASE / "faceswapped_output.mp4"

# --- Opzione 1: Roop (locale) -----------------------------------------------
def run_roop(source_face: Path, target_video: Path, output: Path,
             keep_fps: bool = True, face_enhancer: bool = True):
    """Esegui Roop in headless mode.
    Requisiti: git clone https://github.com/s0md3v/roop && pip install -r requirements.txt
    """
    roop_dir = BASE / "roop"
    if not roop_dir.exists():
        print("[Roop] Clonando repo...")
        subprocess.run(["git", "clone", "https://github.com/s0md3v/roop", str(roop_dir)], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(roop_dir / "requirements.txt")],
                       check=True)

    processors = ["face_swapper"]
    if face_enhancer:
        processors.append("face_enhancer")

    cmd = [
        sys.executable, str(roop_dir / "run.py"),
        "-s", str(source_face),
        "-t", str(target_video),
        "-o", str(output),
        "--frame-processor", *processors,
    ]
    if keep_fps:
        cmd.append("--keep-fps")

    print("[Roop] Avvio:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[Roop] Output salvato in: {output}")


# --- Opzione 2: Segmind API (cloud) ------------------------------------------
def segmind_face_swap(source_img: Path, target_video: Path, output: Path,
                      api_key: str = None) -> Path:
    """Face-swap via Segmind AI Face Swap API.
    Documentazione: https://www.segmind.com/models/ai-face-swap/api
    """
    api_key = api_key or os.environ.get("SEGMIND_API_KEY")
    if not api_key:
        raise ValueError("Set SEGMIND_API_KEY env var or pass api_key")

    # Upload files a URL accessibili (usiamo un server temporaneo o base64)
    # Per semplicità, assumiamo che i file siano già su un URL pubblico.
    # In alternativa, usa un servizio di hosting immagini (es. imgbb, tmpfiles).
    print("[Segmind] Richiede URL pubblici per source_img e target_video.")
    print("[Segmind] Carica i file su un hosting (es. tmpfiles.org) e passa le URL qui.")
    raise NotImplementedError("Implementare upload -> URL -> API call")


# --- Opzione 3: Remark AI / Vmodel API (cloud) -------------------------------
def remark_face_swap(target_img: Path, source_video: Path, output: Path,
                     api_key: str = None) -> Path:
    """Face-swap via Remark AI (Vmodel) — video-face-swap-pro.
    Documentazione: https://devweb.vmodel.ai/models/remaker/video-face-swap-pro/
    """
    api_key = api_key or os.environ.get("VModel_API_TOKEN")
    if not api_key:
        raise ValueError("Set VModel_API_TOKEN env var or pass api_key")

    print("[Remark] Caricamento file in corso...")
    # Esempio di upload (implementare con servizio di storage)
    # Poi: POST https://api.vmodel.ai/tasks/v1/create
    raise NotImplementedError("Implementare upload + polling")


# --- Menu --------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("AI Face-Swap Pipeline")
    print("=" * 60)
    print(f"Source video : {SOURCE_VIDEO}")
    print(f"Reference    : {REFERENCE}")
    print(f"Output       : {OUTPUT}")
    print()
    print("Scegli metodo:")
    print("  1) Roop (locale, gratuito, qualità buona)")
    print("  2) Segmind API (cloud, qualità alta, pagato)")
    print("  3) Remark AI / Vmodel API (cloud, qualità eccellente, pagato)")
    print()

    choice = input("Inserisci scelta [1/2/3]: ").strip()

    if choice == "1":
        run_roop(REFERENCE, SOURCE_VIDEO, OUTPUT)
    elif choice == "2":
        key = input("Segmind API key: ").strip()
        segmind_face_swap(REFERENCE, SOURCE_VIDEO, OUTPUT, api_key=key)
    elif choice == "3":
        key = input("VModel API token: ").strip()
        remark_face_swap(REFERENCE, SOURCE_VIDEO, OUTPUT, api_key=key)
    else:
        print("Scelta non valida.")