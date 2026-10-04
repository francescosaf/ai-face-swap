#!/usr/bin/env python3
"""Sostituzione di volti (o di intere teste) in un video, da N foto di riferimento.

Un solo punto d'ingresso per tre ambienti: Colab, Kaggle, macOS/Linux locale.
Rileva l'hardware e sceglie da solo l'execution provider ONNX
(CUDA > CoreML > CPU), scarica i modelli mancanti, e non duplica la logica:
la condivide con video_face_roster.py.

Su macOS Intel (x86_64) forza onnxruntime==1.23.2 via onnx_compat.py
(le release successive non hanno wheel Intel).

Modalita':
  headswap (default)  sostituisce volto, capelli, orecchie, cappello
  faceswap            sostituisce solo il volto interno (inswapper puro)
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
    """CUDA > CoreML > CPU. Su Mac Intel forza onnxruntime 1.23.2 (CPU)."""
    from onnx_compat import recommended_providers, is_mac_intel, ensure_onnxruntime
    ver = ensure_onnxruntime()
    log(f"[env] onnxruntime {ver}")
    if is_mac_intel():
        log("[env] Mac Intel (x86_64) → CPU con onnxruntime==1.23.2")
        log("      (pip install onnxruntime senza pin fallisce: solo arm64 nelle release recenti)")
    ov = None
    if override:
        if override in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"):
            ov = {"CUDAExecutionProvider": "cuda", "CoreMLExecutionProvider": "coreml",
                  "CPUExecutionProvider": "cpu"}[override]
        else:
            ov = override
    return recommended_providers(ov)
