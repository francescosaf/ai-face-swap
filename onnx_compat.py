"""Compatibilita' ONNX Runtime per macOS Intel (x86_64) e Python recenti.

Vincoli reali (2025–2026):
- onnxruntime > 1.23.x: NESSUN wheel macOS x86_64 (solo arm64).
- onnxruntime 1.23.2: ultima con wheel Intel, ma solo Python 3.10–3.13.
- Python 3.14 + macOS Intel: nessun binary ufficiale → serve
  (A) Python 3.12 via Homebrew, oppure (B) build da sorgente.
"""

from __future__ import annotations

import platform
import subprocess
import sys


def is_mac_intel() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in ("x86_64", "i386")


def is_mac_arm() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in ("arm64", "aarch64")


def _py_tuple() -> tuple[int, int]:
    return sys.version_info.major, sys.version_info.minor


def _mac_intel_help() -> str:
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    return f"""
========================================================================
  BLOCCO: Python {py} su macOS Intel (x86_64)
========================================================================
  Non esiste un wheel ufficiale di onnxruntime per questa combinazione.

  Microsoft ha smesso di pubblicare binary macOS Intel dopo 1.23.x.
  La 1.23.2 supporta solo Python 3.10–3.13.

  --- Opzione A (consigliata, ~2 minuti) ---
  Installa Python 3.12 ACCANTO al 3.14 (non lo rimuove):

    brew install python@3.12
    /usr/local/bin/python3.12 -m venv dlfolder/venv312
    source dlfolder/venv312/bin/activate
    pip install -U pip
    pip install "onnxruntime==1.23.2" opencv-python-headless insightface numpy
    # poi lancia headless_faceswap.py dal venv312

  --- Opzione B (build da sorgente, 30–90 min) ---
  Vedi: scripts/build_onnxruntime_mac_intel.sh
  Richiede: cmake, Xcode CLT, ~4 GB spazio.

  --- Opzione C ---
  Cloud GPU (Vast.ai / RunPod / Colab) con CUDA: nessun limite Python/arch.
========================================================================
"""


def ensure_onnxruntime() -> str:
    """Garantisce un onnxruntime utilizzabile. Ritorna la versione installata."""
    import importlib

    major, minor = _py_tuple()

    # Mac Intel + Python >= 3.14: impossibile via pip
    if is_mac_intel() and (major, minor) >= (3, 14):
        try:
            import onnxruntime as ort  # noqa: F401
            return ort.__version__
        except ImportError:
            raise RuntimeError(_mac_intel_help()) from None

    target = None
    if is_mac_intel():
        target = "onnxruntime==1.23.2"

    try:
        import onnxruntime as ort
        ver = ort.__version__
    except ImportError:
        ver = None

    if is_mac_intel():
        need_install = ver is None
        if ver is not None:
            parts = [int(x) for x in ver.split(".")[:3] if x.isdigit()]
            while len(parts) < 3:
                parts.append(0)
            maj, min_, _ = parts[0], parts[1], parts[2]
            if (maj, min_) > (1, 23) or (maj, min_) < (1, 20):
                need_install = True
        if need_install:
            print(f"[env] Mac Intel: installo {target} (ultima wheel ufficiale x86_64)")
            print(f"      Python {major}.{minor}  arch={platform.machine()}")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", target],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(_mac_intel_help()) from e
            if "onnxruntime" in sys.modules:
                del sys.modules["onnxruntime"]
            import onnxruntime as ort
            importlib.reload(ort)
            ver = ort.__version__
            print(f"[env] onnxruntime {ver} pronto (CPUExecutionProvider)")
    elif ver is None:
        print("[env] installo onnxruntime ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"], check=True)
        import onnxruntime as ort
        ver = ort.__version__

    return ver or "?"


def recommended_providers(override: str | None = None) -> list[str]:
    """CUDA > CoreML > CPU, consapevole di architettura Mac."""
    ensure_onnxruntime()
    import onnxruntime as ort

    if override:
        mapping = {
            "cuda": "CUDAExecutionProvider",
            "coreml": "CoreMLExecutionProvider",
            "cpu": "CPUExecutionProvider",
        }
        chosen = mapping.get(override, override)
        if is_mac_intel() and chosen == "CoreMLExecutionProvider":
            print("[env] Mac Intel: CoreML poco utile con ORT 1.23 → CPU")
            chosen = "CPUExecutionProvider"
        return [chosen]

    avail = ort.get_available_providers()
    if is_mac_intel():
        order = ("CPUExecutionProvider",)
    else:
        order = ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider")

    for p in order:
        if p in avail:
            return [p]
    raise RuntimeError(f"Nessun provider ONNX utilizzabile. Disponibili: {avail}")
