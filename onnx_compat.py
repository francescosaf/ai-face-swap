"""Compatibilita' ONNX Runtime per macOS Intel (x86_64).

Microsoft ha smesso di pubblicare wheel macOS x86_64 da onnxruntime > 1.23.x.
Su Mac Intel va usata obbligatoriamente onnxruntime==1.23.2 (CPU).
Su Apple Silicon (arm64) si puo' usare l'ultima versione con CoreML.
"""

from __future__ import annotations

import platform
import subprocess
import sys


def is_mac_intel() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in ("x86_64", "i386")


def is_mac_arm() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in ("arm64", "aarch64")


def ensure_onnxruntime() -> str:
    """Garantisce un onnxruntime utilizzabile. Ritorna la versione installata.

    - macOS Intel  -> forza onnxruntime==1.23.2 (ultima con wheel x86_64)
    - macOS arm64  -> onnxruntime recente (CoreML)
    - Linux/Win    -> lascia pure (CUDA se disponibile)
    """
    import importlib

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
            major, minor, patch = parts[0], parts[1], parts[2]
            if (major, minor) > (1, 23):
                need_install = True
            if (major, minor) < (1, 20):
                need_install = True
        if need_install:
            print(f"[env] Mac Intel (x86_64): installo {target} (ultima con wheel ufficiale)")
            print("      (le versioni successive supportano solo arm64 su macOS)")
            print(f"      Python: {sys.version.split()[0]}  arch: {platform.machine()}")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", target],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                py = f"{sys.version_info.major}.{sys.version_info.minor}"
                raise RuntimeError(
                    f"Impossibile installare {target} per Python {py} su macOS Intel.\n"
                    f"  Serve Python 3.10–3.13 (x86_64). Verifica:\n"
                    f"    python -c \"import platform; print(platform.machine(), platform.python_version())\"\n"
                    f"  Poi: pip install '{target}'\n"
                    f"  Se pip dice 'from versions: none', stai usando un Python troppo nuovo (es. 3.14)\n"
                    f"  o un venv arm64 sotto Rosetta. Ricrea il venv con Python 3.11/3.12 Intel."
                ) from e
            if "onnxruntime" in sys.modules:
                del sys.modules["onnxruntime"]
            import onnxruntime as ort
            importlib.reload(ort)
            ver = ort.__version__
            print(f"[env] onnxruntime {ver} pronto (CPUExecutionProvider)")
    elif ver is None:
        pkg = "onnxruntime"
        print(f"[env] installo {pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
        import onnxruntime as ort
        ver = ort.__version__

    return ver or "?"


def recommended_providers(override: str | None = None) -> list[str]:
    """CUDA > CoreML > CPU, con consapevolezza architettura Mac."""
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
            print("[env] Mac Intel: CoreML non affidabile con onnxruntime 1.23 → CPU")
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
