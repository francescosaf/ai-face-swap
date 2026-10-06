#!/usr/bin/env python3
"""Sostituisce dlfolder/modules/ui.py con uno stub senza PySide6.

Deep-Live-Cam importa la GUI anche in headless; su Kaggle manca PySide6
e l'import di face_swapper fallisce. Esegui questo script dopo il clone.
"""
from __future__ import annotations

import shutil
from pathlib import Path

STUB = '''"""Stub headless: niente PySide6 (Kaggle / server)."""

def create_root():
    return None

def create_source_target():
    pass

def create_preview():
    pass

def update_preview(*args, **kwargs):
    pass

def update_status(msg, scope=None):
    print(f"[status] {msg}", flush=True)

def init(start=None, destroy=None):
    pass

def create_webcam_preview(*args, **kwargs):
    pass

LIVE_PREVIEW = None
ROOT = None
SOURCE = None
TARGET = None
'''


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ui = root / "dlfolder" / "modules" / "ui.py"
    if not ui.parent.is_dir():
        raise SystemExit(f"Manca dlfolder: clona Deep-Live-Cam in {root / 'dlfolder'}")
    if ui.exists():
        bak = ui.with_suffix(".py.bak")
        if not bak.exists():
            shutil.copy(ui, bak)
    ui.write_text(STUB, encoding="utf-8")
    print(f"OK stub scritto in {ui}")


if __name__ == "__main__":
    main()
