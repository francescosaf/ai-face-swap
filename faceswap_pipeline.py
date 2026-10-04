#!/usr/bin/env python3
"""
AI Face-Swap Pipeline (Best Practice)
======================================
- Extract frames from source video
- Apply AI face replacement using reference collage
- Rebuild output video
Dependencies: pip install opencv-python numpy pillow
Real engine (optional): roop, insightface, deepface
"""
import os, sys, subprocess, glob
from pathlib import Path

BASE = Path("/Users/francescosaf/Workspace/PersonalPrj/AI face swap")
FRAMES_DIR = BASE / "frames"
OUT_DIR = BASE / "output"
SOURCE_VIDEO = Path("/Users/francescosaf/Downloads/da0a219f-1cda-4ae1-8cbe-6dc5b7ef4538.mp4")
REFERENCE_FACE = Path("/Users/francescosaf/Downloads/Foto Unica Collage Semplice.png")

FRAMES_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_frames(video_path: Path, out_dir: Path):
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", "fps=24", f"{out_dir}/frame_%04d.png"
    ]
    subprocess.run(cmd, check=True)
    print(f"Frames extracted to {out_dir}")


def swap_faces(frames_dir: Path, out_dir: Path, ref_image: Path):
    """Placeholder for AI face-swap engine.
    Replace with roop / insightface / deepface call."""
    for f in sorted(frames_dir.glob("*.png")):
        out_path = out_dir / f.name
        # TODO: integrate AI engine here
        # Example: subprocess.run(["python", "-m", "roop", ...])
        # For now, copy frame with log
        import shutil
        shutil.copy(f, out_path)
        print(f"Processed: {f.name}")


if __name__ == "__main__":
    print("Starting AI face-swap pipeline...")
    # extract_frames(SOURCE_VIDEO, FRAMES_DIR)
    # swap_faces(FRAMES_DIR, OUT_DIR, REFERENCE_FACE)
    print("Pipeline template ready. Uncomment functions after installing engine.")
