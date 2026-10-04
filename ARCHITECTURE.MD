# ARCHITECTURE — AI Face-Swap Video Pipeline

## Layer 1 — Input / Reference

- Collage 3-panel (`Foto Unica Collage Semplice.png`)
  - Top: Subject 1 (lead dancer, white shirt, blue jeans)
  - Middle: Subject 2 (selfie holder, flannel shirt)
  - Bottom: Subject 3 (female, white top, black skirt)
- Splitting: `PIL` crop vertical (3 pane, ~1333x666 px ciascuno)
- Output singole: `foto1.jpeg`, `foto2.jpeg`, `foto3.jpeg`

## Layer 2 — Processing Engine

- Deep-Live-Cam (GitHub `hacksider/Deep-Live-Cam`)
  - Frame extraction: `ffmpeg` → `temp/` `%04d.png`
  - Face detection / recognition: `InsightFace` (`buffalo_l` pack)
  - Face swap: `inswapper_128.onnx` (ONNX Runtime)
  - Face enhancement: `GFPGANv1.4.pth`
  - Video reconstruction: `ffmpeg` → `deep_ai.mp4`
- Alternative (se AI locale non disponibile):
  - OpenCV overlay (`cv2.VideoWriter` `avc1`) → `final_result_3subjects.mp4`
  - API cloud: Magic Hour (`api.magichour.ai`), Segmind, Remark AI, Muapi

## Layer 3 — Output

- Vertical 9:16, photorealistic, continuous video
- No text / watermark / logo
- Timeline:
  - 00:00–00:04: Subject 1 dance
  - 00:05–00:06: Selfie transition (Subject 2)
  - 00:07–00:15: Car entrance & walk-up (Subject 3)
  - 00:16: Final atmosphere (Subject 2 + background Subject 1)

## Data Flow

```
Collage PNG → PIL split → foto1/2/3.jpeg
Video MP4 → Deep-Live-Cam → face-swap frames → ffmpeg encode → deep_ai.mp4
```
