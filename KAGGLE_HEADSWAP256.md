# Kaggle head-swap 256 (HifiFace / HyperSwap)

## Perché non InstantID / SDXL qui

Paper di riferimento (InstantID, arXiv:2401.07519): identity-preserving generation con diffusion.
Su T4 free, SDXL+ControlNet per ogni frame di video è troppo lento/instabile (OOM, session kill).

## Cosa usiamo invece (best practice praticabile)

Modelli **256px a testa intera** (HifiFace unofficial / HyperSwap):
- output = testa ricostruita + **maschera del modello** (non ellisse geometrica 128)
- riduce alone grigio / sticker tipici di inswapper_128
- già implementati in `hyper_swap.py` sul branch

Pipeline Kaggle (in-process, no Deep-Live-Cam `get_face_swapper`):
1. InsightFace buffalo_l → detection + embedding
2. HifiFace ONNX 256 → swap testa
3. Write video

Modello: `hififace_unofficial_256.onnx` (~204 MB)
https://huggingface.co/netrunner-exe/Insight-Swap-models-onnx

## Limiti onesti

- Non è diffusion generativa tipo Kling/Magic Hour
- Serve volto frontale/nitido; motion blur → fallback interno di hyper_swap
- Mapping multi-persona grezzo (sinistra→destra) se manca CSV
