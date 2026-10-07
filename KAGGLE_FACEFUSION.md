# FaceFusion su Kaggle

## Avvertenza ToS / account

Port non ufficiali segnalano che Kaggle può **bannare** account che scaricano/eseguono FaceFusion (policy deepfake).
Usi a tuo rischio. Preferibile: PC con GPU NVIDIA o cloud (RunPod) dove controlli tu l'ambiente.

## Perché FaceFusion

Open source, CUDA, modelli 256 (HyperSwap ecc.), enhancer, CLI headless.
Meglio di inswapper-only su CPU se `CUDAExecutionProvider` è attivo.

## Limiti

- Di solito **una foto sorgente** per run (multi-persona = più passaggi o face selector)
- Install pesante; session Kaggle può killare
- `install.py` vuole conda: su Kaggle usare `--skip-conda`
- Verificare sempre: `onnxruntime.get_available_providers()` contenga `CUDAExecutionProvider`

## CLI tipica (headless)

```bash
python facefusion.py headless-run \
  -s /path/foto.jpg \
  -t /path/video.mp4 \
  -o /path/out.mp4 \
  --execution-providers cuda \
  --face-swapper-model hyperswap_1a_256 \
  --output-video-quality 80
```

(I nomi esatti dei modelli dipendono dalla versione FaceFusion clonato.)
