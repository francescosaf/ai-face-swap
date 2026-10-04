# Kaggle Notebook — HeadSwap Pipeline (3 soggetti, ordine di apparizione)

## 1. Carica dati (Input → Notebook)
- Video: `Comment PROMPT...Latent.mp4`
- Foto: `foto1.jpeg`, `foto2.jpeg`, `foto3.jpeg`
- Modelli: `GFPGANv1.4.pth`, `inswapper_128.onnx`, `buffalo_l` (se non già presenti, scarica con `wget`)

## 2. Clona / usa il codice esistente
```python
!git clone https://github.com/your-repo/AI-face-swap.git  # oppure carica il tuo workspace
```
Ma puoi anche usare i file locali già nel worktree (`/Users/francescosaf/Workspace/PersonalPrj/AI face swap/`).

## 3. Installa dipendenze (GPU attiva su Kaggle)
```python
!pip install -q onnxruntime-gpu opencv-python insightface facexlib gfpgan
!pip install -q git+https://github.com/deepinsight/insightface
!pip install -q deep-live-cam
```

## 4. Verifica GPU
```python
!nvidia-smi
```
Se vedi `T4` o `P100`, sei a posto.

## 5. Avvia i 3 passaggi (sequenziali, ordine apparizione)
```bash
# Passaggio 1: foto1 -> primo viso
!python run.py -s foto1.jpeg -t "Comment PROMPT...Latent.mp4" -o output/foto1_head.mp4 --execution-provider cuda --keep-fps --many-faces --frame-processor face_swapper face_enhancer --det-size 640 --quality high

# Passaggio 2: foto2 -> secondo viso
!python run.py -s foto2.jpeg -t "Comment PROMPT...Latent.mp4" -o output/foto2_head.mp4 --execution-provider cuda --keep-fps --many-faces --frame-processor face_swapper face_enhancer --det-size 640 --quality high

# Passaggio 3: foto3 -> terzo viso
!python run.py -s foto3.jpeg -t "Comment PROMPT...Latent.mp4" -o output/foto3_head.mp4 --execution-provider cuda --keep-fps --many-faces --frame-processor face_swapper face_enhancer --det-size 640 --quality high
```

> **Nota**: `--many-faces` + `map-faces` non mappa per ordine; se vuoi il mapping sequenziale preciso (1° viso = foto1, 2° = foto2, 3° = foto3), devi usare `apply_headswap_fixed.py` (o un wrapper con `HeadSwapper`).

## 6. Se vuoi il hair transfer (testa completa, non solo volto)
Usa il codice corretto (`head_swap.py` v3.1) nel worktree:
```python
from head_swap import HeadSwapper
# ... setup con foto sorgente, target face, frame
```
Questo richiede il `swapper_fn` reale (non placeholder) — vedi `dlfolder/modules/processors/frame/face_swapper.py`.

## 7. Scarica risultati
```python
from google.colab import files  # o da Kaggle: clicca su file nel pannello Output
files.download('output/foto1_head.mp4')
```

## Tempo stimato su Kaggle (GPU)
- Per video (411 frame): **~5-15 minuti** (`cuda`, ~2 sec/frame vs ~20 sec CPU)
- Per 3 video: **~15-45 minuti totali** (molto meno delle 6h su CPU)
