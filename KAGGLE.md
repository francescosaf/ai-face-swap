# Kaggle — branch `aifaceswap-grok`

Questo branch parte da `dev` e serve per test headswap su Kaggle (clone diretto del repo, senza zip dataset del codice).

## 1. Repo pubblico

Kaggle può clonare solo repo **pubblici** (o con token).

Su GitHub → Settings del repo `ai-face-swap` → **Change visibility** → **Public**.

## 2. Notebook Kaggle

- Accelerator: GPU T4 (opzionale; su molti ambienti ORT resta CPU)
- Internet: **On**
- Input: solo media (video + foto), es. dataset `faceswap-input`

## 3. Cella unica (clone branch + run headswap)

Vedi anche `scripts/kaggle_headswap.py` oppure incolla la cella documentata lì.

```bash
git clone --branch aifaceswap-grok --depth 1 https://github.com/francescosaf/ai-face-swap.git
cd ai-face-swap
git clone --depth 1 https://github.com/hacksider/Deep-Live-Cam.git dlfolder
# stub UI (niente PySide6)
python scripts/kaggle_stub_ui.py
pip install -q onnxruntime insightface opencv-python-headless
python -u headless_faceswap.py --mode headswap --quality fast --no-enhancer --provider cpu ...
```

## 4. Headswap su questo branch

- Default: ricolorazione capelli/pelle + inswapper (niente trapianto geometrico capelli → no floating hair)
- Serve `bisenet_resnet18.onnx` in `dlfolder/models/`
- `--mode headswap` (non faceswap)

## 5. Limiti

Inswapper 128 non è un head-swap generativo commerciale. GPU cloud commerciali usano altri modelli.
