# macOS Intel + Python 3.14

## Il vincolo

| Combinazione | Wheel ufficiale |
|--------------|-----------------|
| Mac Intel + Python 3.10–3.13 | sì (`onnxruntime==1.23.2`) |
| Mac Intel + Python 3.14 | **no** |
| Mac arm64 + Python 3.14 | sì (ultime versioni) |
| Linux/Win + Python 3.14 | sì |

Non è un bug del progetto: Microsoft non pubblica binary per macOS x86_64
dalla 1.24 in poi, e la 1.23.2 non ha wheel per 3.14.

## Opzione A — consigliata (2 minuti)

Installa Python 3.12 **accanto** al 3.14 (non lo sostituisce):

```bash
brew install python@3.12

cd /Users/francescosaf/Workspace/PersonalPrj/ai-face-swap-grok
/usr/local/bin/python3.12 -m venv dlfolder/venv312
source dlfolder/venv312/bin/activate
pip install -U pip
pip install "onnxruntime==1.23.2" opencv-python-headless insightface numpy

python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
python headless_faceswap.py --video ... --photo ... --map 0,1,2 --quality high
```

## Opzione B — resta su Python 3.14 (build da sorgente)

```bash
brew install cmake ninja protobuf
bash scripts/build_onnxruntime_mac_intel.sh   # 30–90 min
```

## Opzione C — cloud GPU

Vast.ai / RunPod / Colab: niente limiti di architettura Python.
