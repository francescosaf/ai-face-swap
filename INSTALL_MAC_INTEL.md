# macOS Intel — onnxruntime

## Problema

`pip install onnxruntime` senza pin fallisce su Mac Intel:

```
ERROR: No matching distribution found for onnxruntime
```

Microsoft non pubblica più wheel macOS x86_64 dopo la 1.23.x.

## Fix (una volta nel venv)

Serve **Python 3.10–3.13** architettura **x86_64** (non arm64 sotto Rosetta).

```bash
python -c "import platform; print(platform.machine(), platform.python_version())"
# atteso: x86_64  3.11.x  (o 3.10/3.12/3.13)

pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null
pip install "onnxruntime==1.23.2"

python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
# atteso: 1.23.2 ['CPUExecutionProvider']
```

Poi:

```bash
git checkout fix/no-floating-hair
python headless_faceswap.py --video ... --photo ... --quality high
```

Lo script chiama `onnx_compat.ensure_onnxruntime()` e pinna da solo se manca.
