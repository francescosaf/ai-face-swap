#!/bin/bash
# Build onnxruntime da sorgente per macOS Intel + Python corrente (anche 3.14).
# Tempo stimato: 30–90 min su i9. Richiede ~4 GB disco e Xcode CLT.
set -euo pipefail

echo "=== Build onnxruntime per $(python3 -c 'import platform; print(platform.machine(), platform.python_version())') ==="
echo "Prerequisiti: brew install cmake ninja protobuf"
command -v cmake >/dev/null || { echo "manca cmake: brew install cmake"; exit 1; }

WORKDIR="${TMPDIR:-/tmp}/ort-build-$$"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

git clone --recursive --depth 1 --branch v1.23.2 \
  https://github.com/microsoft/onnxruntime.git
cd onnxruntime

./build.sh --config Release \
  --build_wheel \
  --parallel \
  --skip_tests \
  --compile_no_warning_as_error \
  --skip_submodule_sync \
  --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=x86_64

WHEEL=$(find build -name 'onnxruntime-*.whl' | head -1)
if [[ -z "$WHEEL" ]]; then
  echo "ERRORE: wheel non trovata"
  exit 1
fi
echo "Wheel: $WHEEL"
python3 -m pip install --force-reinstall "$WHEEL"
python3 -c "import onnxruntime as ort; print('OK', ort.__version__, ort.get_available_providers())"
echo "Fatto. Puoi cancellare $WORKDIR"
