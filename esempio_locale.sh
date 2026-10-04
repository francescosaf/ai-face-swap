#!/usr/bin/env bash
# Esempio locale: face swap su macOS Intel, senza GPU CUDA.
# Usa l'auto-rilevamento del driver, che su questo Mac sceglie CoreML.
#
#   ./esempio_locale.sh              -> anteprima rapida (60 frame, ~2.5s di video)
#   ./esempio_locale.sh full         -> render completo del clip
#   ./esempio_locale.sh full         -> render completo con GFPGAN
#   ./esempio_locale.sh noenhancer   -> render completo senza GFPGAN

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$DIR/dlfolder/venv/bin/python"
DL="$HOME/Downloads"

# Il video ha un apostrofo tipografico nel nome: lo cerco con un glob.
VIDEO="$(find "$DL" -maxdepth 1 -name 'Comment PROMPT*Latent.mp4' -print -quit)"
FOTO1="$DL/foto1.jpeg"
FOTO2="$DL/foto2.jpeg"
FOTO3="$DL/foto3.jpeg"
OUT="$DL/deep_ai_local.mp4"

for f in "$VIDEO" "$FOTO1" "$FOTO2" "$FOTO3"; do
  [ -f "$f" ] || { echo "file non trovato: $f"; exit 1; }
done

ARGS=(--target "$VIDEO"
      --source "$FOTO1" "$FOTO2" "$FOTO3"
      --output "$OUT")

case "${1:-preview}" in
  preview)   ARGS+=(--limit-frames 60 --no-enhancer) ;;
  full)      : ;;
  noenhancer) ARGS+=(--no-enhancer) ;;
  *) echo "uso: $0 [preview|full|noenhancer]"; exit 2 ;;
esac

echo "Sorgente video : $VIDEO"
echo "Uscita         : $OUT"
"$PY" "$DIR/headless_faceswap.py" "${ARGS[@]}"