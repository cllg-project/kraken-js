#!/usr/bin/env bash
#
# Fetch the PP-OCRv6 test model and export it to tests/fixtures/ppocr.js_mlmodel.
#
# The fixture is ~57 MB and is deliberately not committed; the PP-OCR tests skip
# when it is absent. Requires a Kraken >= 7.1 virtualenv at ./env (see README).
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL='https://zenodo.org/records/22232579/files/ppocr_v6_tau090.safetensors?download=1'
SHA256='c3228d34fb46693c2061aa0b1b030dfcce18b4fb24ff98dfb5c0cee24b8c3637'
SRC="$ROOT/tests/fixtures/ppocr.safetensors"
OUT="$ROOT/tests/fixtures/ppocr.js_mlmodel"
PYTHON="$ROOT/env/bin/python3"

if [ ! -x "$PYTHON" ]; then
  echo "error: no Kraken virtualenv at $ROOT/env (see README: Model format)" >&2
  exit 1
fi

if [ ! -f "$SRC" ]; then
  echo "Downloading PP-OCRv6 model …"
  curl -fSL --progress-bar -o "$SRC" "$URL"
fi

echo "Verifying checksum …"
echo "$SHA256  $SRC" | sha256sum --check --status || {
  echo "error: checksum mismatch for $SRC — delete it and retry" >&2
  exit 1
}

echo "Exporting to $OUT …"
"$PYTHON" "$ROOT/export_kraken_onnx.py" "$SRC" "$OUT"
