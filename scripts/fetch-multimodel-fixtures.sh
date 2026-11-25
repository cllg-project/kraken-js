#!/usr/bin/env bash
#
# Fetch the three multi-model OCR recognisers (base / new / spec) from the
# cllg-project/kraken-js-models release into tests/fixtures/multimodel/.
#
# Already-present files with the right checksum are kept, so this is cheap to rerun.
# The multi-model tests skip when the files are absent.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${MODELS_TAG:-v1.1.0}"
BASE_URL="https://github.com/cllg-project/kraken-js-models/releases/download/$TAG"
DEST="$ROOT/tests/fixtures/multimodel"
mkdir -p "$DEST"

# role  release-asset  sha256 (from SHA256SUMS of the release)
while read -r role asset sha; do
  out="$DEST/$role.js_mlmodel"
  if [ -f "$out" ] && echo "$sha  $out" | sha256sum --check --status; then
    echo "$role: up to date"
    continue
  fi
  echo "$role: downloading $asset …"
  curl -fSL --progress-bar -o "$out.part" "$BASE_URL/$asset"
  echo "$sha  $out.part" | sha256sum --check --status || {
    rm -f "$out.part"
    echo "error: checksum mismatch for $asset" >&2
    exit 1
  }
  mv "$out.part" "$out"
done <<'LIST'
base cllg3_rec.js_mlmodel a6690a8739141856c1c938a4d1e56162f28a8a45f60c9fcc08931d8c13a4871a
new  signs_r2.js_mlmodel bdfd38efb1b656bc6c5b3638ce8bbe99b21d48a90dbe9d16841c83e81ba03dd4
spec letters_masked.js_mlmodel 43370024058cd471083648668f6d4f0ae7c42bf9f08f5db4e3f4e2a0b207fb47
LIST
