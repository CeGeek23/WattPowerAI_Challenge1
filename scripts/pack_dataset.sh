#!/usr/bin/env bash
# Rebuild the release assets from a local dataset/ and (re)publish them.
# Only needed if the dataset changes - normal users just run get_dataset.sh.
#   ./scripts/pack_dataset.sh            # pack into .dataset_cache/
#   PUBLISH=1 ./scripts/pack_dataset.sh  # pack, then create/update the release
set -euo pipefail

TAG="${DATASET_TAG:-dataset-v1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/dataset"
OUT="$ROOT/.dataset_cache"
PDF="TechArena_2026_Topic_1_Challenge1.pdf"

[ -d "$SRC" ] || { echo "error: $SRC introuvable" >&2; exit 1; }
mkdir -p "$OUT"; cd "$SRC"

for d in */; do
  cell="${d%/}"
  echo "packing $cell"
  tar --exclude='.DS_Store' -czf "$OUT/$cell.tar.gz" "$cell"
done
[ -f "$PDF" ] && cp "$PDF" "$OUT/$PDF"

cd "$OUT"
if command -v sha256sum >/dev/null; then sha256sum *.tar.gz ${PDF:+"$PDF"} > SHA256SUMS
else shasum -a 256 *.tar.gz ${PDF:+"$PDF"} > SHA256SUMS; fi
du -ch *.tar.gz | tail -1

if [ -n "${PUBLISH:-}" ]; then
  gh release view "$TAG" >/dev/null 2>&1 \
    && gh release upload "$TAG" *.tar.gz "$PDF" SHA256SUMS --clobber \
    || gh release create "$TAG" *.tar.gz "$PDF" SHA256SUMS \
         --title "Dataset (6 cellules, 559 MB extraits)" \
         --notes "Assets de données pour ce challenge. Récupération : ./scripts/get_dataset.sh"
fi
