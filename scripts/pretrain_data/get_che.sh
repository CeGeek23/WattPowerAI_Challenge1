#!/usr/bin/env bash
# Reproducible download of the Che et al. (2023) battery aging dataset.
#
#   Che, Yunhong; Zheng, Yusheng; Wu, Yue; Sui, Xin; Bharadwaj, Pallavi;
#   Stroe, Daniel-Ioan; Yang, Yalian; Hu, Xiaosong; Teodorescu, Remus.
#   "Data-efficient health diagnosis for the lithium-ion batteries" /
#   dataset for "Increasing generalization capability of battery health
#   estimation using continual learning approach",
#   Cell Reports Physical Science 4(12), 101743, 2023.
#   Mendeley Data, V9, doi:10.17632/n3b54nsw8m.9  -- licence CC BY 4.0
#
# Usage: ./get_che.sh [dest_dir]
# Default dest: <repo>/.dataset_cache/pretrain/che/  (git-ignored)
set -euo pipefail

# Default: <git repo root>/.dataset_cache/pretrain/che (git-ignored).
if [ $# -ge 1 ]; then
  DEST="$1"
else
  ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
  DEST="$ROOT/.dataset_cache/pretrain/che"
fi
DOI_ID="n3b54nsw8m"
VERSION=9
API="https://data.mendeley.com/public-api/datasets/${DOI_ID}/files?folder_id=root&version=${VERSION}"

mkdir -p "$DEST"

# 1. Fetch the file manifest (name, size, sha256, direct download URL).
curl -sSL "$API" -o "$DEST/_manifest.json"

# 2. Download every file listed, skipping any already present with the right size.
python3 - "$DEST" <<'PY'
import json, os, subprocess, sys
dest = sys.argv[1]
files = json.load(open(os.path.join(dest, "_manifest.json")))
for f in files:
    name, size = f["filename"], f["size"]
    url = f["content_details"]["download_url"]
    out = os.path.join(dest, name)
    if os.path.exists(out) and os.path.getsize(out) == size:
        print(f"[skip] {name} ({size} B)"); continue
    print(f"[get ] {name} ({size} B)")
    subprocess.run(["curl", "-sSL", "--retry", "3", "-o", out, url], check=True)
PY

# 3. Verify sha256 of each file against the manifest.
python3 - "$DEST" <<'PY'
import hashlib, json, os, sys
dest = sys.argv[1]
files = json.load(open(os.path.join(dest, "_manifest.json")))
ok = True
for f in files:
    p = os.path.join(dest, f["filename"])
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    exp = f["content_details"]["sha256_hash"]
    status = "OK " if h == exp else "BAD"
    ok &= (h == exp)
    print(f"[{status}] {f['filename']}  {h}")
sys.exit(0 if ok else 1)
PY

echo "Done -> $DEST"
