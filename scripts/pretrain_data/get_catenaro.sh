#!/usr/bin/env bash
# Reproducible download of the Catenaro & Onori (2021) battery discharge dataset.
#
#   Catenaro, Edoardo; Onori, Simona.
#   "Experimental data of three lithium-ion batteries under galvanostatic
#   discharge tests at different C-rates and operating temperatures",
#   V2, Mendeley Data, doi:10.17632/kxsbr4x3j2.2 -- licence CC BY 4.0.
#   Paper: Data in Brief 35, 106894 (2021).
#
# 334 files, ~638 MB total (mostly raw .mat/.m per chemistry x temperature x
# C-rate combination). No ageing data in this dataset (15-24 characterisation
# discharges per cell) -- only the derived reversible-capacity-vs-T table is
# actually used by scripts/pretrain.py; this script exists for full
# traceability/reproducibility of that derived table, not because the raw
# files are needed day to day.
#
# Usage: ./get_catenaro.sh [dest_dir]
# Default dest: <repo>/dataset/catenaro/  (git-ignored)
set -euo pipefail

if [ $# -ge 1 ]; then
  DEST="$1"
else
  ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
  DEST="$ROOT/dataset/catenaro"
fi
DOI_ID="kxsbr4x3j2"
API="https://data.mendeley.com/public-api/datasets/${DOI_ID}"

mkdir -p "$DEST"

# 1. Fetch the dataset record: unlike Che's dataset (flat, root-level files),
#    Catenaro's files sit in nested per-condition folders, so the single
#    dataset-info endpoint (not /files?folder_id=root) already returns the
#    full file list with per-file download URLs and sha256 hashes.
curl -sSL "$API" -o "$DEST/_manifest.json"

# 2. Download every file listed, skipping any already present with the right size.
python3 - "$DEST" <<'PY'
import json, os, subprocess, sys
dest = sys.argv[1]
d = json.load(open(os.path.join(dest, "_manifest.json")))
files = d["files"]
print(f"{len(files)} files, {sum(f['size'] for f in files) / 1e6:.1f} MB total")
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
d = json.load(open(os.path.join(dest, "_manifest.json")))
ok = True
for f in d["files"]:
    p = os.path.join(dest, f["filename"])
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    exp = f["content_details"]["sha256_hash"]
    status = "OK " if h == exp else "BAD"
    ok &= (h == exp)
    print(f"[{status}] {f['filename']}  {h}")
sys.exit(0 if ok else 1)
PY

echo "Done -> $DEST"
