#!/usr/bin/env bash
# Telechargement reproductible du jeu Catenaro & Onori (Mendeley Data).
#
#   Catenaro, Edoardo; Onori, Simona (2021),
#   "Experimental data of three lithium-ion batteries under galvanostatic
#    discharge tests at different C-rates and operating temperatures",
#   Mendeley Data, V2, doi: 10.17632/kxsbr4x3j2.2   -- licence CC BY 4.0
#   (jeu de donnees de l'article Data in Brief 35 (2021) 106894)
#
# Un doublon strict existe : 10.17632/68jkb7x6t4.3 (memes 334 fichiers,
# memes sha256). On prend celui reference par l'article.
#
# Usage :  ./get_catenaro.sh [dossier_destination]
# Defaut :  <racine_depot>/.dataset_cache/pretrain/catenaro   (ignore par git)

set -euo pipefail

DATASET_ID="${CATENARO_ID:-kxsbr4x3j2}"
VERSION="${CATENARO_VERSION:-2}"
# 1er argument, sinon $CATENARO_DEST, sinon <cwd>/.dataset_cache/pretrain/catenaro
DEST="${1:-${CATENARO_DEST:-$PWD/.dataset_cache/pretrain/catenaro}}"

mkdir -p "$DEST"
META="$DEST/_mendeley_metadata.json"

echo "[1/3] metadonnees Mendeley ($DATASET_ID v$VERSION) -> $META"
# NB : l'endpoint /files?folder_id=root renvoie [] (arborescence en sous-dossiers)
# et ?version=N supprime la cle "files" ; l'endpoint nu donne la liste a plat
# de la version publiee courante, avec les URL directes. Les 334 noms sont uniques.
curl -sfL "https://data.mendeley.com/public-api/datasets/${DATASET_ID}" -o "$META"

echo "[2/3] construction de la liste (nom, url, sha256)"
python3 - "$META" "$DEST/_files.tsv" "$VERSION" <<'PY'
import json, sys
meta, out, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(meta))
assert d["version"] == want, f'version publiee {d["version"]} != {want} attendue'
assert d["data_licence"]["short_name"] == "CC BY 4.0", d["data_licence"]["short_name"]
with open(out, "w") as f:
    for fl in sorted(d["files"], key=lambda x: x["filename"]):
        cd = fl["content_details"]
        f.write(f"{fl['filename']}\t{cd['download_url']}\t{cd['sha256_hash']}\t{fl['size']}\n")
print(f"  {len(d['files'])} fichiers, {sum(f['size'] for f in d['files'])/1e6:.1f} Mo")
PY

echo "[3/3] telechargement (reprise : fichier deja present et de bonne taille = saute)"
n=0
while IFS=$'\t' read -r name url sha size; do
  n=$((n+1))
  target="$DEST/$name"
  if [ -f "$target" ] && [ "$(wc -c < "$target" | tr -d ' ')" = "$size" ]; then
    continue
  fi
  printf '  [%3d] %s\n' "$n" "$name"
  curl -sfL --retry 3 --retry-delay 2 "$url" -o "$target"
done < "$DEST/_files.tsv"

echo "verification sha256"
python3 - "$DEST" <<'PY'
import hashlib, os, sys
dest = sys.argv[1]
bad = []
with open(os.path.join(dest, "_files.tsv")) as f:
    for line in f:
        name, url, sha, size = line.rstrip("\n").split("\t")
        p = os.path.join(dest, name)
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != sha:
            bad.append(name)
print("OK" if not bad else f"ECHEC sha256 : {bad}")
sys.exit(1 if bad else 0)
PY

echo "termine -> $DEST"
