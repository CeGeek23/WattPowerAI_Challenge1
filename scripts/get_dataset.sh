#!/usr/bin/env bash
# Récupère le dataset de mesures depuis la release GitHub de ce repo.
# Les données ne sont PAS dans git (107 MB d'archives, 559 MB extraits) :
# elles vivent en assets de la release "$TAG".
#
#   ./scripts/get_dataset.sh                                # toutes les cellules + le PDF
#   ./scripts/get_dataset.sh 102Ah_45degC_1C_cell1 [...]    # seulement celles-là
#   FORCE=1 ./scripts/get_dataset.sh                        # réextraire même si présent
#
# Accès (repo privé) - au choix :
#   a) GitHub CLI : brew install gh (ou winget/apt) puis  gh auth login
#   b) sans rien installer :  export GITHUB_TOKEN=ghp_...  (token classic, scope "repo",
#      créé sur https://github.com/settings/tokens) - curl + python3 suffisent.
# Dans les deux cas il faut être collaborateur du repo.
set -euo pipefail

TAG="${DATASET_TAG:-dataset-v1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/dataset"
CACHE="$ROOT/.dataset_cache"

CELLS=(
  102Ah_25degC_0p5C_cell3
  102Ah_25degC_1C_cell3
  102Ah_35degC_1C_cell1
  102Ah_45degC_0p5C_cell3
  102Ah_45degC_1C_cell1
  102Ah_55degC_1C_cell3
)
PDF="TechArena_2026_Topic_1_Challenge1.pdf"

# --- repo : depuis la remote origin du clone, override possible via DATASET_REPO ---
REPO="${DATASET_REPO:-}"
if [ -z "$REPO" ]; then
  origin=$(git -C "$ROOT" remote get-url origin 2>/dev/null || true)
  [ -n "$origin" ] || { echo "error: pas de remote 'origin' - lance le script depuis un clone du repo, ou export DATASET_REPO=owner/name" >&2; exit 1; }
  REPO=$(printf '%s' "$origin" | sed -E 's#^(git@[^:]+:|https?://[^/]+/)##; s#\.git$##')
fi

# --- backend : gh si dispo et authentifié, sinon curl + GITHUB_TOKEN ---
BACKEND=""
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  BACKEND=gh
elif [ -n "${GITHUB_TOKEN:-}" ] && command -v curl >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  BACKEND=curl
else
  cat >&2 <<'MSG'
error: aucun accès GitHub utilisable. Deux options :
  1) GitHub CLI :  brew install gh   (ou winget install GitHub.cli / apt install gh)
                   gh auth login
  2) Token      :  export GITHUB_TOKEN=ghp_...   (classic, scope "repo")
                   https://github.com/settings/tokens
Il faut aussi être collaborateur de ce repo privé (demande une invitation à l'owner).
MSG
  exit 1
fi

api() { # $1 = chemin d'API -> stdout JSON
  curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" \
       -H "Accept: application/vnd.github+json" "https://api.github.com$1"
}

download() { # $1 = nom d'asset -> $CACHE/$1
  [ -f "$CACHE/$1" ] && return 0
  echo "  téléchargement $1"
  if [ "$BACKEND" = gh ]; then
    gh release download "$TAG" --repo "$REPO" --pattern "$1" --dir "$CACHE" --clobber
  else
    local id
    id=$(api "/repos/$REPO/releases/tags/$TAG" | python3 -c \
      'import json,sys; a=json.load(sys.stdin)["assets"]; n=sys.argv[1]; print(next((x["id"] for x in a if x["name"]==n), ""))' "$1")
    [ -n "$id" ] || { echo "  ! asset $1 introuvable dans la release $TAG" >&2; return 1; }
    curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/octet-stream" \
         -o "$CACHE/$1.part" "https://api.github.com/repos/$REPO/releases/assets/$id"
    mv "$CACHE/$1.part" "$CACHE/$1"
  fi
}

verify() { # $1 = nom d'asset, comparé au SHA256SUMS de la release
  local want_sum have_sum
  want_sum=$(awk -v f="$1" '$2 == f {print $1}' "$CACHE/SHA256SUMS")
  [ -n "$want_sum" ] || { echo "  ! $1 absent de SHA256SUMS" >&2; return 1; }
  if command -v sha256sum >/dev/null 2>&1; then have_sum=$(sha256sum "$CACHE/$1" | cut -d' ' -f1)
  else have_sum=$(shasum -a 256 "$CACHE/$1" | cut -d' ' -f1); fi
  [ "$want_sum" = "$have_sum" ]
}

fetch() { # download + vérif, un seul retry si le cache est corrompu
  download "$1"
  verify "$1" && return 0
  echo "  checksum invalide, nouveau téléchargement"
  rm -f "$CACHE/$1"; download "$1"
  verify "$1" || { echo "  ! $1 toujours corrompu après retéléchargement" >&2; return 1; }
}

want=("$@")
[ ${#want[@]} -eq 0 ] && want=("${CELLS[@]}")

mkdir -p "$CACHE" "$DEST"
echo "repo $REPO, release $TAG (accès : $BACKEND)"
rm -f "$CACHE/SHA256SUMS"; download SHA256SUMS

for cell in "${want[@]}"; do
  if [ -d "$DEST/$cell" ] && [ -z "${FORCE:-}" ]; then
    echo "$cell : déjà présent (FORCE=1 pour réextraire)"
    continue
  fi
  echo "$cell"
  fetch "$cell.tar.gz"
  rm -rf "$DEST/$cell"
  tar -xzf "$CACHE/$cell.tar.gz" -C "$DEST"
  echo "  -> dataset/$cell ($(find "$DEST/$cell" -name '*.csv' | wc -l | tr -d ' ') CSV)"
done

if [ ! -f "$DEST/$PDF" ]; then
  fetch "$PDF"; cp "$CACHE/$PDF" "$DEST/$PDF"
fi

echo
echo "OK - dataset/ prêt ($(du -sh "$DEST" | cut -f1)). Archives en cache dans .dataset_cache/ (supprimable)."
echo "Test :  python run_model.py --model train --input dataset --output-dir output"
