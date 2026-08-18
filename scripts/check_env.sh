#!/usr/bin/env bash
# Rejoue l'etape 2 des organisateurs avec uv : environnement NEUF construit
# depuis requirements.txt SEUL, puis validate_submission.py.
#
#   ./scripts/check_env.sh                # version de Python par defaut
#   ./scripts/check_env.sh 3.11 3.12      # plusieurs versions (uv les telecharge)
#
# Pourquoi : validate_submission.py utilise l'interpreteur depuis lequel on le
# lance. Un paquet installe globalement chez vous mais absent de
# requirements.txt passe ici et casse a la notation officielle. Ce script
# rend cette erreur impossible.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv introuvable. Installation :" >&2
  echo "  macOS/Linux : curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  echo "  Windows     : winget install --id astral-sh.uv   (puis rouvrir le terminal)" >&2
  exit 1
fi

VERSIONS=("$@")
[ ${#VERSIONS[@]} -eq 0 ] && VERSIONS=("")

fail=0
for v in "${VERSIONS[@]}"; do
  venv="$(mktemp -d)/venv"
  label="${v:-par defaut}"
  echo "=============================================================="
  echo " Python $label - environnement neuf depuis requirements.txt"
  echo "=============================================================="

  if [ -n "$v" ]; then uv venv --python "$v" "$venv" >/dev/null
  else uv venv "$venv" >/dev/null; fi

  # macOS/Linux -> bin/python ; Windows (Git Bash) -> Scripts/python.exe
  vpy="$venv/bin/python"; [ -x "$vpy" ] || vpy="$venv/Scripts/python.exe"

  if ! uv pip install --python "$vpy" -r requirements.txt --quiet; then
    echo "  ECHEC : requirements.txt ne s'installe pas sous Python $label"
    fail=$((fail + 1)); rm -rf "$(dirname "$venv")"; continue
  fi
  echo "  installe : $("$vpy" -c 'import sys; print(sys.version.split()[0])')"

  # tout import tiers de my_model/ doit exister dans CET environnement
  if ! "$vpy" - "$ROOT" <<'PY'
import ast, importlib.util, pathlib, sys
root = pathlib.Path(sys.argv[1]); pkg = root / "my_model"
local = {p.stem for p in pkg.rglob("*.py")} | {"my_model", "framework"}
mods = set()
for p in pkg.rglob("*.py"):
    tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"), str(p))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            mods.add(n.module.split(".")[0])
mods -= set(sys.stdlib_module_names) | local
PKG = {"sklearn": "scikit-learn", "cv2": "opencv-python", "PIL": "pillow",
       "yaml": "pyyaml", "skimage": "scikit-image", "Bio": "biopython"}
missing = sorted(m for m in mods if importlib.util.find_spec(m) is None)
print(f"  imports tiers de my_model/ : {', '.join(sorted(mods)) or 'aucun'}")
if missing:
    print("  MANQUANTS dans requirements.txt :")
    for m in missing:
        print(f"    {m}  -> ajouter '{PKG.get(m, m)}'")
    sys.exit(1)
PY
  then fail=$((fail + 1)); rm -rf "$(dirname "$venv")"; continue; fi

  if "$vpy" validate_submission.py; then echo "  -> Python $label : OK"
  else echo "  -> Python $label : ECHEC"; fail=$((fail + 1)); fi
  rm -rf "$(dirname "$venv")"
  echo
done

if [ "$fail" -eq 0 ]; then
  echo "TOUT EST VERT sur ${#VERSIONS[@]} version(s) de Python."
else
  echo "$fail version(s) en echec - voir ci-dessus."; exit 1
fi
