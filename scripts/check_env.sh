#!/usr/bin/env bash
# =============================================================================
# Contrôle complet de l'environnement de soumission.
#
#   ./scripts/check_env.sh 3.11 3.12
#
# Pour chaque version de Python passée en argument :
#   1. crée un venv NEUF (uv) dans .env_check/<version>/,
#   2. installe uniquement requirements.txt (comme les organisateurs),
#   3. vérifie que tous les imports de my_model/ se résolvent,
#   4. exécute validate_submission.py (doit afficher SUBMISSION READY).
#
# Reproduit le point faible n°1 des soumissions : une dépendance présente
# dans l'environnement de dev mais absente de requirements.txt.
# =============================================================================
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CHECK_DIR="$REPO/.env_check"
FAIL=0

if [ $# -eq 0 ]; then
    echo "usage: $0 <python-version> [<python-version> ...]   ex: $0 3.11 3.12"
    exit 2
fi

command -v uv >/dev/null 2>&1 || { echo "ERREUR: uv introuvable"; exit 2; }

for PYVER in "$@"; do
    echo "=============================================================="
    echo " Python $PYVER"
    echo "=============================================================="
    VENV="$CHECK_DIR/$PYVER"
    rm -rf "$VENV"

    if ! uv venv --python "$PYVER" "$VENV" >/dev/null 2>&1; then
        echo "[SKIP] Python $PYVER indisponible via uv"
        continue
    fi
    PY="$VENV/bin/python"

    echo "[1/3] uv pip install -r requirements.txt (env neuf)"
    if ! uv pip install --python "$PY" -r "$REPO/requirements.txt" >/dev/null 2>&1; then
        echo "  [FAIL] installation de requirements.txt"
        FAIL=1
        continue
    fi
    echo "  [PASS]"

    echo "[2/3] résolution des imports de my_model/"
    if ! (cd "$REPO" && "$PY" - <<'EOF'
import ast
import importlib
import pathlib
import sys

repo = pathlib.Path(".").resolve()
mods = set()
for f in (repo / "my_model").glob("*.py"):
    tree = ast.parse(f.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
mods -= {"my_model", "framework"}
bad = []
for m in sorted(mods):
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append(f"{m}: {e}")
if bad:
    print("  [FAIL] imports non résolus depuis requirements.txt seul:")
    for b in bad:
        print("    -", b)
    sys.exit(1)
print(f"  [PASS] {len(mods)} imports résolus: {', '.join(sorted(mods))}")
EOF
    ); then
        FAIL=1
        continue
    fi

    echo "[3/3] validate_submission.py"
    if (cd "$REPO" && "$PY" validate_submission.py > "$CHECK_DIR/validate_$PYVER.log" 2>&1) \
       && grep -q "SUBMISSION READY" "$CHECK_DIR/validate_$PYVER.log"; then
        echo "  [PASS] SUBMISSION READY (log: .env_check/validate_$PYVER.log)"
    else
        echo "  [FAIL] voir .env_check/validate_$PYVER.log"
        FAIL=1
    fi
done

echo "=============================================================="
if [ $FAIL -eq 0 ]; then
    echo "check_env: TOUT EST PASSÉ"
else
    echo "check_env: DES CONTRÔLES ONT ÉCHOUÉ"
fi
exit $FAIL
