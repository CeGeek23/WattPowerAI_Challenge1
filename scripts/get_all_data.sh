#!/usr/bin/env bash
# Recupere tout ce qu'il faut pour reproduire ce projet depuis un clone frais.
#
#   ./scripts/get_all_data.sh          # essentiel : cellules cibles + sources
#                                       # publiques utilisees par le pipeline (~830 Mo)
#   FULL=1 ./scripts/get_all_data.sh   # + archives brutes completes Wheeler/Catenaro,
#                                       # tracabilite seulement, pas necessaires pour
#                                       # entrainer/reproduire le modele (~26 Go de plus)
#
# Rien de tout ca ne vit dans git (dataset/ est gitignore, cf. CLAUDE.md) : chaque
# etape retelecharge depuis la source d'origine (release GitHub "dataset-v1" pour
# les cellules cibles, Recherche Data Gouv / Mendeley Data pour le pre-entrainement
# public) plutot que depuis une copie qu'on hebergerait nous-memes -- git LFS n'a
# rien apporte ici, tout est deja reproductible depuis les sources.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== 1/4 cellules cibles -> dataset/target/ (~559 Mo) ==="
"$ROOT/scripts/get_dataset.sh"

echo
echo "=== 2/4 Wheeler et al. 2025 -> dataset/wheeler/ (~0.5 Mo${FULL:+, +26 Go (FULL=1)}) ==="
"$ROOT/scripts/pretrain_data/get_wheeler.sh"

echo
echo "=== 3/4 Che et al. 2023 -> dataset/che/ (~266 Mo) ==="
"$ROOT/scripts/pretrain_data/get_che.sh"

echo
if [ "${FULL:-0}" = "1" ]; then
    echo "=== 4/4 Catenaro & Onori 2021, jeu complet -> dataset/catenaro/ (~638 Mo) ==="
    "$ROOT/scripts/pretrain_data/get_catenaro.sh"
else
    echo "=== 4/4 Catenaro & Onori 2021 : ignore ==="
    echo "La copie derivee utilisee par pretrain.py est deja dans git"
    echo "(scripts/pretrain_data/pretrain_catenaro_capacite.csv) -- rien a faire."
    echo "FULL=1 telecharge aussi les 334 fichiers bruts (~638 Mo, tracabilite seulement)."
fi

echo
echo "OK. Pour reentrainer le modele livre :"
echo "  python run_model.py --model train --input dataset/target --output-dir output"
