#!/bin/sh
# Wheeler et al. (2025) - "Aging study on twenty A123 18650 Graphite/LFP 1.1 Ah cells"
# DOI 10.57745/OLBXKT (Recherche Data Gouv / Dataverse) - licence CC BY 4.0
#
# Le dataset complet pese ~26 Go (20 archives .zip de time series brutes, ~1.0-1.7 Go chacune).
# Pour une courbe capacite-vs-cycle par cellule, SEUL extractedData.mat est necessaire (0.46 Mo) :
# c'est le resume MATLAB produit par les auteurs (capacite de charge C/3 a chaque check-up).
# Par defaut ce script ne telecharge QUE les 4 petits fichiers utiles (~0.5 Mo au total) ;
# FULL=1 ajoute les 20 archives brutes par cellule (~26 Go).
#
# Usage: sh get_wheeler.sh [dossier_de_destination]
#        defaut: <racine du depot>/dataset/wheeler
#        FULL=1 sh get_wheeler.sh   # + les 20 archives .zip brutes (~26 Go)
set -eu

RACINE="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${1:-$RACINE/dataset/wheeler}"
BASE="https://entrepot.recherche.data.gouv.fr/api/access/datafile"
DOI="doi:10.57745/OLBXKT"

mkdir -p "$DEST"

# id:nom_local  (ids stables de la version V2 du dataset)
for pair in \
    "611322:readme.txt" \
    "606807:extractedData.mat" \
    "607239:Matlab_BasicImportAndDisplay.m" \
    "606698:WLTPcycle.txt"
do
    id="${pair%%:*}"
    name="${pair#*:}"
    if [ -s "$DEST/$name" ]; then
        echo "skip   $name (deja present)"
        continue
    fi
    echo "get    $name (datafile id $id)"
    curl -fsSL "$BASE/$id" -o "$DEST/$name.part"
    mv "$DEST/$name.part" "$DEST/$name"
done

# Metadonnees completes du dataset (licence, citation, liste des 24 fichiers)
if [ ! -s "$DEST/dataverse_metadata.json" ]; then
    echo "get    dataverse_metadata.json"
    curl -fsSL "https://entrepot.recherche.data.gouv.fr/api/datasets/:persistentId/?persistentId=$DOI" \
        -o "$DEST/dataverse_metadata.json"
fi

# --- Archives brutes par cellule (courbes U/I/capacite point par point) -----
# 20 fichiers, ~1.0-1.7 Go chacun, ~26 Go au total. Non necessaires a
# scripts/pretrain.py (qui ne lit que extractedData.mat / pretrain_wheeler.csv) :
# telecharges seulement si FULL=1, pour la tracabilite complete du jeu cite.
if [ "${FULL:-0}" = "1" ]; then
    for pair in \
        "606806:Cell1a.zip" "606805:Cell1b.zip" "606804:Cell1c.zip" \
        "606803:Cell2a.zip" "606802:Cell2b.zip" "606801:Cell2c.zip" \
        "606800:Cell3a.zip" "606799:Cell3b.zip" \
        "606798:Cell4a.zip" "606797:Cell4b.zip" "606796:Cell4c.zip" \
        "606795:Cell5a.zip" "606794:Cell5b.zip" "606793:Cell5c.zip" \
        "606792:Cell6a.zip" "606791:Cell6b.zip" "606790:Cell6c.zip" \
        "606788:Cell7a.zip" "606787:Cell7b.zip" "606789:Cell7c.zip"
    do
        id="${pair%%:*}"
        name="${pair#*:}"
        if [ -s "$DEST/$name" ]; then
            echo "skip   $name (deja present)"
            continue
        fi
        echo "get    $name (datafile id $id)"
        curl -fSL "$BASE/$id" -o "$DEST/$name.part"
        mv "$DEST/$name.part" "$DEST/$name"
    done
fi

echo
echo "OK -> $DEST"
ls -l "$DEST"
