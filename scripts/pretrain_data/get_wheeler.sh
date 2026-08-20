#!/bin/sh
# Wheeler et al. (2025) - "Aging study on twenty A123 18650 Graphite/LFP 1.1 Ah cells"
# DOI 10.57745/OLBXKT (Recherche Data Gouv / Dataverse) - licence CC BY 4.0
#
# Le dataset complet pese ~26 Go (20 archives .zip de time series brutes, ~1.0-1.7 Go chacune).
# Pour une courbe capacite-vs-cycle par cellule, SEUL extractedData.mat est necessaire (0.46 Mo) :
# c'est le resume MATLAB produit par les auteurs (capacite de charge C/3 a chaque check-up).
# Ce script ne telecharge QUE les 4 petits fichiers utiles (~0.5 Mo au total).
#
# Usage: sh get_wheeler.sh [dossier_de_destination]
#        defaut: <racine du depot>/.dataset_cache/pretrain/wheeler
set -eu

DEST="${1:-$(cd "$(dirname "$0")" && pwd)/.dataset_cache/pretrain/wheeler}"
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

echo
echo "OK -> $DEST"
ls -l "$DEST"

# --- Pour aller plus loin (NON telecharge par defaut : ~26 Go) --------------
# Archives brutes par cellule (courbes U/I/capacite point par point) :
#   Cell1a 606806  Cell1b 606805  Cell1c 606804
#   Cell2a 606803  Cell2b 606802  Cell2c 606801
#   Cell3a 606800  Cell3b 606799
#   Cell4a 606798  Cell4b 606797  Cell4c 606796
#   Cell5a 606795  Cell5b 606794  Cell5c 606793
#   Cell6a 606792  Cell6b 606791  Cell6c 606790
#   Cell7a 606788  Cell7b 606787  Cell7c 606789
# Exemple : curl -fL "$BASE/606806" -o "$DEST/Cell1a.zip"
