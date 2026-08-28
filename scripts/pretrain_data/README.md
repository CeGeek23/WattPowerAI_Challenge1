# Données publiques de pré-entraînement

Trois jeux, tous en **CC BY 4.0**. Chaîne complète par jeu, du téléchargement au
CSV que `scripts/pretrain.py` consomme. Les données brutes vont dans
`dataset/<jeu>/` (un sous-dossier par source, à côté de `dataset/target/` qui
contient les cellules cibles), jamais dans le dépôt.

| jeu | téléchargement | extraction | ce qu'on en tire |
| --- | --- | --- | --- |
| Che et al. 2023 | `get_che.sh` (260 Mo) | `build_che_csv.py` | **pente d'Arrhenius 2.58** → `my_model/pretrained.json` |
| Wheeler et al. 2025 | `get_wheeler.sh` (536 ko) | copie déclarée : `pretrain_wheeler.csv` | **dispersion cellule-à-cellule ~6.6 %** (vrais réplicats, pas pooling par (T,C) nominal) → nugget du GP |
| Catenaro & Onori 2021 | `get_catenaro.sh` (638 Mo, 334 fichiers) | copie déclarée : `pretrain_catenaro_capacite.csv` | capacité **réversible** vs T → justifie `a(T)` |

Dépendance hors livraison : `uv pip install h5py` (les `.mat` de Che sont en
v7.3, donc du HDF5). Rien de tout cela n'entre dans `requirements.txt` : le
pré-entraînement tourne hors ligne, seul `pretrained.json` est livré.

Régénérer les priors livrés :

```bash
./scripts/pretrain_data/get_che.sh
.venv/bin/python scripts/pretrain_data/build_che_csv.py \
    dataset/che /tmp/pretrain_che.csv
.venv/bin/python scripts/pretrain.py /tmp/pretrain_che.csv --chimie ""
```

`pretrain_wheeler.csv` est fourni comme copie dérivée déclarée (68 ko, 843
points) : l'extraction demande de croiser `extractedData.mat` avec la Table 2 de
l'article, qui documente les conditions d'essai absentes des données. Le script
de téléchargement reste fourni pour la traçabilité.

Catenaro ne contient **aucun vieillissement** (15 à 24 décharges de
caractérisation par cellule) : rejeté pour la loi de fade. Seul son résultat
dérivé est utilisé par `pretrain.py` — `pretrain_catenaro_capacite.csv` (15 ko,
capacité restituée par (T, C-rate)), qui étaye un chiffre du rapport.
`get_catenaro.sh` télécharge les 334 fichiers bruts (~638 Mo, un par cellule ×
température × C-rate, vérifiés sha256) dans `dataset/catenaro/` pour la
traçabilité complète du pipeline ; ce n'est pas nécessaire au jour le jour
puisque `pretrain.py` ne consomme que la copie dérivée. DOI dans le README
principal.
