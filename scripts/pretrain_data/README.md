# Données publiques de pré-entraînement

Trois jeux, tous en **CC BY 4.0**. Chaîne complète par jeu, du téléchargement au
CSV que `scripts/pretrain.py` consomme. Les données brutes vont dans
`.dataset_cache/pretrain/`, jamais dans le dépôt.

| jeu | téléchargement | extraction | ce qu'on en tire |
| --- | --- | --- | --- |
| Che et al. 2023 | `get_che.sh` (260 Mo) | `build_che_csv.py` | **pente d'Arrhenius 2.58** → `my_model/pretrained.json` |
| Wheeler et al. 2025 | `get_wheeler.sh` (536 ko) | copie déclarée : `pretrain_wheeler.csv` | **dispersion cellule-à-cellule 10 %** → nugget du GP |
| Catenaro & Onori 2021 | `get_catenaro.sh` (638 Mo) | `extract_catenaro.py` | capacité **réversible** vs T → justifie `a(T)` |

Dépendance hors livraison : `uv pip install h5py` (les `.mat` de Che sont en
v7.3, donc du HDF5). Rien de tout cela n'entre dans `requirements.txt` : le
pré-entraînement tourne hors ligne, seul `pretrained.json` est livré.

Régénérer les priors livrés :

```bash
./scripts/pretrain_data/get_che.sh
.venv/bin/python scripts/pretrain_data/build_che_csv.py \
    .dataset_cache/pretrain/che /tmp/pretrain_che.csv
.venv/bin/python scripts/pretrain.py /tmp/pretrain_che.csv --chimie ""
```

`pretrain_wheeler.csv` est fourni comme copie dérivée déclarée (68 ko, 843
points) : l'extraction demande de croiser `extractedData.mat` avec la Table 2 de
l'article, qui documente les conditions d'essai absentes des données. Le script
de téléchargement reste fourni pour la traçabilité.

Catenaro ne contient **aucun vieillissement** (15 à 24 décharges de
caractérisation par cellule) : son CSV a un schéma différent
(`capacite_Ah` par (T, C-rate), pas de colonne `cycle`) et n'alimente pas
`pretrain.py`. Il ne sert qu'à étayer un chiffre du rapport.
