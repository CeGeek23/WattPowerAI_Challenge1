# TechArena 2026 - Challenge 1 (SOH fade forecasting)

Prédire la courbe SOH% vs cycle pour n'importe quel point de fonctionnement
dans **25-55 °C x 0.5-1.0 C**, y compris des combinaisons jamais vues.
Le barème est relatif au modèle de référence : `model_example.py` = 1.0,
**plus bas est meilleur**.

Avant toute session de travail sur le modèle ou avant de packager, lancer
l'agent **techarena-guard** (`.claude/agents/techarena-guard.md`) : il audite
le dépôt contre les contraintes des organisateurs. La source de vérité reste
`submission_instructions_2026.md`.

## Interdits absolus

- **Ne jamais modifier** `run_model.py`, `framework/**`, `validate_submission.py`,
  `sample_data/**`, `sample_input.csv`. Les organisateurs rejouent la notation
  avec **leurs propres copies** : une modification locale ne change rien au
  score et casse seulement les tests locaux.
- Fichiers modifiables : `my_model/**`, `requirements.txt`, `README.md`.
  Tout fichier ajouté (helpers, poids pré-entraînés) va **dans `my_model/`**.
- Jamais de chemin absolu ni spécifique à une machine. Pour lire un fichier
  livré avec le modèle : `Path(__file__).parent / "..."`, jamais un chemin
  relatif au cwd.
- Aucun accès réseau pendant `fit()` : le pré-entraînement se fait hors-ligne,
  les poids sont livrés dans `my_model/`.
- Ne jamais commiter les données : `dataset/` est ignoré par git et se
  récupère via `scripts/get_dataset.sh` (assets de la release `dataset-v1`).

## Contrat d'API (imposé par le framework)

```python
class MyModel:
    def fit(self, cells): ...                       # une seule fois
    def predict_soh(self, temperature_degC, c_rate) # -> np.ndarray 1D
```

- `predict_soh` ne reçoit **que** `(T, C)`. Aucune donnée mesurée : les cellules
  d'évaluation n'en ont pas. Tout ce qui est appris doit finir sur `self.*`,
  car l'objet est picklé après `fit()` et rechargé dans un **autre processus**.
- Retourner 12000 valeurs (cycles 1..12000, indexation 1-based côté framework).
- Valeurs **finies** et dans **(0, 120]** - strictement > 0. Clipper, p. ex.
  `np.clip(soh, 0.5, 119.9)`.
- Rien d'impicklable sur `self` (lambdas, closures, handles, tenseurs CUDA :
  repasser les poids sur CPU).
- `fit()` doit tolérer **peu de cellules et des cycles de début de vie
  seulement** : chaque soumission est rejouée en budget réduit, et la
  robustesse y donne un bonus. Prévoir un repli explicite, jamais une
  exception.
- Les prédictions doivent **varier avec T et C** : un modèle qui sort la même
  courbe partout est un échec de modélisation, pas un bug du framework.

## Environnement

- Notation sur **Windows Server 2022**, 96 cœurs, 554 Go RAM, RTX PRO 94 Go.
  Attention au `multiprocessing` (spawn, pas fork).
- Limites officielles : **2 h** pour `train` sur le dataset complet, **30 min
  par cellule** pour `test`. Les 900 s / 300 s de `validate_submission.py` sont
  des garde-fous locaux, pas la limite du concours.
- Graines aléatoires fixées, résultat reproductible.
- Toute dépendance importée doit être dans `requirements.txt`, et la validation
  se fait dans un **venv neuf installé uniquement depuis ce fichier**.

## Conventions du dépôt

- Commits signés par l'humain seul : **aucun trailer d'attribution IA**
  (`Co-Authored-By: Claude`, « Generated with… »).
- `dataset/`, `model_state/`, `output*/`, `.val_out*/`, `.dataset_cache/`,
  `validation_report.txt` sont des artefacts : jamais commités.
