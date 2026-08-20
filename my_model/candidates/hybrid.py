# -*- coding: utf-8 -*-
"""Combinaisons : correction de résidu et ensemble pondéré."""
import numpy as np

from .base import Candidate, finalize


def _traj(model, T, C):
    return np.asarray(model.predict_soh(T, C), dtype=float)


class Residual(Candidate):
    """`base` prédit, `correcteur` apprend son résidu sur la grille de cycles."""

    def __init__(self, base, correcteur, nom=None, shrink=1.0):
        self.base = base
        self.correcteur = correcteur
        self.name = nom or f"{getattr(base, 'name', 'base')}+{getattr(correcteur, 'name', 'corr')}"
        self.shrink = shrink

    def fit(self, cells):
        self.base.fit(cells)
        residus = []
        for cell in cells:
            d = cell.soh.dropna(subset=["soh_percent"])
            n = np.asarray(d["cycle_number"], dtype=float)
            y = np.asarray(d["soh_percent"], dtype=float)
            if len(n) == 0:
                continue
            traj = _traj(self.base, cell.temperature_degC, cell.c_rate)
            pred = traj[np.clip(n.astype(int), 1, len(traj)) - 1]
            residus.append(_PseudoCell(cell, n, 100.0 + (y - pred)))
        self.correcteur.fit(residus)          # cible recentrée sur 100
        self.fitted_ = True
        return self

    def predict_soh(self, temperature_degC, c_rate):
        base = _traj(self.base, temperature_degC, c_rate)
        corr = _traj(self.correcteur, temperature_degC, c_rate) - 100.0
        return finalize(base + self.shrink * corr)


class _PseudoCell:
    """Cellule factice portant le résidu comme s'il s'agissait d'un SOH."""

    def __init__(self, cell, n, y):
        import pandas as pd
        self.cell_id = getattr(cell, "cell_id", "residu")
        self.temperature_degC = cell.temperature_degC
        self.c_rate = cell.c_rate
        self.soh = pd.DataFrame({"cycle_number": n, "soh_percent": y})

    def time_series(self):
        raise RuntimeError("les candidats n'utilisent pas la time series brute")


class Ensemble(Candidate):
    """Moyenne pondérée, poids ajustés par leave-one-condition-out interne."""

    def __init__(self, membres, nom=None, poids=None):
        self.membres = list(membres)
        self.name = nom or "+".join(getattr(m, "name", "?") for m in self.membres)
        self.poids = poids

    def fit(self, cells):
        cells = list(cells)
        if self.poids is None and len(cells) >= 3:
            erreurs = np.zeros((len(self.membres), len(cells)))
            for j, held in enumerate(cells):
                train = [c for c in cells if c is not held]
                d = held.soh.dropna(subset=["soh_percent"])
                n = np.asarray(d["cycle_number"], dtype=float)
                y = np.asarray(d["soh_percent"], dtype=float)
                for i, m in enumerate(self.membres):
                    try:
                        mm = _clone(m).fit(train)
                        p = _traj(mm, held.temperature_degC, held.c_rate)
                        e = p[np.clip(n.astype(int), 1, len(p)) - 1] - y
                        erreurs[i, j] = float(np.sqrt(np.mean(e ** 2)))
                    except Exception:
                        erreurs[i, j] = np.inf
            score = erreurs.mean(axis=1)
            score = np.where(np.isfinite(score), score, np.nanmax(score[np.isfinite(score)]) * 10)
            w = 1.0 / np.maximum(score, 1e-6) ** 2      # inverse de l'erreur quadratique
            self.poids_ = w / w.sum()
            self.scores_loco_ = score
        else:
            self.poids_ = (np.asarray(self.poids, dtype=float) if self.poids is not None
                           else np.ones(len(self.membres)) / len(self.membres))
            self.poids_ = self.poids_ / self.poids_.sum()
        for m in self.membres:
            m.fit(cells)
        self.fitted_ = True
        return self

    def predict_soh(self, temperature_degC, c_rate):
        trajs = np.stack([_traj(m, temperature_degC, c_rate) for m in self.membres])
        return finalize((self.poids_[:, None] * trajs).sum(axis=0))


def _clone(model):
    """Copie non ajustée d'un candidat (mêmes hyperparamètres)."""
    import copy
    neuf = copy.deepcopy(model)
    for attr in ("net_", "model_", "base_", "fitted_", "curves_", "poids_"):
        neuf.__dict__.pop(attr, None)
    return neuf
