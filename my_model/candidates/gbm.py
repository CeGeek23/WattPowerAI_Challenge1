# -*- coding: utf-8 -*-
"""Boosting d'arbres. Backend : xgboost, sinon lightgbm, sinon sklearn."""
import numpy as np

from .base import CYCLES, SEED, Candidate, design_matrix, sample_weights, stack_training

# indices des colonnes de design_matrix qui croissent avec le cycle : le SOH doit
# décroître quand elles augmentent
_CYCLE_COLS = (2, 3, 4, 5)
_N_FEATURES = 14


def _monotone_vector():
    v = [0] * _N_FEATURES
    for i in _CYCLE_COLS:
        v[i] = -1
    return v


def available_backend():
    """Nom du backend de boosting réellement utilisable ici."""
    for name, mod in (("xgboost", "xgboost"), ("lightgbm", "lightgbm")):
        try:
            __import__(mod)
            return name
        except Exception:
            continue
    return "sklearn"


def _make_model(backend, n_estimators, learning_rate, max_depth, monotone):
    """Un régresseur boosting entraînable, quel que soit le backend disponible."""
    if backend == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
            n_jobs=1, tree_method="hist",
            monotone_constraints=tuple(_monotone_vector()) if monotone else None)
    if backend == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
            n_jobs=1, verbose=-1,
            monotone_constraints=_monotone_vector() if monotone else None)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth if max_depth > 0 else None, l2_regularization=1.0,
        early_stopping=False, random_state=SEED,
        monotonic_cst=_monotone_vector() if monotone else None)


class GbmDirect(Candidate):
    """Boosting brut : (Arrhenius, ln C, features de cycle) -> SOH."""

    name = "gbm"

    def __init__(self, backend=None, n_estimators=400, learning_rate=0.05,
                 max_depth=4, monotone=True):
        self.backend = backend or available_backend()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.monotone = monotone

    def _fit(self, curves):
        X, y, _ = stack_training(curves)
        w = sample_weights(curves)
        self.model_ = _make_model(self.backend, self.n_estimators, self.learning_rate,
                                  self.max_depth, self.monotone)
        self.model_.fit(X, y, sample_weight=w)

    def _trajectory(self, T, C):
        return self.model_.predict(design_matrix(T, C, CYCLES))


class GbmResidual(Candidate):
    """Modèle physique + boosting sur son résidu : l'hybride physique/ML."""

    name = "master+gbm"

    def __init__(self, backend=None, n_estimators=300, learning_rate=0.03,
                 max_depth=3, shrink=1.0):
        self.backend = backend or available_backend()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.shrink = shrink          # poids de la correction, 0 = physique seule

    def fit(self, cells):
        from ..model_template import MyModel
        self.base_ = MyModel().fit(cells)
        return super().fit(cells)

    def _fit(self, curves):
        X, resid, w = [], [], []
        for T, C, n, y in curves:
            traj = np.asarray(self.base_.predict_soh(T, C), dtype=float)
            pred = traj[np.clip(n.astype(int), 1, len(traj)) - 1]
            X.append(design_matrix(T, C, n))
            resid.append(y - pred)
            w.append(np.full(len(n), 1.0 / len(n)))
        self.model_ = _make_model(self.backend, self.n_estimators, self.learning_rate,
                                  self.max_depth, monotone=False)
        self.model_.fit(np.vstack(X), np.concatenate(resid),
                        sample_weight=np.concatenate(w))

    def _trajectory(self, T, C):
        base = np.asarray(self.base_.predict_soh(T, C), dtype=float)
        return base + self.shrink * self.model_.predict(design_matrix(T, C, CYCLES))
