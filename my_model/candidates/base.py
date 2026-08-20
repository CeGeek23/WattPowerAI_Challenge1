# -*- coding: utf-8 -*-
"""Socle commun des candidats : extraction des courbes, features, garde-fous."""
import numpy as np

N_CYCLES = 12000
SOH_LO, SOH_HI = 0.5, 119.9
T_REF_K = 313.15                 # 40 degC, centre du domaine
C_REF = 0.75                     # centre du domaine
MIN_POINTS = 5
OUTLIER_WIN = 15
OUTLIER_TOL = 4.0
SEED = 0

CYCLES = np.arange(1, N_CYCLES + 1, dtype=float)


def extract(cells):
    """[(T, C, n, soh)] pour les cellules exploitables, labels nettoyés."""
    from scipy.ndimage import median_filter

    out = []
    for cell in cells:
        try:
            d = cell.soh.dropna(subset=["soh_percent"])
            n = np.asarray(d["cycle_number"], dtype=float)
            y = np.asarray(d["soh_percent"], dtype=float)
            T, C = float(cell.temperature_degC), float(cell.c_rate)
        except Exception:
            continue
        ok = np.isfinite(n) & np.isfinite(y) & (n >= 1) & (y > 20.0) & (y < 130.0)
        n, y = n[ok], y[ok]
        if len(n) < MIN_POINTS or not (np.isfinite(T) and np.isfinite(C) and C > 0):
            continue
        n, idx = np.unique(n, return_index=True)
        y = y[idx]
        if len(n) >= 3 * OUTLIER_WIN:
            smooth = median_filter(y, size=OUTLIER_WIN, mode="nearest")
            keep = np.abs(y - smooth) <= OUTLIER_TOL
            if keep.sum() >= MIN_POINTS:
                n, y = n[keep], y[keep]
        out.append((T, C, n, y))
    return out


def soh_ref(curves):
    """SOH de départ commun : les cellules démarrent au-dessus de 100%."""
    if not curves:
        return 102.5
    return float(np.clip(np.mean([np.percentile(y, 99.5) for *_, y in curves]), 99.0, 106.0))


def cond_features(T, C):
    """Conditions d'essai, centrées : Arrhenius et log C-rate."""
    T = np.asarray(T, dtype=float)
    C = np.asarray(C, dtype=float)
    x = 1000.0 / (T + 273.15) - 1000.0 / T_REF_K
    lc = np.log(np.maximum(C, 1e-3) / C_REF)
    return np.column_stack([x, lc])


def cycle_features(n):
    """Transformations du cycle où la dégradation est ~linéaire (un arbre n'extrapole pas)."""
    n = np.maximum(np.asarray(n, dtype=float), 1.0)
    return np.column_stack([n / 1000.0, np.sqrt(n) / 30.0, n ** 0.8 / 300.0,
                            np.log1p(n) / 10.0])


def design_matrix(T, C, n):
    """Features (conditions + cycle + interactions) pour un modèle tabulaire."""
    n = np.atleast_1d(np.asarray(n, dtype=float))
    cond = cond_features(np.full(len(n), T), np.full(len(n), C))
    cyc = cycle_features(n)
    inter = np.column_stack([cond[:, 0] * cyc[:, i] for i in range(cyc.shape[1])]
                            + [cond[:, 1] * cyc[:, i] for i in range(cyc.shape[1])])
    return np.column_stack([cond, cyc, inter])


def stack_training(curves):
    """Empile toutes les cellules en (X, y, groupes) pour un modèle tabulaire."""
    X, y, g = [], [], []
    for k, (T, C, n, s) in enumerate(curves):
        X.append(design_matrix(T, C, n))
        y.append(s)
        g.append(np.full(len(n), k))
    return np.vstack(X), np.concatenate(y), np.concatenate(g)


def sample_weights(curves):
    """Chaque cellule pèse pareil, quelle que soit la longueur de son essai."""
    return np.concatenate([np.full(len(n), 1.0 / len(n)) for *_, n, _ in curves])


def finalize(soh):
    """Contraintes du framework : fini, décroissant, dans (0, 120], 12000 valeurs."""
    soh = np.asarray(soh, dtype=float).ravel()
    if len(soh) != N_CYCLES:
        soh = np.interp(CYCLES, np.linspace(1, N_CYCLES, len(soh)), soh)
    soh = np.where(np.isfinite(soh), soh, SOH_LO)
    soh = np.minimum.accumulate(soh)
    return np.clip(soh, SOH_LO, SOH_HI)


def soft_floor(loss, cap, sharp=8.0):
    """Sature la perte de SOH au lieu de la laisser diverger au-delà des données."""
    cap = max(float(cap), 1.0)
    with np.errstate(over="ignore"):
        return loss / (1.0 + (loss / cap) ** sharp) ** (1.0 / sharp)


class Candidate:
    """Contrat commun. Un candidat implémente _fit() et _trajectory()."""

    name = "candidate"
    needs = ()                    # paquets requis, pour message d'erreur clair

    def fit(self, cells):
        self.curves_ = extract(cells)
        self.a_ = soh_ref(self.curves_)
        self.fitted_ = False
        if self.curves_:
            self._fit(self.curves_)
            self.fitted_ = True
        return self

    def predict_soh(self, temperature_degC, c_rate):
        try:
            T = float(temperature_degC)
            C = float(c_rate)
        except (TypeError, ValueError):
            T, C = 40.0, C_REF
        if not np.isfinite(T):
            T = 40.0
        if not np.isfinite(C) or C <= 0:
            C = C_REF
        if not getattr(self, "fitted_", False):
            return finalize(np.full(N_CYCLES, self.a_ if hasattr(self, "a_") else 102.5))
        try:
            return finalize(self._trajectory(T, C))
        except Exception:
            return finalize(np.full(N_CYCLES, getattr(self, "a_", 102.5)))

    def _fit(self, curves):
        raise NotImplementedError

    def _trajectory(self, T, C):
        raise NotImplementedError
