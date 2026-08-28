# -*- coding: utf-8 -*-
"""Courbe maîtresse : une forme de fade partagée, une échelle de temps par point (T, C).

    SOH(n) = a - L(n / tau(T, C)),  L(u) = A(1-exp(-u^p)) + B u^q
    ln tau = tendance Arrhenius + résidu GP

Justification et validation : README.md et analysis/00_exploration.ipynb.
"""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares

N_CYCLES = 12000                 # trajectory length returned by predict_soh
SOH_LO, SOH_HI = 0.5, 119.9      # framework requires values in (0, 120]
T_REF_K = 313.15                 # 40 degC, centre of the 25-55 degC domain
C_REF = 0.75                     # centre of the 0.5-1.0 C domain

# --- physical priors -------------------------------------------------------
# Défauts physiques, écrasés par my_model/pretrained.json quand il est livré :
# ce fichier porte la forme et la pente d'Arrhenius apprises hors ligne sur les
# jeux LFP publics (scripts/pretrain.py). Aucun accès réseau ici.
TH_PRIOR = np.array([15.0, 0.65, 0.35, 2.70])   # A, p, B, q of L(u)
TH_SCALE = np.array([10.0, 0.30, 0.30, 1.00])   # prior width, same order
TH_LO = np.array([2.0, 0.25, 0.0, 1.0])
TH_HI = np.array([40.0, 1.50, 20.0, 5.0])
TH_PRIOR_W = 0.45                # weight of the prior, in "equivalent cells"

LN_TAU_PRIOR = np.log(1000.0)    # ~1000 cycles per unit of reduced time at 40 degC
SLOPE_PRIOR = 2.8                # d ln(tau) / d (1000/T_K), i.e. Ea ~ 23 kJ/mol
C_EXP_PRIOR = -0.15              # tau ~ C**-0.15: higher current ages slightly faster
# tested: Wheeler et al. 2025's fitted C-exponent (+0.15, LFP, 6x wider C-range
# than target's 0.5-1.0C) helps full LOCO (2.463->2.278) but hurts loco-800/
# loco-400 monotonically (1.906->2.416, 4.957->5.620) - rejected.
CURV_PRIOR = 0.0                 # d2 ln(tau) / d(1000/T_K)2
PRETRAINED = {}


def _charge_pretrained():
    """Charge les priors pré-entraînés livrés à côté de ce fichier, s'ils existent."""
    global TH_PRIOR, SLOPE_PRIOR, C_EXP_PRIOR, CURV_PRIOR, PRETRAINED
    chemin = Path(__file__).parent / "pretrained.json"
    try:
        with open(chemin) as fh:
            d = json.load(fh)
    except Exception:
        return                    # pas de poids livrés : priors par défaut
    # chaque clé est indépendante : le pré-entraînement n'exporte que ce qu'il juge
    # transférable, et le modèle reste fonctionnel avec les autres priors par défaut.
    forme = np.asarray(d.get("shape", []), dtype=float)
    if forme.shape == (4,) and np.all(np.isfinite(forme)):
        TH_PRIOR = np.clip(forme, TH_LO, TH_HI)
    for cle, nom in (("slope", "SLOPE_PRIOR"), ("c_exp", "C_EXP_PRIOR"), ("curv", "CURV_PRIOR")):
        v = d.get(cle)
        if isinstance(v, (int, float)) and np.isfinite(v):
            globals()[nom] = float(v)
    PRETRAINED = d.get("meta", {})


_charge_pretrained()
RIDGE = np.array([1e-6, 0.02, 0.30, 0.005])  # [intercept, slope, C exponent, curvature]

GP_SIGMA_F = 0.30                # amplitude of condition-specific departures
# Dispersion cellule-a-cellule de ln(tau) : ~6.6% dans les vrais groupes de
# replicats Wheeler (meme protocole, Cell1-7) ; pooler par (T,C) nominal sans
# distinguer les protocoles gonfle ce chiffre a ~16%. 3% optimal en validation
# croisee, qui ne peut pas le voir (zero replicat cote cible). Valeur
# intermediaire, cout mesure 0.013.
GP_SIGMA_N = 0.06                # cell-to-cell scatter of ln tau
GP_ELL_X = 0.12                  # correlation length in 1000/T_K (~12 degC)
GP_ELL_C = 0.80                  # correlation length in ln C (~ the whole range)
GP_CAP = 0.30                    # a condition may not depart from the trend by >35%

# Le SOH de depart monte legerement avec T : capacite cinetique reversible, pas du
# vieillissement (Catenaro & Onori 2021 : +1.7 point de 25 a 35 degC sur LFP ; ici
# +0.58 point sur 25-55 degC). Pente fortement contrainte.
A_RIDGE = 0.03                   # shrinkage de la pente de a(T) : garde ~70% de la pente
DEPTH_REF = 5.0                  # SOH points a cell must lose for a solid tau estimate
# past the knee: saturate instead of free-falling onto the clipping floor
SOH_FLOOR = 20.0                 # asymptote of the trajectory
CAP_SHARP = 8.0                  # sharpness of that saturation

# borne haute du temps caracteristique: empeche un "cette cellule ne vieillit jamais"
TAU_LO, TAU_HI = 20.0, 20000.0
MIN_POINTS = 5                   # minimum labelled cycles for a cell to be usable
OUTLIER_WIN = 15                 # median filter width used to drop label outliers
OUTLIER_TOL = 4.0                # SOH points away from the local median


def fade_shape(u, theta, cap=None):
    """Shared degradation shape: SOH loss, in points, versus reduced time u.

    `cap` saturates the loss smoothly (it is untouched while far below the cap).
    Without it the knee term would send the trajectory below zero well before
    cycle 12000 and the whole tail would sit on the clipping floor.
    """
    A, p, B, q = theta
    u = np.maximum(np.asarray(u, dtype=float), 0.0)
    loss = A * (1.0 - np.exp(-np.clip(u ** p, 0.0, 60.0))) + B * u ** q
    if cap is None:
        return loss
    cap = max(float(cap), 1.0)
    with np.errstate(over="ignore"):
        return loss / (1.0 + (loss / cap) ** CAP_SHARP) ** (1.0 / CAP_SHARP)


def _features(T, C):
    """Arrhenius abscissa and log C-rate, both centred on the domain."""
    x = 1000.0 / (np.asarray(T, dtype=float) + 273.15) - 1000.0 / T_REF_K
    lc = np.log(np.maximum(np.asarray(C, dtype=float), 1e-3) / C_REF)
    return x, lc


class MyModel:

    def __init__(self):
        self.theta_ = TH_PRIOR.copy()
        self.a_ = 102.5
        self.w_ = np.array([LN_TAU_PRIOR, SLOPE_PRIOR, C_EXP_PRIOR, CURV_PRIOR])
        self.free_ = np.array([True, True, True, True])
        self.wa_ = None
        self.gp_z_ = np.zeros((0, 2))
        self.gp_alpha_ = np.zeros(0)
        self.n_cells_ = 0
        self.fade_depth_ = 0.0

    # ------------------------------------------------------------------ fit
    def fit(self, cells):
        """Learn the shared shape and the tau(T, C) law from the training cells."""
        curves = [c for c in (self._clean(cell) for cell in cells) if c is not None]
        self.n_cells_ = len(curves)
        if not curves:                       # nothing usable: keep the priors
            return self
        try:
            theta, a_i, tau_i = self._fit_shape(curves)
        except Exception:                    # never let fit() raise
            theta, a_i, tau_i = self._fallback_shape(curves)
        self.theta_ = theta
        self.a_ = float(np.clip(np.mean(a_i), 99.0, 106.0))
        self._fit_a([c[0] for c in curves], [c[1] for c in curves], a_i)
        # how far down the fade each cell was actually followed: a cell observed
        depth = np.array([max(float(a_i[i]) - float(curves[i][3].min()), 0.0)
                          for i in range(len(curves))])
        self.fade_depth_ = float(depth.max())
        self._fit_law([c[0] for c in curves], [c[1] for c in curves], np.log(tau_i),
                      np.clip(depth / DEPTH_REF, 0.15, 1.0))
        return self

    def _fit_a(self, T, C, a_i):
        """SOH de depart en fonction de T, pente ridgee vers 0."""
        x, _ = _features(T, C)
        X = np.column_stack([np.ones_like(x), x])
        lam = np.diag([1e-6, A_RIDGE])
        try:
            w = np.linalg.solve(X.T @ X + lam,
                                X.T @ np.asarray(a_i, dtype=float) + lam @ np.array([self.a_, 0.0]))
        except np.linalg.LinAlgError:
            w = np.array([self.a_, 0.0])
        self.wa_ = w if np.all(np.isfinite(w)) else np.array([self.a_, 0.0])

    def _a(self, T, C):
        x, _ = _features(np.atleast_1d(T), np.atleast_1d(C))
        wa = getattr(self, "wa_", None)
        if wa is None:
            return self.a_
        return float(np.clip(wa[0] + wa[1] * x[0], 99.0, 106.0))

    def _clean(self, cell):
        """(T, C, n, soh) for one cell, or None if it carries no usable label."""
        try:
            d = cell.soh.dropna(subset=["soh_percent"])
            n = np.asarray(d["cycle_number"], dtype=float)
            y = np.asarray(d["soh_percent"], dtype=float)
            T = float(cell.temperature_degC)
            C = float(cell.c_rate)
        except Exception:
            return None
        ok = np.isfinite(n) & np.isfinite(y) & (n >= 1) & (y > 20.0) & (y < 130.0)
        n, y = n[ok], y[ok]
        if len(n) < MIN_POINTS or not (np.isfinite(T) and np.isfinite(C) and C > 0):
            return None
        n, idx = np.unique(n, return_index=True)      # one label per cycle number
        y = y[idx]
        if len(n) >= 3 * OUTLIER_WIN:
            # A handful of partial cycles survive the framework's own cleaning
            smooth = median_filter(y, size=OUTLIER_WIN, mode="nearest")
            keep = np.abs(y - smooth) <= OUTLIER_TOL
            if keep.sum() >= MIN_POINTS:
                n, y = n[keep], y[keep]
        return T, C, n, y

    def _fit_shape(self, curves):
        """Joint least squares: shape theta shared, (a_i, tau_i) free per cell."""
        m = len(curves)
        a0 = np.array([np.percentile(y, 99.5) for *_, y in curves])
        # each cell is normalised to weight 1, so the prior weight is in "cells"
        wts = [1.0 / np.sqrt(len(n)) for *_, n, _ in curves]
        tau0 = np.array([max(np.median(n), 50.0) for *_, n, _ in curves])
        p0 = np.concatenate([TH_PRIOR, a0, np.log(tau0)])
        lo = np.concatenate([TH_LO, np.full(m, 99.0), np.full(m, np.log(TAU_LO))])
        hi = np.concatenate([TH_HI, np.full(m, 106.0), np.full(m, np.log(TAU_HI))])

        def residuals(p):
            theta, a, ln_tau = p[:4], p[4:4 + m], p[4 + m:]
            parts = [(a[i] - fade_shape(curves[i][2] / np.exp(ln_tau[i]), theta)
                      - curves[i][3]) * wts[i] for i in range(m)]
            parts.append(TH_PRIOR_W * (theta - TH_PRIOR) / TH_SCALE)
            return np.concatenate(parts)

        r = least_squares(residuals, np.clip(p0, lo, hi), bounds=(lo, hi),
                          max_nfev=20000, x_scale="jac")
        theta = r.x[:4]
        return theta, r.x[4:4 + m], np.exp(r.x[4 + m:])

    def _fallback_shape(self, curves):
        """Prior shape, with tau_i fitted one cell at a time (closed-form scan)."""
        a_i, tau_i = [], []
        grid = np.exp(np.linspace(np.log(TAU_LO), np.log(TAU_HI), 400))
        for _, _, n, y in curves:
            a = float(np.percentile(y, 99.5))
            err = [np.mean((a - fade_shape(n / t, TH_PRIOR) - y) ** 2) for t in grid]
            a_i.append(a)
            tau_i.append(float(grid[int(np.argmin(err))]))
        return TH_PRIOR.copy(), np.array(a_i), np.array(tau_i)

    def _fit_law(self, T, C, ln_tau, weight):
        """ln tau = Arrhenius trend (shrunk to the prior) + GP residual."""
        x, lc = _features(T, C)
        X = np.column_stack([np.ones_like(x), x, lc, x ** 2])
        W = np.diag(weight)
        prior = np.array([LN_TAU_PRIOR, SLOPE_PRIOR, C_EXP_PRIOR, CURV_PRIOR])
        # Only estimate what the training grid can identify. Slope needs >=3
        # cells (not just >=2 temps): at exactly 2 cells differing in both T
        # and C, the C-exponent stays pinned (needs >=3 conditions) so any
        # real C-rate effect leaks into the barely-regularised slope. Matches
        # the one documented 2-cell loss to baseline (25C/1C + 45C/0.5C).
        n_T, n_C, n = len(set(np.round(T, 3))), len(set(np.round(C, 3))), len(ln_tau)
        free = np.array([True, n_T >= 2 and n >= 3, n_C >= 2 and n >= 3, n_T >= 3 and n >= 4])
        w = prior.copy()
        Xf, lam = X[:, free], np.diag(RIDGE[free])
        try:
            offset = ln_tau - X[:, ~free] @ prior[~free]
            w[free] = np.linalg.solve(Xf.T @ W @ Xf + lam,
                                      Xf.T @ W @ offset + lam @ prior[free])
        except np.linalg.LinAlgError:
            w = prior.copy()
        if not np.all(np.isfinite(w)):
            w = prior.copy()
        self.w_ = w
        self.free_ = free

        resid = ln_tau - X @ w
        z = np.column_stack([x / GP_ELL_X, lc / GP_ELL_C])
        d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
        K = GP_SIGMA_F ** 2 * np.exp(-0.5 * d2)
        # shallow cells get a bigger nugget: their tau is noisy, so the GP is
        # not allowed to chase it away from the trend.
        nugget = np.diag(GP_SIGMA_N ** 2 / np.asarray(weight, dtype=float))
        try:
            alpha = np.linalg.solve(K + nugget, resid)
        except np.linalg.LinAlgError:
            alpha = np.zeros(len(resid))
        self.gp_z_ = z
        self.gp_alpha_ = alpha if np.all(np.isfinite(alpha)) else np.zeros(len(resid))

    def _tau(self, T, C):
        x, lc = _features(np.atleast_1d(T), np.atleast_1d(C))
        ln_tau = float(np.array([1.0, x[0], lc[0], x[0] ** 2]) @ self.w_)
        if len(self.gp_alpha_):
            z = np.array([x[0] / GP_ELL_X, lc[0] / GP_ELL_C])
            k = GP_SIGMA_F ** 2 * np.exp(-0.5 * ((z - self.gp_z_) ** 2).sum(-1))
            ln_tau += float(np.clip(k @ self.gp_alpha_, -GP_CAP, GP_CAP))
        if not np.isfinite(ln_tau):
            ln_tau = LN_TAU_PRIOR
        return float(np.clip(np.exp(ln_tau), TAU_LO, TAU_HI))

    # -------------------------------------------------------------- predict
    def predict_soh(self, temperature_degC, c_rate):
        """SOH (%) for cycles 1..12000 at one operating point."""
        try:
            T = float(temperature_degC)
            C = float(c_rate)
        except (TypeError, ValueError):
            T, C = 40.0, C_REF
        if not np.isfinite(T):
            T = 40.0
        if not np.isfinite(C) or C <= 0:
            C = C_REF
        n = np.arange(1, N_CYCLES + 1, dtype=float)
        a = self._a(T, C)
        soh = a - fade_shape(n / self._tau(T, C), self.theta_, cap=a - SOH_FLOOR)
        soh = np.where(np.isfinite(soh), soh, SOH_LO)
        soh = np.minimum.accumulate(soh)          # degradation never recovers
        return np.clip(soh, SOH_LO, SOH_HI)
