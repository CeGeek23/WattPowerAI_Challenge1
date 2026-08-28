# -*- coding: utf-8 -*-
"""Résidu physique contraint : petit réseau + perte de courbure (autograd).

Forme `L(u)` et tendance Arrhenius identiques à MyModel (même fit conjoint) ;
seul le résidu sur `ln tau(T,C)` change, GP -> réseau `NN(x, ln C)`, entraîné
sur deux pertes : coller aux tau_i par cellule (comme le GP), + pénaliser
d²NN/dx² sur des points de collocation tirés dans tout le domaine (pas
seulement les 6 conditions observées) - la contrainte physique qui distingue
ça d'un MLP libre.

Mesuré et écarté (`scripts/benchmark.py master baseline pinn_gpu`) : rel.
RMSE in-sample 0.87 / loco 1.05 / loco-800 1.60 / loco-400 1.14, contre
0.24 / 0.58 / 0.52 / 0.71 pour master - pire sur les quatre protocoles, pire
que la baseline elle-même sur loco/loco-800/loco-400. 71.8s (GPU MPS) contre
1.1s. Même contraint, un réseau ne bat pas le GP fermé sur 6 cellules sans
réplicat - cf. neural.py, même constat pour LSTM->Transformer. Gardé pour
la trace, pas activé.
"""
import numpy as np

from .base import Candidate

T_REF_K = 313.15
C_REF = 0.75
TH_LO = np.array([2.0, 0.25, 0.0, 1.0])
TH_HI = np.array([40.0, 1.50, 20.0, 5.0])
TH_PRIOR = np.array([15.0, 0.65, 0.35, 2.70])
TH_SCALE = np.array([10.0, 0.30, 0.30, 1.00])
TH_PRIOR_W = 0.45
LN_TAU_PRIOR = np.log(1000.0)
SLOPE_PRIOR = 2.8
C_EXP_PRIOR = -0.15
TAU_LO, TAU_HI = 20.0, 20000.0
DOMAIN_T = (25.0, 55.0)          # degC, pour les points de collocation
DOMAIN_C = (0.5, 1.0)


def _torch():
    import torch
    return torch


def _features(T, C):
    x = 1000.0 / (np.asarray(T, dtype=float) + 273.15) - 1000.0 / T_REF_K
    lc = np.log(np.maximum(np.asarray(C, dtype=float), 1e-3) / C_REF)
    return x, lc


def _fade_shape(u, theta):
    A, p, B, q = theta
    u = np.maximum(np.asarray(u, dtype=float), 0.0)
    return A * (1.0 - np.exp(-np.clip(u ** p, 0.0, 60.0))) + B * u ** q


class PinnTauModel(Candidate):
    """Forme partagée identique à MyModel ; résidu de tau appris par un petit réseau contraint."""

    name = "pinn"
    needs = ("torch",)

    def __init__(self, hidden=8, epochs=1500, lr=0.01, weight_decay=1e-3,
                 lambda_smooth=0.3, n_colloc=64, device="cpu"):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_smooth = lambda_smooth   # poids de la perte de courbure
        self.n_colloc = n_colloc             # points de collocation (physique) par époque
        self.device = device

    # --------------------------------------------------------------- forme
    def _fit_shape(self, curves):
        """Identique à MyModel._fit_shape : moindres carrés conjoint theta/(a_i, tau_i)."""
        from scipy.optimize import least_squares
        m = len(curves)
        a0 = np.array([np.percentile(y, 99.5) for *_, y in curves])
        wts = [1.0 / np.sqrt(len(n)) for *_, n, _ in curves]
        tau0 = np.array([max(np.median(n), 50.0) for *_, n, _ in curves])
        p0 = np.concatenate([TH_PRIOR, a0, np.log(tau0)])
        lo = np.concatenate([TH_LO, np.full(m, 99.0), np.full(m, np.log(TAU_LO))])
        hi = np.concatenate([TH_HI, np.full(m, 106.0), np.full(m, np.log(TAU_HI))])

        def residuals(p):
            theta, a, ln_tau = p[:4], p[4:4 + m], p[4 + m:]
            parts = [(a[i] - _fade_shape(curves[i][2] / np.exp(ln_tau[i]), theta)
                      - curves[i][3]) * wts[i] for i in range(m)]
            parts.append(TH_PRIOR_W * (theta - TH_PRIOR) / TH_SCALE)
            return np.concatenate(parts)

        r = least_squares(residuals, np.clip(p0, lo, hi), bounds=(lo, hi),
                          max_nfev=20000, x_scale="jac")
        return r.x[:4], r.x[4:4 + m], np.exp(r.x[4 + m:])

    # ---------------------------------------------------------------- tau
    def _fit(self, curves):
        torch = _torch()
        torch.manual_seed(0)
        self.theta_, a_i, tau_i = self._fit_shape(curves)
        self.a_ = float(np.clip(np.mean(a_i), 99.0, 106.0))

        T = np.array([c[0] for c in curves])
        C = np.array([c[1] for c in curves])
        n_labels = np.array([len(c[2]) for c in curves])
        ln_tau = np.log(tau_i)
        x, lc = _features(T, C)

        # tendance Arrhenius : meme forme fermee que MyModel (ridge leger),
        # seul le residu passe par le reseau.
        X = np.column_stack([np.ones_like(x), x, lc])
        ridge = np.diag([1e-6, 0.02, 0.30])
        w = np.linalg.solve(X.T @ X + ridge, X.T @ ln_tau + ridge @ np.array(
            [LN_TAU_PRIOR, SLOPE_PRIOR, C_EXP_PRIOR]))
        self.w_ = w
        resid = ln_tau - X @ w

        d = self._appareil()
        self.appareil_ = d
        depth = np.array([max(float(a_i[i]) - float(curves[i][3].min()), 0.0)
                          for i in range(len(curves))])
        weight = np.clip(depth / 5.0, 0.15, 1.0)

        xt = torch.tensor(np.column_stack([x, lc]), dtype=torch.float32, device=d)
        rt = torch.tensor(resid, dtype=torch.float32, device=d)
        wt = torch.tensor(weight, dtype=torch.float32, device=d)

        self.net_ = torch.nn.Sequential(
            torch.nn.Linear(2, self.hidden), torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, self.hidden), torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, 1),
        ).to(d)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        rng = np.random.default_rng(0)
        x_lo, x_hi = _features(np.array(DOMAIN_T), np.array([C_REF, C_REF]))[0]
        _, lc_lo = _features(np.array([40.0]), np.array([DOMAIN_C[0]]))
        _, lc_hi = _features(np.array([40.0]), np.array([DOMAIN_C[1]]))
        x_lo, x_hi = min(x_lo, x_hi), max(x_lo, x_hi)
        lc_lo, lc_hi = float(lc_lo[0]), float(lc_hi[0])

        histo = []
        self.net_.train()
        for ep in range(self.epochs):
            opt.zero_grad()
            pred = self.net_(xt).squeeze(-1)
            perte_data = (wt * (pred - rt) ** 2).sum() / wt.sum()

            # points de collocation : la reponse doit rester lisse partout dans
            # le domaine, pas seulement aux 6 conditions observees.
            xc = torch.tensor(rng.uniform(x_lo, x_hi, self.n_colloc),
                              dtype=torch.float32, device=d, requires_grad=True)
            lcc = torch.tensor(rng.uniform(lc_lo, lc_hi, self.n_colloc),
                               dtype=torch.float32, device=d, requires_grad=True)
            zc = torch.stack([xc, lcc], dim=1)
            fc = self.net_(zc).squeeze(-1)
            (dfx,) = torch.autograd.grad(fc.sum(), xc, create_graph=True)
            (d2fx,) = torch.autograd.grad(dfx.sum(), xc, create_graph=True)
            perte_lisse = (d2fx ** 2).mean()

            perte = perte_data + self.lambda_smooth * perte_lisse
            perte.backward()
            opt.step()
            if ep % 50 == 0:
                histo.append((ep, float(perte.detach())))
        self.net_.eval()
        self.historique_ = histo
        self.perte_ = histo[-1][1] if histo else float("nan")

    def _appareil(self):
        torch = _torch()
        choix = self.device
        if choix != "auto":
            return torch.device(choix)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _tau(self, T, C):
        torch = _torch()
        x, lc = _features(np.array([T]), np.array([C]))
        ln_tau = float(np.array([1.0, x[0], lc[0]]) @ self.w_)
        with torch.no_grad():
            z = torch.tensor([[x[0], lc[0]]], dtype=torch.float32, device=self.appareil_)
            ln_tau += float(self.net_(z).cpu().item())
        if not np.isfinite(ln_tau):
            ln_tau = LN_TAU_PRIOR
        return float(np.clip(np.exp(ln_tau), TAU_LO, TAU_HI))

    def _trajectory(self, T, C):
        n = np.arange(1, 12001, dtype=float)
        tau = self._tau(T, C)
        return self.a_ - _fade_shape(n / tau, self.theta_)

    def __getstate__(self):
        etat = dict(self.__dict__)
        etat.pop("appareil_", None)
        net = etat.pop("net_", None)
        if net is not None:
            etat["_poids"] = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
        return etat

    def __setstate__(self, etat):
        poids = etat.pop("_poids", None)
        self.__dict__.update(etat)
        if poids is not None:
            torch = _torch()
            self.appareil_ = self._appareil()
            self.net_ = torch.nn.Sequential(
                torch.nn.Linear(2, self.hidden), torch.nn.Tanh(),
                torch.nn.Linear(self.hidden, self.hidden), torch.nn.Tanh(),
                torch.nn.Linear(self.hidden, 1),
            ).to(self.appareil_)
            self.net_.load_state_dict({k: torch.tensor(v, device=self.appareil_)
                                       for k, v in poids.items()})
            self.net_.eval()
