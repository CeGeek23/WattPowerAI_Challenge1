# -*- coding: utf-8 -*-
"""Hors livraison, importe seulement par scripts/benchmark.py.

baseline = bareme (1.00). master = modele livre. Reste : candidats testes
et ecartes (LSTM/Transformer seuls, couplage, pinn) - gardes pour la trace.
"""
import os

from .base import Candidate
from .neural import LstmTransformerModel
from .pinn import PinnTauModel

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Wheeler est livre (68 ko) ; Che est regenere par scripts/pretrain_data/ et
# reste dans le cache, trop gros pour le depot.
CSV_PUBLICS = [os.path.join(RACINE, "scripts", "pretrain_data", "pretrain_wheeler.csv"),
               os.path.join(RACINE, "dataset", "che", "pretrain_che.csv")]


def courbes_publiques(chemins=None):
    """[(T, C, n, soh)] des cellules publiques, pour le pre-entrainement."""
    import numpy as np
    import pandas as pd

    out = []
    for f in (chemins or CSV_PUBLICS):
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        for _, g in d.groupby(["source", "cell_id"]):
            g = g.dropna(subset=["cycle", "soh_pct"]).sort_values("cycle")
            n = g["cycle"].to_numpy(float)
            y = g["soh_pct"].to_numpy(float)
            ok = np.isfinite(n) & np.isfinite(y) & (n >= 1) & (y > 20) & (y < 130)
            n, y = n[ok], y[ok]
            if len(n) < 20 or (y.max() - y.min()) < 3.0:
                continue
            depart = float(np.percentile(y[:max(3, len(y) // 20)], 90))
            if depart > 50:                      # ramener la courbe a 100 au depart
                y = y * (100.0 / depart)
            T, C = float(g["T_degC"].median()), float(g["c_rate_discharge"].median())
            if np.isfinite(T) and np.isfinite(C) and C > 0:
                out.append((T, C, n, y))
    return out


class MasterCurve(Candidate):
    """Le modele livre, comme reference."""

    name = "master"

    def fit(self, cells):
        from ..model_template import MyModel
        self.modele_ = MyModel().fit(cells)
        self.fitted_ = True
        return self

    def predict_soh(self, temperature_degC, c_rate):
        return self.modele_.predict_soh(temperature_degC, c_rate)


class Baseline(Candidate):
    """`model_example.py`, la reference du bareme (score 1.00)."""

    name = "baseline"

    def fit(self, cells):
        import contextlib
        import io

        from ..model_example import ExampleModel
        self.modele_ = ExampleModel()
        with contextlib.redirect_stdout(io.StringIO()):
            self.modele_.fit(cells)
        self.fitted_ = True
        return self

    def predict_soh(self, temperature_degC, c_rate):
        return self.modele_.predict_soh(temperature_degC, c_rate)


CANDIDATS = {
    "baseline": lambda: Baseline(),
    "master": lambda: MasterCurve(),
    "lstm+transformer": lambda: LstmTransformerModel(),
    "lstm+transformer_pre": lambda: LstmTransformerModel(pretrain_curves=courbes_publiques()),
    "pinn": lambda: PinnTauModel(),
    "pinn_gpu": lambda: PinnTauModel(device="auto"),
}


def build(nom, **hyper):
    """Instancie un candidat ; `hyper` ecrase ses hyperparametres (recherche)."""
    if nom not in CANDIDATS:
        raise KeyError(f"candidat inconnu : {nom!r}. Disponibles : {sorted(CANDIDATS)}")
    modele = CANDIDATS[nom]()
    for k, v in hyper.items():
        if not hasattr(modele, k):
            raise KeyError(f"{nom} n'a pas d'hyperparametre {k!r}")
        setattr(modele, k, type(getattr(modele, k))(v) if getattr(modele, k) is not None else v)
    return modele


__all__ = ["CANDIDATS", "build", "courbes_publiques", "Candidate"]
