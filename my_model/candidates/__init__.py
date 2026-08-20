# -*- coding: utf-8 -*-
"""Candidats comparables. Hors livraison : importés seulement par scripts/benchmark.py."""
from .base import Candidate
from .gbm import GbmDirect, GbmResidual, available_backend
from .hybrid import Ensemble, Residual
from .neural import LstmModel, TransformerModel


class MasterCurve(Candidate):
    """Le modèle livré (courbe maîtresse + loi tau), comme référence."""

    name = "master"

    def fit(self, cells):
        from ..model_template import MyModel
        self.modele_ = MyModel().fit(cells)
        self.fitted_ = True
        return self

    def predict_soh(self, temperature_degC, c_rate):
        return self.modele_.predict_soh(temperature_degC, c_rate)


class Baseline(Candidate):
    """`model_example.py`, la référence du barème (score 1.00)."""

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
    "gbm": lambda: GbmDirect(),
    "lstm": lambda: LstmModel(),
    "transformer": lambda: TransformerModel(),
    "master+gbm": lambda: GbmResidual(),
    "lstm+gbm": lambda: Residual(LstmModel(), GbmDirect(), nom="lstm+gbm"),
    "transformer+gbm": lambda: Residual(TransformerModel(), GbmDirect(), nom="transformer+gbm"),
    "lstm+transformer": lambda: Ensemble([LstmModel(), TransformerModel()],
                                         nom="lstm+transformer"),
    "master+lstm": lambda: Residual(MasterCurve(), LstmModel(), nom="master+lstm"),
}


def build(nom):
    """Instancie un candidat par son nom."""
    if nom not in CANDIDATS:
        raise KeyError(f"candidat inconnu : {nom!r}. Disponibles : {sorted(CANDIDATS)}")
    return CANDIDATS[nom]()


__all__ = ["CANDIDATS", "build", "available_backend", "Candidate", "Ensemble", "Residual"]
