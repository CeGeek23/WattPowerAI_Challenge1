#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Banc d'essai des candidats. Score relatif à la baseline.

Les organisateurs notent la trajectoire « du cycle 1 jusqu'à 70% SOH, knee-point
inclus » contre des cellules vieillies profondément à travers le knee (PDF du
challenge, §1). Noter tous les cycles étiquetés des 6 cellules livrées surestime
donc largement le modèle : la moitié d'entre elles s'arrêtent avant le knee, et
le début de vie — où on excelle — pèse alors l'essentiel du score.

D'où les protocoles `deep*` (fenêtre profonde seulement) et `profond` (cellules
qui atteignent réellement la fin de vie). Sur les mêmes prédictions LOCO :
tous cycles 0.58, `profond` 0.67, `deep` 0.84 — contre 0.87 mesuré au barème.

L'agrégation compte autant que la fenêtre : `--model test` est invoqué « 30 min
par cellule », donc un barème par cellule est le plus probable. On reporte donc
le ratio-des-moyennes (`rel`), la moyenne-des-ratios (`rel_cell`) et le pire
ratio par cellule (`rel_pire`) — c'est ce dernier qui nous coule dans le knee.
"""
import argparse
import contextlib
import io
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from framework.data import load_cells  # noqa: E402
from my_model.candidates import CANDIDATS, build  # noqa: E402

INTERIEUR = {"102Ah_35degC_1C_cell1", "102Ah_45degC_0p5C_cell3", "102Ah_45degC_1C_cell1"}
SEUIL_PROFOND = 75.0             # une cellule "profonde" descend au moins jusque-là
SEUIL_DEEP = 80.0                # fenêtre notée par les protocoles deep*

# nom -> (n_max d'entraînement, seuil de notation, cellules profondes seulement)
PROTOCOLES = {
    "in-sample":  (None, None, False),
    "loco":       (None, None, False),
    "loco-800":   (800,  None, False),
    "loco-400":   (400,  None, False),
    "deep":       (None, SEUIL_DEEP, False),
    "deep-800":   (800,  SEUIL_DEEP, False),
    "deep-400":   (400,  SEUIL_DEEP, False),
    "profond":    (None, None, True),
    "paires":     (None, None, False),   # les 15 couples d'entraînement possibles
    "solo":       (None, None, False),   # une seule cellule d'entraînement
}
DEFAUT = ["in-sample", "loco", "profond", "deep", "loco-800", "loco-400", "deep-800"]


class Tronquee:
    """Vue d'une cellule limitée aux cycles <= n_max (rejeu budget réduit)."""

    def __init__(self, cell, n_max):
        self.cell_id = cell.cell_id
        self.temperature_degC = cell.temperature_degC
        self.c_rate = cell.c_rate
        self.soh = cell.soh[cell.soh["cycle_number"] <= n_max].reset_index(drop=True)

    def time_series(self):
        raise RuntimeError("non utilisé par les candidats")


def courbe(cell):
    d = cell.soh.dropna(subset=["soh_percent"])
    return (np.asarray(d["cycle_number"], dtype=float),
            np.asarray(d["soh_percent"], dtype=float))


def profonde(cell):
    _, y = courbe(cell)
    return len(y) and y.min() <= SEUIL_PROFOND


def erreurs(modele, cell, seuil=None):
    """RMSE/MAE/max sur la cellule, restreints à soh <= seuil si demandé."""
    n, y = courbe(cell)
    traj = np.asarray(modele.predict_soh(cell.temperature_degC, cell.c_rate), dtype=float)
    e = traj[np.clip(n.astype(int), 1, len(traj)) - 1] - y
    if seuil is not None:
        garde = y <= seuil
        if garde.sum() < 5:                       # trop peu de points : cellule écartée
            return None
        e = e[garde]
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e))), float(np.max(np.abs(e)))


def _plis(cells, proto):
    """(étiquette du pli, cellules d'entraînement, cellules notées)."""
    if proto == "in-sample":
        return [("tout", cells, cells)]
    if proto == "paires":
        return [(f"{a.cell_id[5:]}+{b.cell_id[5:]}", [a, b],
                 [c for c in cells if c.cell_id not in (a.cell_id, b.cell_id)])
                for a, b in itertools.combinations(cells, 2)]
    if proto == "solo":
        return [(a.cell_id[5:], [a], [c for c in cells if c.cell_id != a.cell_id])
                for a in cells]
    return [(h.cell_id[5:], [c for c in cells if c.cell_id != h.cell_id], [h]) for h in cells]


def evaluer(nom, cells, protocoles):
    """Une ligne par (protocole, pli, cellule notée) : l'agrégation se fait après."""
    lignes = []
    for proto in protocoles:
        n_max, seuil, prof = PROTOCOLES[proto]
        t0 = time.time()
        cible = [c for c in cells if profonde(c)] if prof else cells
        for pli, train, notees in _plis(cells, proto):
            if n_max is not None:
                train = [Tronquee(c, n_max) for c in train]
            m = build(nom)
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(train)
            for c in notees:
                if prof and c.cell_id not in {x.cell_id for x in cible}:
                    continue
                e = erreurs(m, c, seuil)
                if e is None:
                    continue
                lignes.append(dict(candidat=nom, protocole=proto, pli=pli, cell=c.cell_id,
                                   rmse=e[0], mae=e[1], emax=e[2]))
        n = sum(1 for r in lignes if r["protocole"] == proto)
        print(f"    {proto:10s} {n:3d} notes  ({time.time() - t0:.0f}s)", flush=True)
    return lignes


def agreger(det):
    """Ratios à la baseline, appariés pli par pli et cellule par cellule."""
    base = det[det.candidat == "baseline"].set_index(["protocole", "pli", "cell"])["rmse"]
    out = []
    for (nom, proto), g in det.groupby(["candidat", "protocole"], sort=False):
        b = base.reindex(pd.MultiIndex.from_frame(g[["protocole", "pli", "cell"]])).to_numpy()
        r = g.rmse.to_numpy()
        ok = np.isfinite(b) & (b > 0)
        ratios = r[ok] / b[ok]
        out.append(dict(
            candidat=nom, protocole=proto, n=len(g),
            rmse=r.mean(), mae=g.mae.mean(), emax=g.emax.max(),
            rmse_int=g[g.cell.isin(INTERIEUR)].rmse.mean(),
            rel=r[ok].mean() / b[ok].mean() if ok.any() else np.nan,
            rel_cell=ratios.mean() if ok.any() else np.nan,
            rel_pire=ratios.max() if ok.any() else np.nan))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidats", nargs="*", default=None)
    ap.add_argument("--data", default=os.path.join(ROOT, "dataset", "target"))
    ap.add_argument("--protocoles", default=",".join(DEFAUT),
                    help=f"parmi : {', '.join(PROTOCOLES)}")
    ap.add_argument("--quick", action="store_true", help="in-sample + loco + profond seulement")
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark.csv"))
    args = ap.parse_args()

    noms = args.candidats or list(CANDIDATS)
    if "baseline" not in noms:                    # sans elle, aucun ratio n'est calculable
        noms = noms + ["baseline"]
    protocoles = (["in-sample", "loco", "profond"] if args.quick
                  else [p.strip() for p in args.protocoles.split(",") if p.strip()])
    inconnus = [p for p in protocoles if p not in PROTOCOLES]
    if inconnus:
        ap.error(f"protocole inconnu : {', '.join(inconnus)}")

    cells = sorted(load_cells(args.data, verbose=False),
                   key=lambda c: (c.temperature_degC, c.c_rate))
    prof = [c.cell_id for c in cells if profonde(c)]
    print(f"{len(cells)} cellules, dont {len(prof)} profondes (SOH <= {SEUIL_PROFOND:.0f}) · "
          f"protocoles : {', '.join(protocoles)}\n")

    det = []
    for nom in noms:
        print(f"  {nom}", flush=True)
        try:
            det += evaluer(nom, cells, protocoles)
        except Exception as exc:                  # un candidat qui plante n'arrête pas le banc
            print(f"    ECHEC : {type(exc).__name__}: {exc}", flush=True)

    det = pd.DataFrame(det)
    det.to_csv(args.out.replace(".csv", "_detail.csv"), index=False)
    df = agreger(det)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    for cle, titre in (("rel", "ratio des moyennes"),
                       ("rel_cell", "moyenne des ratios (barème par cellule)"),
                       ("rel_pire", "pire cellule")):
        print(f"\n=== RMSE relatif, {titre} (baseline = 1.00, plus bas est meilleur) ===")
        print(df.pivot_table(index="candidat", columns="protocole",
                             values=cle).reindex(columns=protocoles).round(3).to_string())
    print("\n=== RMSE absolu (points de SOH) ===")
    print(df.pivot_table(index="candidat", columns="protocole",
                         values="rmse").reindex(columns=protocoles).round(2).to_string())
    print(f"\ndétail par cellule dans {args.out.replace('.csv', '_detail.csv')}")


if __name__ == "__main__":
    main()
