#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Banc d'essai des candidats : in-sample, LOCO, budget réduit. Score relatif à la baseline."""
import argparse
import contextlib
import io
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


def erreurs(modele, cell):
    n, y = courbe(cell)
    traj = np.asarray(modele.predict_soh(cell.temperature_degC, cell.c_rate), dtype=float)
    e = traj[np.clip(n.astype(int), 1, len(traj)) - 1] - y
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e))), float(np.max(np.abs(e)))


def evaluer(nom, cells, protocoles):
    lignes = []
    for proto in protocoles:
        t0 = time.time()
        if proto == "in-sample":
            m = build(nom)
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(cells)
            res = [(c.cell_id, *erreurs(m, c)) for c in cells]
        else:
            n_max = None if proto == "loco" else int(proto.split("-")[1])
            res = []
            for held in cells:
                train = [c for c in cells if c.cell_id != held.cell_id]
                if n_max is not None:
                    train = [Tronquee(c, n_max) for c in train]
                m = build(nom)
                with contextlib.redirect_stdout(io.StringIO()):
                    m.fit(train)
                res.append((held.cell_id, *erreurs(m, held)))
        d = pd.DataFrame(res, columns=["cell", "rmse", "mae", "emax"])
        lignes.append(dict(candidat=nom, protocole=proto,
                           rmse=d.rmse.mean(), mae=d.mae.mean(), emax=d.emax.max(),
                           rmse_int=d[d.cell.isin(INTERIEUR)].rmse.mean(),
                           secondes=time.time() - t0))
        print(f"    {proto:10s} rmse={lignes[-1]['rmse']:7.3f} mae={lignes[-1]['mae']:7.3f} "
              f"({lignes[-1]['secondes']:.0f}s)", flush=True)
    return lignes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidats", nargs="*", default=None)
    ap.add_argument("--data", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--quick", action="store_true", help="in-sample + loco seulement")
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark.csv"))
    args = ap.parse_args()

    noms = args.candidats or list(CANDIDATS)
    protocoles = ["in-sample", "loco"] if args.quick else ["in-sample", "loco", "loco-800", "loco-400"]
    cells = sorted(load_cells(args.data, verbose=False),
                   key=lambda c: (c.temperature_degC, c.c_rate))
    print(f"{len(cells)} cellules · "
          f"protocoles : {', '.join(protocoles)}\n")

    lignes = []
    for nom in noms:
        print(f"  {nom}", flush=True)
        try:
            lignes += evaluer(nom, cells, protocoles)
        except Exception as exc:                     # un candidat qui plante n'arrête pas le banc
            print(f"    ECHEC : {type(exc).__name__}: {exc}", flush=True)
            lignes.append(dict(candidat=nom, protocole="ECHEC", rmse=np.nan, mae=np.nan,
                               emax=np.nan, rmse_int=np.nan, secondes=0.0))

    df = pd.DataFrame(lignes)
    df.to_csv(args.out, index=False)
    base = df[df.candidat == "baseline"].set_index("protocole")["rmse"]
    base_mae = df[df.candidat == "baseline"].set_index("protocole")["mae"]
    if len(base):
        df["rmse_rel"] = [r.rmse / base.get(r.protocole, np.nan) for r in df.itertuples()]
        df["mae_rel"] = [r.mae / base_mae.get(r.protocole, np.nan) for r in df.itertuples()]
    pd.set_option("display.width", 200)
    print("\n=== RMSE relatif (baseline = 1.00, plus bas est meilleur) ===")
    print(df.pivot_table(index="candidat", columns="protocole", values="rmse_rel").round(3).to_string())
    print("\n=== RMSE absolu (points de SOH) ===")
    print(df.pivot_table(index="candidat", columns="protocole", values="rmse").round(2).to_string())
    print("\n=== temps d'ajustement cumulé (s) ===")
    print(df.groupby("candidat").secondes.sum().round(1).to_string())
    print(f"\ndétail écrit dans {args.out}")


if __name__ == "__main__":
    main()
