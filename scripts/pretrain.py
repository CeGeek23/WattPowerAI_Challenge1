#!/usr/bin/env python3
"""Pré-entraînement hors ligne sur données publiques -> my_model/pretrained.json.

    python scripts/pretrain.py scratchpad/pretrain_*.csv

Transfère ce qui est transférable entre chimies et formats : la FORME de la
courbe de fade et la pente d'Arrhenius. Pas les niveaux : une 18650 de 1.1 Ah et
une prismatique de 102 Ah n'ont pas la même vitesse absolue.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from my_model.model_template import MyModel  # noqa: E402

COLS = ["source", "cell_id", "chemistry", "nominal_Ah", "T_degC",
        "c_rate_charge", "c_rate_discharge", "cycle", "soh_pct"]
MIN_POINTS = 20
MIN_FADE = 3.0          # points de SOH perdus, sinon la cellule ne dit rien de la forme
RMSE_FORME_MAX = 1.5    # au-dela, la forme publique ne decrit pas ces cellules : on ne l'exporte pas
RMSE_FORME_P90_MAX = 5.0  # et il faut aussi une queue utilisable, pas seulement une mediane
N_TEMPS_MIN = 3         # temperatures distinctes requises pour exporter une pente d'Arrhenius
N_CELLS_CURV_MIN = 4    # + N_TEMPS_MIN temperatures : meme seuil d'identifiabilite que
                        # MyModel._fit_law pour le terme de courbure (n_T>=3 et n>=4)
N_CRATES_MIN = 2        # + chimie LFP (gate plus bas). Che (NMC) ecarte par la chimie,
                        # pas un veto aveugle -- Wheeler (LFP, contraste 0.33C/2.0C) etait
                        # rejete par l'ancien seuil de 99 sans jamais etre regarde.


class PseudoCell:
    """Courbe publique présentée comme une Cell du framework."""

    def __init__(self, cell_id, T, C, n, y):
        self.cell_id = cell_id
        self.temperature_degC = float(T)
        self.c_rate = float(C)
        self.soh = pd.DataFrame({"cycle_number": n, "soh_percent": y})

    def time_series(self):
        raise RuntimeError("pré-entraînement : courbes SOH seulement")


def charge_csv(chemins, chimie=None):
    frames = []
    for motif in chemins:
        for f in sorted(glob.glob(motif)):
            d = pd.read_csv(f)
            manquantes = [c for c in COLS if c not in d.columns]
            if manquantes:
                print(f"  [ignoré] {os.path.basename(f)} : colonnes manquantes {manquantes}")
                continue
            frames.append(d[COLS])
            print(f"  [lu] {os.path.basename(f)} : {len(d)} lignes, "
                  f"{d.cell_id.nunique()} cellules")
    if not frames:
        sys.exit("aucun CSV exploitable")
    d = pd.concat(frames, ignore_index=True)
    if chimie:
        avant = d.cell_id.nunique()
        d = d[d.chemistry.str.contains(chimie, case=False, na=False)]
        print(f"  filtre chimie {chimie!r} : {avant} -> {d.cell_id.nunique()} cellules")
    return d


def en_cellules(d):
    """Une PseudoCell par cellule publique, courbe renormalisée à 100 au départ.

    Indispensable : les jeux publics normalisent par la capacité nominale, et la
    capacité neuve réelle en est souvent loin (Wheeler : 0.97-1.03 Ah pour un
    nominal de 1.1, donc des courbes qui démarrent à 88-93%). Sans renormalisation
    la forme ajustée serait celle d'une cellule déjà usée.
    """
    cells, rejets = [], 0
    for (src, cid), g in d.groupby(["source", "cell_id"]):
        g = g.dropna(subset=["cycle", "soh_pct"]).sort_values("cycle")
        n = g["cycle"].to_numpy(float)
        y = g["soh_pct"].to_numpy(float)
        ok = np.isfinite(n) & np.isfinite(y) & (n >= 1) & (y > 20) & (y < 130)
        n, y = n[ok], y[ok]
        if len(n) >= 3:
            depart = float(np.percentile(y[:max(3, len(y) // 20)], 90))
            if depart > 50:
                y = y * (100.0 / depart)
        if len(n) < MIN_POINTS or (y.max() - y.min()) < MIN_FADE:
            rejets += 1
            continue
        T = float(g["T_degC"].median())
        C = float(g["c_rate_discharge"].median())
        if not (np.isfinite(T) and np.isfinite(C) and C > 0):
            rejets += 1
            continue
        cells.append(PseudoCell(f"{src}:{cid}", T, C, n, y))
    print(f"  {len(cells)} cellules retenues, {rejets} écartées "
          f"(< {MIN_POINTS} points ou < {MIN_FADE} points de fade)")
    return cells


def qualite(modele, cells):
    """RMSE de la forme partagée sur chaque cellule publique : la preuve du transfert."""
    err = []
    for c in cells:
        n = c.soh["cycle_number"].to_numpy(float)
        y = c.soh["soh_percent"].to_numpy(float)
        p = np.asarray(modele.predict_soh(c.temperature_degC, c.c_rate))
        p = p[np.clip(n.astype(int), 1, len(p)) - 1]
        err.append(float(np.sqrt(np.mean((p - y) ** 2))))
    return np.array(err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="CSV normalisés (colonnes imposées)")
    ap.add_argument("--chimie", default="LFP", help="filtre chimie, vide pour tout garder")
    ap.add_argument("--out", default=os.path.join(ROOT, "my_model", "pretrained.json"))
    args = ap.parse_args()

    d = charge_csv(args.csv, args.chimie or None)
    cells = en_cellules(d)
    if len(cells) < 3:
        sys.exit("moins de 3 cellules exploitables : pré-entraînement inutile")

    modele = MyModel().fit(cells)
    err = qualite(modele, cells)
    temps = sorted({c.temperature_degC for c in cells})
    crates = sorted({c.c_rate for c in cells})

    med, p90 = float(np.median(err)), float(np.percentile(err, 90))
    poids, ecarte = {}, []
    # La mediane seule ne suffit pas : sur Che elle vaut 1.48 (sous le seuil)
    # alors que le p90 vaut 10.4 points de SOH -- une forme qui decrit la moitie
    # des cellules et rate l'autre completement. Importee, elle degrade tous les
    # rejeux tronques (loco-800 0.493 -> 0.729). On exige les deux.
    if med <= RMSE_FORME_MAX and p90 <= RMSE_FORME_P90_MAX:
        poids["shape"] = [float(v) for v in modele.theta_]
    elif med > RMSE_FORME_MAX:
        ecarte.append(f"forme (RMSE mediane {med:.2f} > {RMSE_FORME_MAX})")
    else:
        ecarte.append(f"forme (RMSE p90 {p90:.1f} > {RMSE_FORME_P90_MAX} : "
                      f"mediane {med:.2f} correcte mais queue inutilisable)")
    if len(temps) >= N_TEMPS_MIN:
        poids["slope"] = float(modele.w_[1])
    else:
        ecarte.append(f"pente Arrhenius ({len(temps)} temperature(s) seulement)")
    if len(temps) >= N_TEMPS_MIN and len(cells) >= N_CELLS_CURV_MIN:
        # Le SIGNE de la courbure transfere (Che et la cible sont negatifs tous
        # les deux), pas sa MAGNITUDE : -7.2 sur NMC contre -1.4 ajuste sur la
        # cible. Exporter la valeur publique telle quelle reviendrait a imposer
        # un chiffre d'une autre chimie, le ridge sur ce terme etant trop faible
        # pour que les donnees le corrigent. On la trace sans la livrer ;
        # MyModel.CURV_PRIOR retient une valeur entre les deux estimations.
        curv_public = float(modele.w_[3])
        ecarte.append(f"courbure (valeur publique {curv_public:.2f} : signe retenu, "
                      f"magnitude specifique a la chimie, cf. CURV_PRIOR)")
    else:
        ecarte.append(f"courbure ({len(temps)} temperature(s), {len(cells)} cellule(s) : "
                      f"seuil {N_TEMPS_MIN} temperatures et {N_CELLS_CURV_MIN} cellules)")
    chimies = sorted(d.chemistry.dropna().unique().tolist())
    est_lfp = bool(chimies) and all("lfp" in str(c).lower() for c in chimies)
    if len(crates) >= N_CRATES_MIN and est_lfp:
        poids["c_exp"] = float(modele.w_[2])
    elif not est_lfp:
        ecarte.append(f"exposant C (chimie {chimies} != LFP, protocoles non comparables a la cible)")
    else:
        ecarte.append(f"exposant C ({len(crates)} C-rate(s) seulement)")
    poids.update({
        "meta": {
            "curv_public": (round(float(modele.w_[3]), 4)
                            if len(temps) >= N_TEMPS_MIN and len(cells) >= N_CELLS_CURV_MIN
                            else None),
            "ecarte": ecarte,
            "n_cells": len(cells),
            "sources": sorted(d.source.unique().tolist()),
            "chemistries": sorted(d.chemistry.dropna().unique().tolist()),
            "temperatures_degC": temps,
            "c_rates": crates,
            "rmse_forme_partagee": {"median": round(float(np.median(err)), 3),
                                    "p90": round(float(np.percentile(err, 90)), 3)},
        },
    })
    with open(args.out, "w") as fh:
        json.dump(poids, fh, indent=1, sort_keys=True)

    print(f"\nforme     A={modele.theta_[0]:.2f} p={modele.theta_[1]:.3f} "
          f"B={modele.theta_[2]:.3f} q={modele.theta_[3]:.3f}")
    print(f"loi       pente Arrhenius={modele.w_[1]:.2f} "
          f"(Ea≈{modele.w_[1] * 8.314:.0f} kJ/mol) · exposant C={modele.w_[2]:.3f} "
          f"· courbure={modele.w_[3]:.3f}")
    print(f"exporte   {sorted(k for k in poids if k != 'meta')} · ecarte : {ecarte or 'rien'}")
    print(f"couverture T={temps} C={crates}")
    print(f"RMSE forme partagée sur les cellules publiques : médiane "
          f"{np.median(err):.2f}, p90 {np.percentile(err, 90):.2f} points de SOH")
    print(f"écrit : {args.out}")


if __name__ == "__main__":
    main()
