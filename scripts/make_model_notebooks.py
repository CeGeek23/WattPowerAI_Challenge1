#!/usr/bin/env python3
"""Génère un notebook d'évaluation par candidat : perte, courbes, résidus, erreurs.

    python scripts/make_model_notebooks.py            # le couplage
    python scripts/make_model_notebooks.py lstm+transformer_pre
"""
import json
import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAUT = ["lstm+transformer"]

SETUP = '''import os, sys, time
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.getcwd()
while not os.path.isdir(os.path.join(ROOT, "my_model")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from framework.data import load_cells
from my_model.candidates import build, courbes_publiques

CANDIDAT = "{nom}"
cells = sorted(load_cells(os.path.join(ROOT, "dataset"), verbose=False),
               key=lambda c: (c.temperature_degC, c.c_rate))
COULEUR = dict(zip([c.cell_id for c in cells],
                   pc.sample_colorscale("Turbo", [i / 5 for i in range(6)])))
DASH = {{0.5: "dash", 1.0: "solid"}}

def mise_en_forme(fig, titre, hauteur=520, **kw):
    fig.update_layout(width=1020, height=hauteur, template="plotly_white",
                      title=dict(text=titre, x=0.5), margin=dict(t=90, r=24, b=60, l=70), **kw)

def courbe(cell):
    d = cell.soh.dropna(subset=["soh_percent"]).sort_values("cycle_number")
    return d["cycle_number"].to_numpy(float), d["soh_percent"].to_numpy(float)

def erreurs(modele, cell):
    n, y = courbe(cell)
    p = np.asarray(modele.predict_soh(cell.temperature_degC, cell.c_rate), float)
    e = p[np.clip(n.astype(int), 1, len(p)) - 1] - y
    return e, float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))

print(CANDIDAT, "·", len(cells), "cellules ·", len(courbes_publiques()), "courbes publiques")'''

ENTRAINEMENT = '''t0 = time.time()
modele = build(CANDIDAT).fit(cells)
duree = time.time() - t0
print(f"ajustement : {duree:.0f} s · perte finale {getattr(modele, 'perte_', float('nan')):.2e}")

fig = go.Figure()
for nom, attr in (("pré-entraînement", "histo_pretrain_"), ("ajustement", "historique_")):
    h = getattr(modele, attr, None)
    if h:
        fig.add_trace(go.Scatter(x=[e for e, _ in h], y=[v for _, v in h],
                                 mode="lines", name=nom))
mise_en_forme(fig, f"Convergence · {CANDIDAT}", 420, yaxis_type="log",
              xaxis_title="époque", yaxis_title="perte (MSE pondérée)")
fig.show()'''

TRAJECTOIRES = '''fig = go.Figure()
n_grille = np.arange(1, 12001)
lignes = []
for c in cells:
    n, y = courbe(c)
    p = np.asarray(modele.predict_soh(c.temperature_degC, c.c_rate), float)
    e, rmse, mae = erreurs(modele, c)
    lignes.append(dict(cellule=f"{c.temperature_degC}°C/{c.c_rate}C", n=len(n),
                       rmse=round(rmse, 3), mae=round(mae, 3),
                       biais=round(float(e.mean()), 3), err_max=round(float(np.abs(e).max()), 2)))
    fig.add_trace(go.Scattergl(x=n, y=y, mode="lines", name=f"{c.temperature_degC}°C/{c.c_rate}C",
                               line=dict(color=COULEUR[c.cell_id], width=3), opacity=0.45,
                               legendgroup=c.cell_id))
    fig.add_trace(go.Scattergl(x=n_grille, y=p, mode="lines", showlegend=False,
                               legendgroup=c.cell_id,
                               line=dict(color=COULEUR[c.cell_id], width=1.4,
                                         dash=DASH[c.c_rate])))
mise_en_forme(fig, f"Observé (épais) vs prédit (fin) · {CANDIDAT}", 560,
              xaxis_title="cycle_number", yaxis_title="SOH (%)",
              legend=dict(x=1.02, y=1.0))
fig.update_xaxes(range=[0, 6000])
fig.show()
pd.DataFrame(lignes)'''

RESIDUS = '''fig = make_subplots(rows=1, cols=2, subplot_titles=["résidu vs cycle", "résidu vs SOH observé"])
for c in cells:
    n, y = courbe(c)
    e, *_ = erreurs(modele, c)
    for col, x in ((1, n), (2, y)):
        fig.add_trace(go.Scattergl(x=x, y=e, mode="markers",
                                   marker=dict(size=2.5, color=COULEUR[c.cell_id]),
                                   name=f"{c.temperature_degC}°C/{c.c_rate}C",
                                   legendgroup=c.cell_id, showlegend=(col == 1)), row=1, col=col)
fig.add_hline(y=0, line=dict(color="black", dash="dot", width=1))
mise_en_forme(fig, f"Résidus (prédit - observé) · {CANDIDAT}", 460, legend=dict(x=1.02, y=1.0))
fig.update_yaxes(title_text="points de SOH", row=1, col=1)
fig.update_xaxes(title_text="cycle_number", row=1, col=1)
fig.update_xaxes(title_text="SOH observé (%)", row=1, col=2)
fig.show()'''

LOCO = '''# Leave-one-condition-out : la seule mesure qui compte pour une condition sans données.
from my_model.candidates import build as _build

resultats = []
for held in cells:
    train = [c for c in cells if c.cell_id != held.cell_id]
    for nom in (CANDIDAT, "baseline", "master"):
        t0 = time.time()
        m = _build(nom).fit(train)
        _, rmse, mae = erreurs(m, held)
        resultats.append(dict(candidat=nom, tenue_a_l_ecart=f"{held.temperature_degC}°C/{held.c_rate}C",
                              rmse=round(rmse, 3), mae=round(mae, 3), secondes=round(time.time() - t0)))
loco = pd.DataFrame(resultats)

piv = loco.pivot_table(index="tenue_a_l_ecart", columns="candidat", values="rmse")
fig = go.Figure()
for nom in piv.columns:
    fig.add_trace(go.Bar(x=piv.index, y=piv[nom], name=nom))
mise_en_forme(fig, f"LOCO · RMSE par condition tenue à l'écart", 440, barmode="group",
              yaxis_title="RMSE (points de SOH)")
fig.show()

moyennes = loco.groupby("candidat")[["rmse", "mae"]].mean().round(3)
base = moyennes.loc["baseline"]
moyennes["rmse_rel"] = (moyennes.rmse / base.rmse).round(3)
moyennes["mae_rel"] = (moyennes.mae / base.mae).round(3)
moyennes'''

SURFACE = '''# Le modèle répond-il vraiment à T et C, ou sort-il la même courbe partout ?
T_GRILLE = np.arange(25, 56, 2.5)
C_GRILLE = [0.5, 0.75, 1.0]
CIBLES = [1000, 2000, 4000]

surf = []
for T in T_GRILLE:
    for C in C_GRILLE:
        p = np.asarray(modele.predict_soh(float(T), float(C)), float)
        for k in CIBLES:
            surf.append(dict(T=float(T), C=C, cycle=k, soh=round(float(p[k - 1]), 2)))
surf = pd.DataFrame(surf)

fig = make_subplots(rows=1, cols=len(CIBLES),
                    subplot_titles=[f"cycle {k}" for k in CIBLES], shared_yaxes=True)
for j, k in enumerate(CIBLES, start=1):
    for C in C_GRILLE:
        s = surf[(surf.cycle == k) & (surf.C == C)]
        fig.add_trace(go.Scatter(x=s["T"], y=s.soh, mode="lines+markers", name=f"{C}C",
                                 legendgroup=str(C), showlegend=(j == 1)), row=1, col=j)
mise_en_forme(fig, f"Réponse en (T, C) · {CANDIDAT}", 420, legend=dict(x=1.02, y=1.0))
fig.update_xaxes(title_text="température (°C)")
fig.update_yaxes(title_text="SOH (%)", row=1, col=1)
fig.show()

etendue = surf.groupby("cycle").soh.agg(["min", "max"])
etendue["etendue"] = (etendue["max"] - etendue["min"]).round(2)
print("étendue de SOH sur la grille (T, C) — si elle est nulle, le modèle ignore les conditions :")
etendue.round(2)'''

MD = {
    "titre": "# Évaluation du candidat `{nom}`\n\nPerte d'entraînement, trajectoires, résidus, erreur en "
             "leave-one-condition-out et réponse en (T, C).\n\nRegénérer : "
             "`python scripts/make_model_notebooks.py {nom}`",
    "conv": "## 1 · Convergence\n\nPerte pondérée par époque. Un pré-entraînement, s'il y en a un, "
            "apparaît comme une courbe séparée.",
    "traj": "## 2 · Trajectoires\n\nCourbe mesurée (épaisse) contre courbe prédite (fine), même couleur "
            "par cellule. Le tableau donne RMSE, MAE, biais et erreur maximale.",
    "res": "## 3 · Résidus\n\nUn résidu structuré en fonction du cycle ou du SOH signale un défaut de "
           "forme, pas du bruit.",
    "loco": "## 4 · Leave-one-condition-out\n\nChaque condition est retirée à tour de rôle, le modèle "
            "est réajusté sur les cinq autres et doit prédire celle qui manque — comparé à la baseline "
            "et au modèle livré.",
    "surf": "## 5 · Réponse en (T, C)\n\nSOH prédit sur une grille de conditions. Une étendue nulle "
            "signifierait que le modèle sort la même courbe partout, ce qui est un échec de modélisation.",
}


def cellule(source, code=True):
    lignes = source.split("\n")
    s = [x + "\n" for x in lignes[:-1]] + [lignes[-1]]
    if code:
        return {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8],
                "metadata": {}, "outputs": [], "source": s}
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": s}


def notebook(nom):
    cells = [cellule(MD["titre"].format(nom=nom), False),
             cellule(SETUP.format(nom=nom)),
             cellule(MD["conv"], False), cellule(ENTRAINEMENT),
             cellule(MD["traj"], False), cellule(TRAJECTOIRES),
             cellule(MD["res"], False), cellule(RESIDUS),
             cellule(MD["loco"], False), cellule(LOCO),
             cellule(MD["surf"], False), cellule(SURFACE)]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"}}}


def main():
    noms = sys.argv[1:] or DEFAUT
    os.makedirs(os.path.join(ROOT, "analysis"), exist_ok=True)
    for nom in noms:
        f = os.path.join(ROOT, "analysis", f"modele_{nom.replace('+', '_')}.ipynb")
        with open(f, "w") as fh:
            json.dump(notebook(nom), fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        print(f"écrit {os.path.relpath(f, ROOT)}")


if __name__ == "__main__":
    main()
