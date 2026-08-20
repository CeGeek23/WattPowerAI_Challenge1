#!/usr/bin/env python3
"""Catenaro & Onori (Mendeley 10.17632/kxsbr4x3j2.2) -> capacite restituee vs (T, C-rate).

Le jeu ne contient AUCUN vieillissement : une seule decharge galvanostatique
par (cellule, C-rate, temperature). On en extrait la capacite de decharge
Q = integrale |I| dt sur l'etape a courant negatif, ce qui quantifie la part
REVERSIBLE de la capacite (effet cinetique de T et du C-rate), pas du fade.

Sortie : pretrain_catenaro_capacite.csv
  source,cell_id,chemistry,nominal_Ah,T_degC,c_rate,capacite_Ah
"""
import csv, re, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import openpyxl

NOMINAL_AH = {"LFP": 2.500, "NCA": 3.350, "NMC": 4.850}  # A123 ANR26650m1-B, Panasonic NCR-18650B, LG INR21700-M50
SOURCE = "catenaro2021"
PAT = re.compile(r"^(LFP|NCA|NMC)_(k\d)_([0-9_]+)C_(\d+)degC\.xlsx$")


def parse_name(name):
    m = PAT.match(name)
    if not m:
        return None
    chem, k, crate_s, t_s = m.groups()
    return chem, k, float(crate_s.replace("_", ".").rstrip(".")), float(t_s)


def discharge_capacity_ah(path):
    """Q_dch (Ah), T_moy de la decharge (degC), duree (s), I_moy (A)."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    it = ws.iter_rows(values_only=True)
    next(it)  # entete
    steps = {}  # step_index -> [Ah, dt_total, sum(I*dt), sum(T*dt)]
    prev_t = None
    for r in it:
        try:
            t = float(r[1]); si = r[3]; i = float(r[5]); temp = float(r[6])
        except (TypeError, ValueError):
            continue
        dt = 0.0 if prev_t is None else t - prev_t
        prev_t = t
        if dt <= 0 or dt > 60 or i >= 0:
            continue
        s = steps.setdefault(si, [0.0, 0.0, 0.0, 0.0])
        s[0] += -i * dt / 3600.0
        s[1] += dt
        s[2] += -i * dt
        s[3] += temp * dt
    wb.close()
    if not steps:
        return None
    si = max(steps, key=lambda k: steps[k][0])
    ah, dur, iw, tw = steps[si]
    return ah, tw / dur, dur, iw / dur


def job(path):
    meta = parse_name(path.name)
    if meta is None:
        return None
    chem, k, crate, t_amb = meta
    res = discharge_capacity_ah(path)
    if res is None:
        return None
    ah, t_moy, dur, i_moy = res
    return dict(source=SOURCE, cell_id=f"{chem}_{k}", chemistry=chem,
                nominal_Ah=NOMINAL_AH[chem], T_degC=t_amb, c_rate=crate,
                capacite_Ah=round(ah, 5),
                _T_cell_moy=round(t_moy, 2), _duree_s=round(dur, 1),
                _I_moy_A=round(i_moy, 4), _fichier=path.name)


def main(src_dir, out_csv, out_csv_full):
    files = sorted(p for p in Path(src_dir).glob("*.xlsx") if PAT.match(p.name))
    print(f"{len(files)} fichiers de decharge")
    rows = []
    with ProcessPoolExecutor() as ex:
        for n, r in enumerate(ex.map(job, files, chunksize=2), 1):
            if r:
                rows.append(r)
            if n % 25 == 0:
                print(f"  {n}/{len(files)}", flush=True)
    rows.sort(key=lambda r: (r["chemistry"], r["cell_id"], r["T_degC"], r["c_rate"]))
    cols = ["source", "cell_id", "chemistry", "nominal_Ah", "T_degC", "c_rate", "capacite_Ah"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    with open(out_csv_full, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["_T_cell_moy", "_duree_s", "_I_moy_A", "_fichier"])
        w.writeheader(); w.writerows(rows)
    print(f"ecrit {len(rows)} lignes -> {out_csv}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
