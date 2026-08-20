"""Build pretrain_che.csv from the Che et al. (2023) Mendeley dataset.

Columns: source,cell_id,chemistry,nominal_Ah,T_degC,c_rate_charge,
         c_rate_discharge,cycle,soh_pct

IMPORTANT CAVEAT (from the official Mendeley description, v9):
  "the charge capacity in the datasets is the amount for the ampere-hour
   throughput in charging mode ... For simplicity and easy use, the SOH curves
   are normalized by the charge capacity for verification in this work."
  => The single `Capacity` array per cell is the CHARGE capacity, not the
     discharge capacity. No discharge capacity is shipped in the .mat files,
     so an exact conversion is impossible. For Dataset3/Dataset4 (CC-CV charge,
     CC discharge) charge ~= discharge capacity to within coulombic efficiency
     (<1% bias). For Dataset1 the charge capacity also absorbs the charging
     pulses of the dynamic discharge profile and is NOT a discharge capacity.
"""
import csv, re, sys
from pathlib import Path
import h5py, numpy as np

CACHE = Path(sys.argv[1])
OUT = Path(sys.argv[2])

# --- per-dataset metadata; see notes in the final report -------------------
META = {
    "Dataset1_Dataset2": dict(chemistry="NMC_pouch", nominal_Ah=float("nan"),
                              normalized=True),
    "Dataset3": dict(chemistry="NMC_pouch", nominal_Ah=100.0, normalized=False),
    "Dataset4": dict(chemistry="NMC_pouch", nominal_Ah=100.0, normalized=False),
}

def mstr(f, ref):
    return ''.join(chr(int(c)) for c in f[ref][()].flatten())

def parse_profile(top, cell, wp):
    """Return (T_degC, c_charge, c_discharge) parsed from the profile label."""
    wp = wp.replace('℃', 'C').strip()
    T = c_ch = c_dis = ''
    m = re.search(r'_(\d+(?:\.\d+)?)C$', wp)          # trailing temperature
    if m:
        T = float(m.group(1))
        head = wp[:m.start()]
        # Dataset3: "1c_1cC" -> charge 1C, discharge 1C ; "0.3c_1cC"
        m2 = re.match(r'^(\d*\.?\d+)c_(\d*\.?\d+)cC$', head)
        # Dataset4: "0.5C" / "1C" -> symmetric charge = discharge
        m3 = re.match(r'^(\d*\.?\d+)C$', head)
        if m2:
            c_ch, c_dis = float(m2.group(1)), float(m2.group(2))
        elif m3:
            c_ch = c_dis = float(m3.group(1))
    return T, c_ch, c_dis

rows = []
for fn, top in [("Dataset1_Dataset2.mat", "Dataset1_Dataset2"),
                ("Dataset3.mat", "Dataset3"),
                ("Dataset4.mat", "Dataset4")]:
    meta = META[top]
    with h5py.File(CACHE / fn, 'r') as f:
        g = f[top]
        n = g['cell'].shape[0]
        for i in range(n):
            name = mstr(f, g['cell'][i, 0])
            wp = mstr(f, g['Workingprofile'][i, 0])
            cap = np.ravel(f[g['Capacity'][i, 0]][()]).astype(float)

            if top == "Dataset1_Dataset2":
                # 15 fresh cells (Dataset1) then 15 second-life cells (Dataset2)
                ds = "Dataset1" if i < 15 else "Dataset2"
                soh = 100.0 * cap                     # already normalized to cycle 1
                nominal = ''
            else:
                ds = top
                nominal = meta["nominal_Ah"]
                soh = 100.0 * cap / nominal

            T, c_ch, c_dis = parse_profile(top, name, wp)
            cell_id = f"che_{ds}_{name.replace(' ', '')}"
            for k, v in enumerate(soh, start=1):
                if not np.isfinite(v):
                    continue
                rows.append((f"che2023_{ds}", cell_id, meta["chemistry"],
                             nominal, T, c_ch, c_dis, k, round(float(v), 6)))

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(["source", "cell_id", "chemistry", "nominal_Ah", "T_degC",
                "c_rate_charge", "c_rate_discharge", "cycle", "soh_pct"])
    w.writerows(rows)
print(f"wrote {len(rows)} rows -> {OUT}")
