# TechArena 2026 - Challenge 1: SOH fade forecasting

Predicting the SOH(cycle) trajectory of a 102 Ah LFP cell at any operating
point in 25-55 °C x 0.5-1.0 C, including combinations never observed.

**Model**: one shared fade shape, one time scale per operating point
(`my_model/model_template.py`), pre-trained on open cycling datasets and
fine-tuned on the released 102 Ah cells. The pre-trained priors ship as
`my_model/pretrained.json` (482 bytes, plain text, loaded offline).

The exploratory analysis that motivated the model (notebook and evaluation
harness) lives in the team repository and is not part of this submission; the
findings it produced are summarised in section 1 below.

---

## 0. Pre-training on open data

All three datasets are **CC BY 4.0**, downloaded by the reproducible scripts in
`scripts/pretrain_data/` into a local cache (not shipped). What matters is what
each one could and could not contribute - a public cell is not a 102 Ah
prismatic cell, and transferring the wrong quantity transfers noise.

| dataset | cells | what transferred | what was rejected, and why |
| --- | --- | --- | --- |
| Wheeler et al. 2025 [1] | 20 LFP 18650 | **cell-to-cell scatter of ln τ ≈ 6.6%** (within true same-protocol replicate groups; pooling by nominal (T,C) without distinguishing protocols inflates this to ~16%) | fade shape: shared-shape RMSE **3.43** SOH points, the 7 protocols are not one family |
| Che et al. 2023 [2] | 17 NMC pouch | **Arrhenius slope 2.58** (Ea ≈ 21 kJ/mol), from 25/35/55 °C | shape (RMSE 1.51, above the 1.5 gate); C-rate exponent **+1.51**, contradicted by the target cells (-0.06) |
| Catenaro & Onori 2021 [3] | 18 (LFP/NCA/NMC) | **reversible capacity vs T** (LFP 1C: 94.9% at 5 °C, 100.1% at 25 °C, 101.8% at 35 °C), which justifies modelling a(T) | everything else: the dataset has **no ageing at all**, only 15-24 characterisation discharges per cell |

Two of these numbers are things the released dataset **cannot** provide. It has
one cell per condition, so no cell-to-cell variance; the ~6.6% figure sets the GP
nugget, which was otherwise guesswork. And the initial-SOH rise with temperature
is reversible kinetics, not ageing - Catenaro measures it independently, which is
what licenses fitting a(T) rather than a single constant.

The Arrhenius slope is the cleanest transfer: **2.58 measured on 17 NMC pouch
cells against 2.81 fitted on the 6 target LFP cells**, two chemistries agreeing
within 9%. It is shipped as the prior, so a reduced-budget run that cannot
identify the slope itself still gets a defensible one.

`scripts/pretrain.py` regenerates `pretrained.json` and **refuses to export what
it cannot justify**: the shape is exported only if the shared-shape fit stays
under 1.5 SOH points on the public cells, the slope only if at least three
temperatures are covered, the C exponent never. The rejection reasons are written
into the file's `meta.ecarte` field.

[1] Wheeler, W., Venet, P., Bultel, Y. & Sari, A. (2025). *Aging study on twenty
A123 18650 Graphite/LFP 1.1 Ah cells*, V2, Recherche Data Gouv,
doi:10.57745/OLBXKT. CC BY 4.0. Paper: *Sci Data* **12**, 392 (2025),
doi:10.1038/s41597-025-04712-7.
[2] Che, Y. et al. (2024). *Battery aging datasets for "Increasing generalization
capability of battery health estimation using continual learning"*, V9, Mendeley
Data, doi:10.17632/n3b54nsw8m.9. CC BY 4.0. Paper: *Cell Reports Physical
Science* **4**(12), 2023.
[3] Catenaro, E. & Onori, S. (2021). *Experimental data of three lithium-ion
batteries under galvanostatic discharge tests at different C-rates and operating
temperatures*, V2, Mendeley Data, doi:10.17632/kxsbr4x3j2.2. CC BY 4.0. Paper:
*Data in Brief* **35**, 106894.

---

## 1. What the data says

Six cells, one per condition, no replicates: 25 °C/0.5C, 25 °C/1C, 35 °C/1C,
45 °C/0.5C, 45 °C/1C, 55 °C/1C. The grid has two holes (35 °C/0.5C and
55 °C/0.5C) and the cells stop between cycle 1375 and 5356.

Three findings drove the model design.

**The SOH labels are almost noiseless.** Around a rolling median, the labels
scatter by 0.01-0.09 SOH point (MAE). Whatever error the model makes is
modelling error, not measurement noise - so the shape of the fade curve is
worth getting right.

**The apparent diversity of fade shapes is an artefact of the observation
window.** Fitting a free power law `a - b·n^p` cell by cell gives exponents
from 0.38 (25 °C/0.5C, decelerating) to 1.72 (45 °C/0.5C, accelerating) with
no pattern in T or C. But rescaling the cycle axis collapses all six cells
onto a single curve: the number of cycles needed to reach a given SOH,
divided by the number needed to reach 95%, agrees to **within 8% across all
six cells**, from 100% down to 70% SOH:

| SOH reached | 100% | 98% | 95% | 90% | 85% | 80% | 75% | 70% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n / n(95%), min over cells | 0.10 | 0.27 | 1.00 | 2.46 | 4.05 | 5.23 | 5.90 | 6.53 |
| n / n(95%), max over cells | 0.12 | 0.35 | 1.00 | 2.90 | 4.88 | 5.83 | 6.58 | 6.93 |

A cell that "decelerates" is simply one that was only followed through the
early, diffusion-limited part of the curve; a cell that "accelerates" is one
that was followed past the knee. The exponent varies because the window
varies, not because the physics does.

**Temperature and C-rate act on the time scale, not on the shape - and the
temperature response is not monotone.** Fitted time scales (cycles per unit
of reduced time): 1392 at 25 °C/0.5C, 1228 at 25 °C/1C, 1299 at 35 °C/1C,
831 at 45 °C/0.5C, 890 at 45 °C/1C, 507 at 55 °C/1C. Ageing is *slowest*
around 35 °C, not at the cold end, and the C-rate effect changes sign with
temperature (1C ages ~13% faster at 25 °C, ~7% slower at 45 °C). With one
cell per condition, part of that inversion may be cell-to-cell scatter; the
model is built so that neither effect has to be assumed.

---

## 2. The model

```text
SOH(n | T, C) = a - L( n / tau(T, C) )

L(u) = A (1 - exp(-u^p)) + B u^q          shared shape, saturating SEI + knee
ln tau(T, C) = w0 + w1 x + w2 ln(C/0.75) + w3 x^2 + GP(x, ln C),  x = 1000/T_K
```

**Shape `L`.** Fitted jointly on every training cell in a single least-squares
problem: the shape parameters are shared, and each cell contributes its own
`(a_i, tau_i)`. A shallow cell therefore still informs the early part of the
curve and *borrows* the knee from the deeper cells, which is exactly what the
collapse above licenses. Each cell is weighted equally (residuals scaled by
1/sqrt(number of labels)) so that the 5356-cycle cell does not drown the
988-cycle one. Fitted values: A = 14.86, p = 0.639, B = 0.347, q = 2.711.

**Starting SOH `a(T)`.** SOH starts above 100 - the cells exceed the 102 Ah
nominal - and it rises slightly with temperature (+0.58 point across 25-55 °C,
r = +0.48). That is reversible kinetics, not ageing, which Catenaro & Onori
measure independently on LFP. Modelling it as a ridged linear function of
1000/T rather than a single constant is worth 2.6% on leave-one-out; the ridge
collapses it back to a constant when the grid cannot identify a slope.

**Time scale `tau`.** An Arrhenius trend carries the physics and extrapolates;
a Gaussian-process residual over `(1000/T_K, ln C)` absorbs what the trend
cannot explain, with a length scale of ~12 °C. Its nugget is set to 6%, between
the 3% that cross-validation prefers and the ~6.6% of cell-to-cell scatter
measured within true same-protocol replicate groups on the public replicates
(pooling across protocols that merely share a nominal (T,C) inflates this to
~16%) - cross-validation cannot see that scatter, since the
released grid has no replicates at all. The fitted
trend gives an apparent activation energy of 23.0 kJ/mol. The quadratic term
lets the data express the non-monotone temperature response; its prior is
zero (plain Arrhenius) and *forcing* a curvature in was tested and clearly
hurt (mean relative RMSE 0.55 -> 1.08), so it stays data-driven.

**Everything is regularised toward physical priors**, and the trend only
estimates the terms the training grid can identify (slope needs 2 distinct
temperatures, C exponent 2 distinct C-rates and 3 conditions, curvature 3
temperatures and 4 conditions). The others keep their prior coefficient. This
is what makes the reduced-budget replay behave: with a single training cell
the model still predicts a temperature and a C-rate dependence, instead of
collapsing to one curve for every operating point (which is what the
reference baseline does below 3 cells).

**Tail.** Past the knee, `B u^q` would send the trajectory through zero long
before cycle 12000 and the whole tail would sit on the clipping floor. The
loss saturates smoothly at 20% SOH instead - sharply enough to leave the
fitted range untouched (< 0.6 SOH point). No released cell was followed below
58% SOH, so anything under that is an extrapolation nobody can check; a bounded
guess is better than a free fall.

**Data hygiene inside `fit()`**: labels are de-duplicated by cycle number, and
a median filter drops the few partial cycles that survive the framework's own
cleaning (e.g. an isolated 42.8% between neighbours at 58.7%).

`fit()` uses `cell.soh` only. It never reads the raw time series, needs no
network, is deterministic, and the whole pipeline (loading the 559 MB dataset
included) runs in **2 seconds**; the pickled state is 708 bytes.

An effective-temperature variant was tested and rejected: measured cell
temperature deviates from the setpoint by +1.6 to +12.0 °C (self-heating,
strongest at low setpoint), and rebuilding the Arrhenius law on the measured
temperature does linearise the trend - but predicting that deviation from
(T, C) at inference time costs more than it gains (relative RMSE 0.54 -> 0.62).

---

## 3. Validation methodology

No scoring script is provided, and with six cells any single split is fragile.
Three complementary views are used, all measured against the reference
baseline (`model_example.py` = 1.00, lower is better).

**a. Leave-one-condition-out.** Hold out one of the six cells entirely, fit on
the other five, predict the held-out condition and score against its full
measured trajectory. This is the honest estimate for an operating point with
no data. It is *pessimistic* for the actual task at the extremes: holding out
25 °C or 55 °C forces extrapolation beyond the training range, which will not
happen at scoring time (the hidden points lie inside 25-55 °C). The interior
conditions (35 °C/1C, 45 °C/0.5C, 45 °C/1C) are the representative case, and
are reported separately.

**b. In-sample fidelity.** Fit on all six cells, score on the same six. This
is not a generalisation estimate - it is the proxy for the *sibling* cells of
the released conditions that the hidden test set contains. What it measures is
whether the model can represent a curve it has seen at all, which the
baseline's fixed exponent cannot.

**c. Reduced-budget replay.** The same leave-one-condition-out protocol, but
training only on early-life cycles (<= 1500, 800, 400, 200) and, separately,
on two cells only. Evaluation is always against the full measured trajectory.
This mirrors the organizers' automatic data-efficiency rerun.

### Results (relative to the baseline, lower is better)

| Protocol | rel. RMSE | rel. MAE | interior conditions only |
| --- | --- | --- | --- |
| In-sample (sibling proxy) | **0.24** | 0.21 | 0.29 |
| Leave-one-condition-out, all cycles | **0.58** | 0.47 | 0.66 |
| LOCO, cycles <= 1500 | 0.67 | 0.63 | 0.84 |
| LOCO, cycles <= 800 | 0.52 | 0.44 | 0.64 |
| LOCO, cycles <= 400 | 0.71 | 0.56 | 0.82 |
| LOCO, cycles <= 200 | 0.55 | 0.43 | 0.68 |

Absolute numbers, leave-one-condition-out on all cycles (SOH points RMSE):
0.19 at 25 °C/0.5C, 3.80 at 25 °C/1C, 2.99 at 35 °C/1C, 3.14 at 45 °C/0.5C,
0.29 at 45 °C/1C, 4.36 at 55 °C/1C - mean 2.46 against 4.28 for the baseline.
In-sample: mean 0.66 against 2.76.

The reduced-budget rows are where the open-data priors earn their place: with
400 or 200 cycles per training cell the model cannot identify the Arrhenius
slope itself, and falls back on the one measured across 17 public cells.

Hyperparameters (GP amplitude, nugget, length scale, prior weights) were
chosen on a coarse grid over the union of these protocols, not on any single
one; the top configurations differ by less than 0.01 in mean relative RMSE, so
the ranking is not sensitive to that choice.

### Known weaknesses

- **35 °C is anomalously slow-ageing** relative to its neighbours. Held out,
  it cannot be recovered by interpolating 25 °C and 45 °C: this single
  condition carries most of the leave-one-out error in the shallow-budget
  runs. With one cell per condition it is impossible to tell a real
  temperature optimum from cell-to-cell scatter.
- **Two training cells only**, at 25 °C/1C and 45 °C/0.5C, is the one scenario
  where the model loses to the baseline (4.97 vs 3.55), for the same reason.
  The reverse pair (35 °C/1C + 55 °C/1C) wins clearly (2.53 vs 3.81).
- **Beyond 58% SOH** the shape is unconstrained by data; the saturating tail
  is a safety device, not a prediction.
- **No replicates**, so no cell-to-cell variance estimate. The GP nugget
  assumes 6% scatter on `ln tau` (between the 3% cross-validation prefers and
  the ~6.6% measured within same-protocol replicate groups on public data);
  if siblings scatter more, the model is slightly over-confident at the
  released conditions.

The evaluation harness that produced this table is not part of the submission
(it needs the released cells as ground truth); it re-runs `fit()` and
`predict_soh()` exactly as the framework does, so the numbers are reproducible
from this model file plus the released dataset.

---

## 4. Running it

```bash
python run_model.py --model train --input dataset/target --output-dir output
python run_model.py --model test  --input input.csv       --output-dir output
python validate_submission.py          # prints SUBMISSION READY
```

`my_model/__init__.py` points `ActiveModel` at `MyModel`
(`my_model/model_template.py`). The model needs `numpy`, `scipy` and the
framework's `pandas`; nothing else. The only shipped artefact is
`my_model/pretrained.json`, read from disk at import time - no network access at
any point.

The dataset is not shipped with the submission: point `--input` at the folder
containing the `102Ah_<T>degC_<C>_cell<N>` directories (locally,
`./scripts/get_dataset.sh` fetches these into `dataset/target/`; the sibling
`dataset/wheeler/`, `dataset/che/`, `dataset/catenaro/` hold the open
pre-training sources from §0 and are not read by `run_model.py`).
