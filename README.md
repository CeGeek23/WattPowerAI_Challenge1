# TechArena 2026 - Challenge 1: SOH fade forecasting

Predicting the SOH(cycle) trajectory of a 102 Ah LFP cell at any operating
point in 25-55 °C x 0.5-1.0 C, including combinations never observed.

**Model**: one shared fade shape, one time scale per operating point
(`my_model/model_template.py`), pre-trained on open cycling datasets and
fine-tuned on the released 102 Ah cells. The pre-trained priors ship as
`my_model/pretrained.json` (plain text, loaded offline, no network).

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
| Wheeler et al. 2025 [1] | 20 LFP 18650 | **cell-to-cell scatter of ln τ ≈ 6.6%** (within true same-protocol replicate groups; pooling by nominal (T,C) without distinguishing protocols inflates this to ~16%) | fade shape: shared-shape RMSE **3.52** SOH points, the 7 protocols are not one family |
| Che et al. 2023 [2] | 17 NMC pouch | **Arrhenius slope 2.56** (Ea ≈ 21 kJ/mol), from 25/35/55 °C; and the **sign** of the temperature curvature | shape (median RMSE 1.48 but p90 **10.4** points - it describes half the cells and misses the rest); C-rate exponent **+1.51**, contradicted by the target cells (-0.06); the curvature *magnitude* (-7.19, chemistry-specific) |
| Catenaro & Onori 2021 [3] | 18 (LFP/NCA/NMC) | **reversible capacity vs T** (LFP 1C: 94.9% at 5 °C, 100.1% at 25 °C, 101.8% at 35 °C), which justifies modelling a(T) | everything else: the dataset has **no ageing at all**, only 15-24 characterisation discharges per cell |

Two of these numbers are things the released dataset **cannot** provide. It has
one cell per condition, so no cell-to-cell variance; the ~6.6% figure sets the GP
nugget, which was otherwise guesswork. And the initial-SOH rise with temperature
is reversible kinetics, not ageing - Catenaro measures it independently, which is
what licenses fitting a(T) rather than a single constant.

The Arrhenius slope is the cleanest transfer: **2.56 measured on 17 NMC pouch
cells against 2.85 fitted on the 6 target LFP cells**, two chemistries agreeing
within 11%. It is shipped as the prior, so a reduced-budget run that cannot
identify the slope itself still gets a defensible one.

`scripts/pretrain.py` regenerates `pretrained.json` and **refuses to export what
it cannot justify**. The slope needs at least three covered temperatures. The
shape needs both a median shared-shape RMSE under 1.5 SOH points *and* a p90
under 5: on Che the median is 1.48 but the p90 is 10.4, so the shape describes
half the public cells and misses the other half - importing it degraded every
truncated protocol (`loco-800` 0.49 -> 0.73), and it is rejected. The C exponent
requires an LFP chemistry, which Che is not. The curvature is measured and
recorded (`meta.curv_public`) but not exported, because only its sign transfers
across chemistries - see §2. Every rejection reason is written into the file's
`meta.ecarte` field, so the file states what it declined to carry and why.

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
of reduced time): 1370 at 25 °C/0.5C, 1192 at 25 °C/1C, 1286 at 35 °C/1C,
813 at 45 °C/0.5C, 875 at 45 °C/1C, 495 at 55 °C/1C. Ageing is *slowest*
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
988-cycle one. Within a cell, points are weighted by fade depth relative to
that cell's own deepest point, floored at 0.8: a cell carries thousands of
labels above 90% SOH and only a few hundred in the knee, so uniform weighting
fits the shape to early life and extrapolates it into the region that is
actually scored. The floor keeps a cell that was only followed through early
life contributing all of its data, which is what makes this safe under the
reduced-budget replay (it improves every truncated protocol; see §3).

**Starting SOH `a(T)`.** SOH starts above 100 - the cells exceed the 102 Ah
nominal - and it rises slightly with temperature (+0.39 point across 25-55 °C
as shipped, shrunk by the ridge from a raw spread of 1.45 points). That is
reversible kinetics, not ageing, which Catenaro & Onori measure independently
on LFP. Modelling it as a ridged linear function of
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
trend gives an apparent activation energy of 23.7 kJ/mol.

The quadratic term carries the non-monotone temperature response, and it is the
one place where a *sign* transfers from open data. At 1C the target's own time
scale is longer at 35 °C (1286 cycles) than at 25 °C (1192): a pure Arrhenius
cannot represent that optimum. Two independent estimates say the curvature is
negative - the six target cells converge to -1.37 on their own with a zero
prior, and Che's 17 cells give -7.19 through `scripts/pretrain.py`. The
magnitude does not transfer across chemistries and the ridge on this term is
too weak to correct an imported one, so the prior is set between the two
estimates (-3.0) rather than to either. It matters most exactly where the term
is not identifiable, i.e. under reduced budget: against a zero prior it moves
`deep` 0.73 -> 0.66, `loco-800` 0.49 -> 0.41 and `deep-800` 0.69 -> 0.56, with
`loco-400` the only protocol that regresses (0.62 -> 0.63).

**The knee position is marginalised, not asserted.** Under squared-error loss
the optimal prediction is not the best-fit curve but its expectation, and a
sharp knee placed at the wrong cycle is expensive: held out, the predicted
cycle at 70% SOH misses by up to +31%, while the model's own assumed scatter
was 6%. So `predict_soh` returns `a - E[L(n/tau)]` with
`ln tau ~ N(ln tau_hat, s^2)`, integrated over 15 Gauss-Hermite nodes (4 ms per
call). `s` is the Gaussian process's *posterior* standard deviation - the
quantity the fit previously computed and threw away - plus the cell-to-cell
term. It is not a tuned knob: it is 0.08 at the released conditions, 0.10 at
30 °C/0.7C and 0.14-0.17 in the two grid holes (35 °C and 55 °C at 0.5C), so
the curve is blurred only where the model genuinely does not know. Checked
against the six leave-one-out residuals, `sqrt(mean(z^2)) = 1.01` with
`z = residual/s` - calibrated, with nothing fitted to make it so.

`s` also carries the uncertainty of the trend coefficients themselves: four
terms fitted on five or six cells, previously treated as exact. Because
`tau = exp(phi . w)`, that uncertainty is *exactly* one more variance term on
`ln tau` - closed form, same quadrature, no extra cost and no sampling. Taken
whole it takes the calibration ratio from 0.97 to 0.69, i.e. wider than the
scatter of `tau` alone justifies, so half of it is used (`TREND_VAR_W = 0.5`,
ratio 0.79). That half is worth `deep` 0.66 -> 0.59, `loco` 0.45 -> 0.39 and
`profond` 0.51 -> 0.44, against `in-sample` 0.25 -> 0.27 - and that last cost
is largely an artefact of the proxy, which scores against the very cell that
was fitted and therefore contains no sibling scatter at all.

The blur is
switched off when the training set never reached the knee (fade depth < 20 SOH
points), because smearing the position of a knee whose shape is still a prior
adds variance without information; that gate is inert at full budget and is
what keeps the reduced-budget replays from regressing.

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
included) runs in **2 seconds**; the pickled state is 1042 bytes. One
`predict_soh` call costs 4 ms (15 quadrature nodes over 12000 cycles).

An effective-temperature variant was tested and rejected: measured cell
temperature deviates from the setpoint by +1.6 to +12.0 °C (self-heating,
strongest at low setpoint), and rebuilding the Arrhenius law on the measured
temperature does linearise the trend - but predicting that deviation from
(T, C) at inference time costs more than it gains (relative RMSE 0.54 -> 0.62).

---

## 3. Validation methodology

No scoring script is provided, and with six cells any single split is fragile.
Everything below is measured against the reference baseline
(`model_example.py` = 1.00, lower is better).

**Which cycles are scored is the decisive methodological choice, and it is easy
to get wrong.** The challenge asks for the trajectory "from cycle 1 down to 70%
SOH, including the knee-point", judged against cells "cycled deep through the
knee". But three of the six released cells stop at 93.1%, 84.8% and 72.6% SOH -
they never reach the knee. Scoring only where the released cells happen to have
labels therefore measures mostly early life, which is the easy part: it rates
this model at 0.58 while the knee region, which is what the task is about,
rates it at 0.84. The whole table below is reported over four windows for that
reason, and the shortfall is deliberately not hidden.

The *aggregation* matters as much as the window. `--model test` is invoked once
per evaluation cell, so a per-cell score is the likely scheme; a cell where the
model is 3x worse than the baseline then costs far more than a cell where it is
2x better buys back. Three aggregations are reported: ratio of mean RMSEs,
mean of per-cell ratios, and the worst single cell.

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

**c. Knee-region windows.** `profond` scores the full trajectory of only the
four cells that actually age past 75% SOH - the closest available stand-in for
a hidden cell followed to end of life. `deep` scores every cell but only where
SOH <= 80, isolating the knee itself.

**d. Reduced-budget replay.** The same leave-one-condition-out protocol, but
training only on early-life cycles (<= 800, <= 400) and, separately, on two
cells (`paires`, all 15 combinations) or one (`solo`). Evaluation is always
against the full measured trajectory. This mirrors the organizers' automatic
data-efficiency rerun, which cuts both cycles and cells.

### Results (relative to the baseline, lower is better)

| Protocol | ratio of means | mean of per-cell ratios | worst cell | abs. RMSE (SOH pts) |
| --- | --- | --- | --- | --- |
| In-sample (sibling proxy) | **0.27** | 0.26 | 0.40 | 0.74 vs 2.76 |
| Leave-one-condition-out | **0.39** | 0.41 | 1.32 | 1.66 vs 4.27 |
| `profond` (deep cells, full life) | **0.44** | 0.57 | 1.32 | 2.38 vs 5.37 |
| `deep` (SOH <= 80) | **0.59** | 0.89 | 2.22 | 4.61 vs 7.88 |
| LOCO, cycles <= 800 | 0.40 | 0.44 | 0.83 | 1.47 vs 3.67 |
| LOCO, cycles <= 400 | 0.64 | 0.58 | 1.29 | 4.46 vs 6.99 |
| `deep`, cycles <= 800 | 0.54 | 0.60 | 1.30 | 3.79 vs 6.96 |
| Two training cells (15 pairs) | 0.67 | 0.89 | 7.64 | 3.44 vs 5.12 |
| One training cell (6 folds) | 0.58 | 1.01 | 10.86 | 3.53 vs 6.09 |

Per-condition leave-one-out RMSE (SOH points): 0.13 at 25 °C/0.5C, 2.67 at
25 °C/1C, 3.53 at 35 °C/1C, 1.31 at 45 °C/0.5C, 0.32 at 45 °C/1C, 2.02 at
55 °C/1C - mean 1.66 against 4.27 for the baseline.

The two rows that matter most are `profond` and `deep`, and the honest reading
is that the knee is still the weak spot: 0.59 there against 0.39 over the full
trajectory. Per cell in the knee window the model is at 0.44, 0.59 and 0.29 on
25 °C/1C, 45 °C/0.5C and 55 °C/1C, but **2.22 on 35 °C/1C** - the anomalously
slow-ageing condition, which cannot be recovered by interpolating its
neighbours. That single cell is the whole gap between 0.59 and ~0.42.

The two- and one-cell rows have a large *worst cell* because a single training
condition cannot identify the tau law at all; the means stay below the baseline.
Under the budget the organizers actually replay - fewer cells **and** early-life
cycles only - the pairs score 0.85 (<= 800 cycles) and 0.57 (<= 400).

The reduced-budget rows are where the open-data priors earn their place: with
400 cycles per training cell the model cannot identify the Arrhenius slope
itself, and falls back on the one measured across 17 public cells.

Hyperparameters (GP amplitude, nugget, length scale, prior weights, and the
shape-fit depth weighting) were chosen on a coarse grid over the union of these
protocols, never on a single one. That discipline is not decorative: shifting
the C-rate prior toward the value fitted on Wheeler's LFP cells improves
leave-one-out (2.46 -> 2.28) while degrading both reduced-budget rows
monotonically, and was rejected for that reason.

### Known weaknesses

- **35 °C is anomalously slow-ageing** relative to its neighbours. Held out,
  it cannot be recovered by interpolating 25 °C and 45 °C: this single
  condition carries most of the leave-one-out error in the shallow-budget
  runs. With one cell per condition it is impossible to tell a real
  temperature optimum from cell-to-cell scatter.
- **35 °C is also where the knee-region score is lost.** In the `deep` window
  the model is at 0.44, 0.59 and 0.29 on the other three deep cells but 2.22
  there. Marginalising the knee position helps the three and cannot help this
  one, because the error is a *biased* time scale, not an uncertain one.
- **A single training condition cannot identify the tau law**, so the one- and
  two-cell replays have a large worst cell (10.9 and 7.6) even though their
  means stay below the baseline (0.58 and 0.68). Averaged over all 15 pairs the
  model is at 3.46 against 5.12.
- **Beyond 58% SOH** the shape is unconstrained by data; the saturating tail
  is a safety device, not a prediction.
- **No replicates**, so no cell-to-cell variance estimate. The GP nugget
  assumes 6% scatter on `ln tau` (between the 3% cross-validation prefers and
  the ~6.6% measured within same-protocol replicate groups on public data);
  if siblings scatter more, the model is slightly over-confident at the
  released conditions.

The evaluation harness that produced this table (`scripts/benchmark.py` in the
team repository) is not part of the submission - it needs the released cells as
ground truth. It re-runs `fit()` and `predict_soh()` exactly as the framework
does, so every number above is reproducible from this model file plus the
released dataset:

```text
python scripts/benchmark.py master baseline \
    --protocoles in-sample,loco,profond,deep,loco-800,loco-400,deep-800,paires,solo
```

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
