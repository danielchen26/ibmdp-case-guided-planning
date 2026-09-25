# CNS Brain-Penetration Surrogate Dataset (`cns220_synthetic.csv`)

## What this is

`cns220_synthetic.csv` is a **distribution-matched *synthetic* surrogate** of the
proprietary CNS brain-penetration dataset used in the IBMDP paper
("Case-Guided Sequential Assay Planning in Drug Discovery"). It is released so
that reviewers and readers can **run the IBMDP pipeline end-to-end and reproduce
the method's behavior** without access to the proprietary data.

- **220 synthetic compounds** (`SYN-0001` … `SYN-0220`), 15 numeric columns
  matching the real schema (`kpuu`, `100nM_PgP`, `1uM_PgP`, `100nM_BCRP`,
  QSAR predictors, MRT columns, etc.).
- It contains **no real Merck/MSD compound identifiers and no individually
  identifying real measured value.** Every row is a fresh sample drawn from a
  statistical model fitted to the real data's *aggregate* distribution — not a
  perturbation of any real record. Values shared by many real compounds (assay
  ceilings/floors) are reproduced, since such a level is a property of the
  marginal distribution rather than of any one compound; `verify_surrogate.py`
  reports these separately and asserts that no singleton real value appears.

> **It is NOT the real data.** Numbers here must not be interpreted as
> measurements of any compound. Use it only to exercise the code / reproduce the
> qualitative method.

## How it was generated

Gaussian-copula resampling (`synthesize_cns_surrogate.py`, fixed seed):
1. log-transform each positive heavy-tailed column;
2. map each column to standard-normal scores via empirical ranks (copula
   marginal transform) and estimate the 15×15 correlation matrix;
3. draw 220 fresh multivariate-normal samples with that correlation;
4. invert through the empirical log-quantile function and exponentiate back
   (every value is interpolated from the empirical quantile function — new, not copied);
5. clip `kpuu` to [0, 1] and rank-calibrate it **within each class** so the
   good/bad split (kpuu > 0.5) exactly matches the real **88 good / 132 bad**.
   The calibration uses a *smoothed* (gap-interior) quantile map: each synthetic
   rank is placed at a random interior point of the corresponding gap between two
   consecutive real order statistics, so the emitted value is never one of them.
   Emitted 6-significant-figure text is then checked against the real column and
   any chance collision with a value that only one real compound has is nudged
   into the neighbouring gap, so the guarantee holds on the released text, not
   merely on the underlying float.

## Verification (`verify_surrogate.py`)

| Property | Real | Synthetic |
|---|---|---|
| Marginal distributions (2-sample KS; p>0.05 = no difference detected) | — | **15/15 columns** p>0.05, max KS 0.091 |
| Class balance (kpuu > 0.5) | 88 good / 132 bad | **88 good / 132 bad** |
| 15×15 correlation matrix (Frobenius distance) | — | **1.78** (0 = identical) |
| corr(QSAR, assay) e.g. 1uM_PgP↔qsar | +0.79 | +0.70 |
| **Leakage (row-level):** exact/near-duplicate synthetic rows | — | **0** |
| synth→real nearest-neighbor distance (std.) | — | min 0.37, median 1.47 (≈ real→real min 0.53, median 1.23) |
| **Leakage (value-level):** individually identifying real measurements present | — | **0 in all 15 columns** |
| **Leakage (column-level):** columns that are a permutation of a real column | — | **0 of 15** |

No synthetic point coincides with a real compound; synthetic points sit at
normal inter-sample distances, so the real values cannot be recovered from this file.

### Per-column KS statistics

The table above summarises the marginal check as "15/15 columns p>0.05". The
per-column two-sample KS statistics and p-values, as printed by
`verify_surrogate.py` against the proprietary source, are:

| Column | KS | p |
|---|---|---|
| `blood_frac_conc` | 0.032 | 1.000 |
| `brain_conc` | 0.045 | 0.977 |
| `brain_binding` | 0.041 | 0.993 |
| `plasma_protein_binding` | 0.091 | 0.324 |
| `kpuu` | 0.009 | 1.000 |
| `kpuu_numerator` | 0.059 | 0.838 |
| `kpuu_denominator` | 0.073 | 0.607 |
| `100nM_PgP` | 0.055 | 0.900 |
| `1uM_PgP` | 0.073 | 0.607 |
| `100nM_BCRP` | 0.041 | 0.993 |
| `1uM_PgP_qsar` | 0.073 | 0.607 |
| `100_nM_Mouse_BCRP_qsar` | 0.064 | 0.766 |
| `cassette_mrt` | 0.050 | 0.947 |
| `full_pk_mrt` | 0.064 | 0.766 |
| `qsar_mrt` | 0.073 | 0.607 |

These are aggregate two-sample statistics over N=220 per arm; they disclose no
individual measurement. The weakest match is `plasma_protein_binding`
(KS=0.091, p=0.32), still far from rejection. A reader cannot re-run this check
without the proprietary source, which is why the output is recorded here.

**How to read a non-significant KS test.** `p > 0.05` means the test found no
difference large enough to detect at 220 samples per arm. It is *not* evidence
that the two marginals are equal: a two-sample KS test cannot be inverted into an
equivalence claim, and at this sample size it has limited power against
differences confined to the tails. The load-bearing quantity is therefore the KS
statistic itself — the largest discrepancy found anywhere in the CDF, at most
0.091 across the fifteen columns — not the p-value beside it.

### Why the value-level check matters

Row-wise nearest-neighbor distance is **not sufficient** to establish absence of
leakage. A column that is a *permutation* of the corresponding real column
discloses every real measurement in it while leaving every row-wise
nearest-neighbor distance comfortably large, because the permutation destroys the
row pairing but preserves the multiset of values. An earlier revision of the
generator had exactly this defect in `kpuu`: rank-matching with
`q = rank/(n_out−1)·(n−1)` makes `q` integral whenever `n_out == n` (here both are
220), so the interpolation weight was identically zero and the calibrated column
reduced to a verbatim permutation of all 220 real `kpuu` values. Check (C) of
`verify_surrogate.py` now tests per column, at the level of values, and the script
exits non-zero if any check fails.

Values that *many* real compounds share — assay ceilings and floors, e.g. the
`kpuu` upper clip — do still appear, and are reported separately as "shared-level
matches". Such a level characterizes the marginal distribution rather than any
individual compound, and reproducing it is necessary for the surrogate to behave
like the real data.

## Method reproduction

Running IBMDP on this surrogate reproduces the paper's **cost-vs-uncertainty
Pareto behavior**, level by level: at ε = 0 the median first-batch cost is
**\$4,000** (measure the decisive *in vivo* k_puu assay directly), and at **every
one** of the ten looser grid levels ε = 0.1 … 1.0 it is **\$800** (cheap *in
vitro* PgP/BCRP proxies suffice). Those eleven per-level medians equal the ones
the paper reports for the real cohort.

Quote the per-level medians, not a band-pooled one. A median taken over the
pooled band ε ≤ 0.1 comes out at \$4,000, but only because the ε = 0 level
contributes 185 expensive first batches and so dominates the 193 rows at ε = 0.1,
whose own median is already \$800 — that is a property of the pooling, not of the
tolerance. The paper makes the same point about its own earlier revision.

Reproduce with `julia --project=. cns220_worker_synth.jl 1 220 out.csv 2000 10`,
then `julia --project=. cns220_figure.jl --results out.csv --no-plot`, which
prints the per-level medians directly.

## Files
- `cns220_synthetic.csv` — the surrogate dataset (this directory)
- `../synthesize_cns_surrogate.py` — generator
- `../verify_surrogate.py` — distribution-match + leakage verification
- `../cns220_worker_synth.jl` — IBMDP runner pointed at the surrogate
