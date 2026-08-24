# IBMDP — Case-Guided Sequential Assay Planning in Drug Discovery

Reference implementation of the **Implicit Bayesian Markov Decision Process
(IBMDP)** — a simulator-free, similarity-weighted ensemble-MCTS planner for
sequential assay selection — together with the **VI-Theo and VI-Sim
value-iteration baselines**, the synthetic policy-recovery benchmark behind
Table 1, the search-budget sensitivity test reported in the ESI, and a
**distribution-matched synthetic surrogate** of the CNS cohort.

> **Data policy.** The real CNS brain-penetration dataset used in the paper is
> proprietary Merck & Co., Inc., Rahway, NJ, USA (known as MSD outside the United
> States and Canada) research data and is **not** included or released here. This
> repository ships a **synthetic surrogate** (`data/cns220_synthetic.csv`) whose
> marginal distributions, 15×15 correlation structure and good/bad class balance
> match the real data's *aggregate* statistics.
>
> The surrogate contains **no real compound identifiers** and **no individually
> identifying real measurement**: for every column, no value held by only one
> real record appears in the surrogate, at any precision this repository's code
> emits. Values shared by many real records — assay ceilings and floors — do
> recur, because they characterise the marginal rather than any individual
> compound. This is verified mechanically by `verify_surrogate.py`, which is a
> **release gate**: it exits non-zero if any such value is present.
> See [`data/README_synthetic.md`](data/README_synthetic.md).

## Contents

```
LICENSE                                   MIT
Project.toml                              Julia deps; resolves CEEDesigns from engine/ (see SETUP.md)
Manifest.toml                             pinned dependency versions (the tested environment)
SETUP.md                                  installation, and why the engine is vendored — read this first
src/RIPK1.jl                              CNS case-study layer: similarity model, assay costs, config
cns220_worker_synth.jl                    N=220 cost-vs-uncertainty sweep driver (Figure 3 analogue)
cns220_figure.jl                          aggregates a sweep into the figure + per-category / confirmation counts
kernel_diagnostics.jl                     similarity-kernel locality: N_eff, weight mass, variance prefactor (ESI)
niter_equiv_test.jl                       search-budget sensitivity test (ESI); shardable
niter_equiv_results.csv                   its raw output: 125 records over 7 compounds x 2 budgets x 11 tolerances
                                            (not all 154 cells yield a plan; see the ESI)
niter_equiv_report.py                     recomputes every ESI budget-test figure from that CSV
synthesize_cns_surrogate.py               surrogate generator (needs the proprietary source; not released)
verify_surrogate.py                       surrogate verifier / release gate — must exit 0
data/cns220_synthetic.csv                 220 synthetic compounds (SYN-0001 …), 15 columns
data/cns220_synthetic_ibmdp_results.csv   pre-computed N=220 sweep output on the surrogate
data/README_synthetic.md                  synthetic-data card: generation, verification, limitations
benchmark/route1_benchmark.jl             synthetic policy-recovery benchmark (Table 1)
benchmark/score_table1.jl                 scores Table 1 from the shipped summary CSV
benchmark/route1_results/route1_summary.csv   pre-computed Table 1 output
engine/                                   vendored CEEDesigns planning engine (MIT) — REQUIRED, see SETUP.md
  src/GenerativeDesigns/                    IBMDP: ensemble MCTS-DPW, similarity belief, ensemble analysis
  src/ValueIteration/                       VI-Theo and VI-Sim baselines; generate_historical_dataset
```

`engine/` is **not** a verbatim copy of upstream `Merck/CEEDesigns.jl`; SETUP.md
lists both why it is vendored and the changes made for this paper's benchmark —
two behavioural and one cosmetic, all confined to `ValueIteration.jl`.

## Reproduce

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# N=220 sweep: one row per (compound, uncertainty tolerance ε).
julia --project=. cns220_worker_synth.jl 1 220 results.csv 2000 10
#                                        lo hi  output      n_iter ensemble

# Aggregate a sweep into the four-panel figure and the per-category counts.
julia --project=. cns220_figure.jl --results results.csv          # add --no-plot for counts only
#   or, without re-running the sweep, from the shipped surrogate output:
julia --project=. cns220_figure.jl --results data/cns220_synthetic_ibmdp_results.csv --no-plot

# Table 1: IBMDP vs VI-Theo vs VI-Sim policy recovery.
julia --project=. benchmark/route1_benchmark.jl
julia --project=. benchmark/score_table1.jl        # or score the shipped CSV directly

# ESI search-budget sensitivity (n_itr = 2,000 vs 20,000).
julia --project=. niter_equiv_test.jl              # ~2.8 core-hours for all 14 (compound, budget) jobs
python3 niter_equiv_report.py                      # reads the shipped CSV; no Julia needed

# ESI kernel-locality diagnostics (N_eff, weight mass, variance prefactor).
julia --project=. kernel_diagnostics.jl            # the CNS coefficients; add --unit or --gamma 0.5
```

`cns220_worker_synth.jl` seeds each compound with `1000 + row`, so a sweep may be
sharded across processes without changing results.

## Expected behavior

On the synthetic cohort, IBMDP reproduces the paper's **cost-versus-uncertainty
Pareto shift**:

- **tight tolerance (ε ≤ 0.1):** band-pooled median first-batch cost
  **\$4,000** — the planner acquires the decisive *in vivo* k_puu assay directly;
- **loose tolerance (ε ≥ 0.5):** **\$800** — cheap *in vitro* PgP/BCRP proxies
  suffice.

Both band medians equal the values reported for the real cohort, so the surrogate
reproduces the *method*'s qualitative behaviour and these two aggregates.

What the surrogate does **not** reproduce, and is not claimed to: the
per-quadrant counts and the 48/59 confirmation count of the main text, which
depend on the real QSAR–k_puu joint structure. Numbers computed on the surrogate
should not be quoted as the paper's real-cohort results.

Note on the search budget: the N=220 sweep runs at `n_iter=2000`, `N_e=10`,
which is **not** equivalent to the four-compound study's `n_itr=20,000`,
`N_e=50`. `niter_equiv_test.jl` quantifies the iteration half of that gap on
seven surrogate compounds — the exact recommended batch agrees in 42/58 paired
cells (72%) and the k_puu-in-batch criterion in 57/58 (98%), while both band
medians ($4,000 tight / $800 loose) are unchanged. The `N_e=10→50` half was not
tested. Run `python3 niter_equiv_report.py` to reproduce these figures from the
shipped CSV; see the ESI for the full comparison.

## Citation

Chen, T., Bima, J., Wu, S. L., Ritter, O., Yang, B., Yu, X.
*Case-Guided Sequential Assay Planning in Drug Discovery.* (2025).
