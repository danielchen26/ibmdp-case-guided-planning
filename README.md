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
> **release gate**: it exits non-zero if any such value is present. The gate
> compares the surrogate against the real cohort, so only the authors can run it;
> the verdict it returned for the file shipped here, and every statistic behind
> that verdict, are recorded in [`data/README_synthetic.md`](data/README_synthetic.md).

## Contents

```
LICENSE                                   MIT
Project.toml                              Julia deps; resolves CEEDesigns from engine/ (see SETUP.md)
Manifest.toml                             pinned dependency versions (the tested environment)
SETUP.md                                  installation, and why the engine is vendored — read this first
src/RIPK1.jl                              CNS case-study layer: similarity model, assay costs, config
cns220_worker_synth.jl                    N=220 cost-vs-uncertainty sweep driver (Figure 3 analogue)
cns220_figure.jl                          aggregates a sweep into Figure 3 + per-category / confirmation counts
kernel_diagnostics.jl                     similarity-kernel locality: N_eff, weight mass, variance prefactor, paired coefficient ratios (ESI)
niter_equiv_test.jl                       search-budget sensitivity test (ESI); shardable
niter_equiv_results.csv                   its raw output: 125 records over 7 compounds x 2 budgets x 11 tolerances
                                            (not all 154 cells yield a plan; see the ESI)
niter_equiv_report.py                     recomputes every ESI budget-test figure from that CSV
synthesize_cns_surrogate.py               surrogate generator (needs the proprietary source; not released)
verify_surrogate.py                       surrogate verifier / release gate — needs the proprietary
                                            cohort, so author-side only; its verdict is recorded in
                                            data/README_synthetic.md
data/cns220_synthetic.csv                 220 synthetic compounds (SYN-0001 …), 15 columns
data/cns220_synthetic_ibmdp_results.csv   pre-computed N=220 sweep output on the surrogate
data/README_synthetic.md                  synthetic-data card: generation, verification, limitations
benchmark/route1_benchmark.jl             synthetic policy-recovery benchmark (Table 1)
benchmark/score_table1.jl                 scores Table 1 from the shipped summary CSV
benchmark/route1_results/route1_summary.csv   pre-computed Table 1 output
figures/README.md                         how every figure is regenerated — read this one for figures
figures/fig1_scheme.py                    main Figure 1: the IBMDP workflow schematic
figures/fig2_fronts.jl                    main Figure 2: four-panel ensemble Pareto fronts
figures/toc_graphical_abstract.py         the journal table-of-contents graphical abstract
figures/esi_figS1_votes.py                ESI Fig. S1: ensemble vote histogram at one planning state
figures/esi_figS2_tau.jl                  ESI Fig. S2: MLASP loci at two goal-likelihood floors
figures/esi_figS3_adme.py                 ESI Fig. S3: the same, on the public clearance dataset
figures/extract_art.py                    re-extracts the four line-art panels from art/TOCScheme.svg
figures/inkmeasure.py                     measures true on-page font size from a rendered PDF
figures/art/                              the authors' line art, and the SVG it is extracted from
figures/data/                             the planner output the generators read (8 CSVs, no measured property)
engine/                                   vendored CEEDesigns planning engine (MIT) — REQUIRED, see SETUP.md
  src/GenerativeDesigns/                    IBMDP: ensemble MCTS-DPW, similarity belief, ensemble analysis
  src/ValueIteration/                       VI-Theo and VI-Sim baselines; generate_historical_dataset
```

Every figure in the paper and the ESI is regenerated by a script in `figures/`
(main Figure 3 by `cns220_figure.jl`), from inputs shipped in this repository.
No figure carries a hand-entered number. See [`figures/README.md`](figures/README.md)
for the exact command per figure.

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
julia --project=. kernel_diagnostics.jl --ratio    # all three settings paired per compound

# Every figure, from shipped inputs.  See figures/README.md for the full commands.
python3 figures/fig1_scheme.py --art figures/art --out IBMDP_scheme
julia  --project=. figures/fig2_fronts.jl --s1 … --k1 0.529 …       # four panels
julia  --project=. cns220_figure.jl --results data/cns220_synthetic_ibmdp_results.csv
python3 figures/esi_figS3_adme.py --data figures/data --out adme_tau_comparison.png
```

`cns220_worker_synth.jl` seeds each compound with `1000 + row`, so a sweep may be
sharded across processes without changing results.

## Expected behavior

On the synthetic cohort, IBMDP reproduces the paper's **cost-versus-uncertainty
Pareto shift**, level by level:

- **ε = 0** (no residual uncertainty tolerated): median first-batch cost
  **\$4,000** — the planner acquires the decisive *in vivo* k_puu assay directly,
  and does so for 84.1% of the 220 compounds;
- **every one of the ten looser grid levels, ε = 0.1 … 1.0:** median **\$800** —
  cheap *in vitro* PgP/BCRP proxies suffice, with the k_puu fraction falling to
  32.1% at ε = 0.1 and to 1.5–3.6% for ε ≥ 0.5.

All eleven per-level medians equal the ones reported for the real cohort, so the
surrogate reproduces the *method*'s qualitative behaviour and these aggregates.

**Quote the per-level medians, not a band-pooled one.** A median over the pooled
band ε ≤ 0.1 comes out at \$4,000, but only because the ε = 0 level contributes
185 expensive first batches and outweighs the 193 compounds at ε = 0.1, whose own
median is already \$800. That is a property of the pooling rather than of the
tolerance, and it hides where the shift actually happens: between ε = 0 and
ε = 0.1, not spread across a band. `cns220_figure.jl` prints the per-level
medians first; the band figures it also prints exist only for comparison with an
earlier revision of this repository and should not be quoted on their own.

What the surrogate does **not** reproduce, and is not claimed to: the
per-quadrant counts and the 48/59 confirmation count of the main text, which
depend on the real QSAR–k_puu joint structure. Numbers computed on the surrogate
should not be quoted as the paper's real-cohort results.

Note on the search budget: the N=220 sweep runs at `n_iter=2000`, `N_e=10`,
which is **not** equivalent to the four-compound study's `n_itr=20,000`,
`N_e=50`. `niter_equiv_test.jl` quantifies the iteration half of that gap on
seven surrogate compounds. The exact recommended batch agrees in 42/58 paired
cells (72%), total plan cost in 52/58 (90%), and the k_puu-in-batch criterion in
57/58 (98%); of the 16 batch disagreements, 10 are equal-cost permutations and 5
of the remaining 6 differ by exactly \$400. The per-level median first-batch cost
differs at **3 of the 11 levels** (ε = 0.2, 0.3, 0.4; 4 of 11 counting unpaired
cells, which adds ε = 0.5), and at each of them by one \$200 half-rung — the
median of an even number of solved cells falling between the \$400 and \$800
rungs, so these are count artifacts rather than plan changes. The band medians
(\$4,000 tight / \$800 loose)
are unchanged, but that agreement is weaker evidence than it looks — pooling
levels is what hides those three per-level differences. The `N_e=10→50` half of
the gap was **not** tested. Run `python3 niter_equiv_report.py` to reproduce every
one of these figures from the shipped CSV; see the ESI for the full comparison.

## Citation

Chen, T., Bima, J., Wu, S. L., Ritter, O., Yang, B., Yu, X.
*Case-Guided Sequential Assay Planning in Drug Discovery.* (2025).
