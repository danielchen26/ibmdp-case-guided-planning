# Setup

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

That is the whole installation. Julia 1.9+ (tested on 1.11.6).

## Why the planning engine is vendored in `engine/`

`Project.toml` resolves `CEEDesigns` from the vendored copy in `engine/` via a
`[sources]` entry, **not** from the registered public package. This is required,
not a convenience:

The public releases of `Merck/CEEDesigns.jl` (v0.3.5–v0.3.9, the only registered
versions) do **not** contain the ensemble-analysis layer or the value-iteration
module this study is built on. Concretely, these names are absent from every
public version:

| Name | Needed by |
|---|---|
| `process_ensemble_results_enhanced` | `src/RIPK1.jl`, both sweep drivers |
| `find_max_likelihood_action_sets_with_utility` | `src/RIPK1.jl` |
| `find_top_n_action_sets_with_utility` | `cns220_worker_synth.jl`, `niter_equiv_test.jl` |
| `plot_multiple_max_likelihood_action_sets` | `src/RIPK1.jl` |
| `validate_ensemble_frequencies` | `src/RIPK1.jl` |
| `CEEDesigns.ValueIteration` (whole module) | `benchmark/route1_benchmark.jl` (VI-Theo, VI-Sim, `generate_historical_dataset`) |

Upstream's own integration note (`src/GenerativeDesigns/Readme.md` in
`Merck/CEEDesigns.jl`) records that "`ValueIteration/`-related code was
intentionally excluded".

**This is a silent failure if you get it wrong.** Julia does not raise an error
when `using M: name` names something `M` does not export — it only prints
`WARNING: could not import ...` and continues. The script then loads, runs, and
writes rows containing `ERROR:MethodError(...)` with `cost=NaN` for every
compound. Against public CEEDesigns v0.3.9 the driver exits 0 and produces a CSV
in which *every* data row is such an error row. If you see
`WARNING: could not import` on startup, your `CEEDesigns` is not the vendored
one — check `pkgdir(CEEDesigns)`.

The vendored engine is MIT-licensed and carries the same copyright holder as this
work; see `engine/LICENSE`.

## How the vendored engine differs from upstream

`engine/` is a snapshot of the internal CEEDesigns tree, not a verbatim copy of
any public release. Beyond the missing-module issue above, deliberate changes
were made for this paper's synthetic benchmark. They are confined to a single
file, `engine/src/ValueIteration/ValueIteration.jl` — every other file under
`engine/src/` is byte-identical to the internal snapshot — and there are two
that change behaviour, plus one that changes only console output. All are
visible in the file and are stated here so they are not mistaken for upstream
behaviour:

**Which tree to diff against.** The three changes below are stated relative to
the *internal snapshot* that `engine/` is taken from. Two public repositories
carry related code, and neither is that snapshot:

- `Merck/CEEDesigns.jl` (registered v0.3.5–v0.3.9) has no `ValueIteration/`
  directory at all, so there is nothing there to diff the changed file against.
- `MSDLLCpapers/IBMDPDesigns.jl` *does* ship `src/ValueIteration/ValueIteration.jl`,
  and diffing our copy against it shows the three changes below plus their
  explanatory comments — that is the useful comparison for the changed file.
  But its wider tree is a different lineage of the same code base: the MDP layer
  is named `MDP/` rather than `GenerativeDesigns/`, and our snapshot additionally
  carries `StaticDesigns/` and `EfficientValueMDP.jl`. Those differences are not
  changes we made; they are two snapshots of an evolving internal tree.

So a whole-tree diff against either public repository will show far more than the
three items listed here. The claim is about the changed file, against the
snapshot `engine/` was cut from.

1. **All six features are informative.** Upstream's
   `generate_historical_dataset` overwrote the *last* feature's distribution with
   a zero-mean, near-zero-variance noise distribution
   (`feature_distributions[end] = truncated(Normal(0,5), -10, 10)`) while still
   giving it a nonzero `β` coefficient — so feature 6 carried a coefficient but
   almost no signal, and `g = β·y` was not the model the baselines assumed.
   Here every feature is drawn from the same truncated-normal family
   (`μ_a = 50·a/6`, `σ_a = 0.3 μ_a`, support `[0, 2μ_a]`), making the linear
   model well-posed. This matters for the reported benchmark: the ESI's Table 1
   comparison is only meaningful if VI-Theo's closed-form optimum is the true
   optimum of the generated data.

2. **`value_iteration_analysis` accepts `seed`, `custom_coeffs` and
   `shared_data`.** Upstream hard-coded `seed = 42` inside the function body and
   regenerated its own dataset, which meant the VI baselines were solved on a
   *different* draw from the one IBMDP planned against — a one-iteration lag that
   silently invalidates a paired comparison. Passing `shared_data` makes both
   sides use the identical per-trial dataset; passing `custom_coeffs` forwards
   the per-trial coefficient vector `β⁽ᵗ⁾` to `generate_historical_dataset`
   (which already accepted such an argument upstream) rather than letting it be
   redrawn. All three are keyword arguments defaulting to upstream behaviour
   (`seed = 42`, both others `nothing`), so a caller that passes nothing gets
   the original semantics apart from item 1.

3. **Console output only:** `verbose` was flipped from `true` to `false` in the
   two `solve_mdp`-style calls, so a 100-trial sweep does not emit 100 solver
   traces. This changes no computed value.

Note that the additive noise term `ε` is switched off in this benchmark
(`add_noise = false`, the fourth *positional* argument to
`generate_historical_dataset`), so `g_i = β·y_i` exactly. The ESI states this
and quantifies what a nonzero `σ_ε` would change.

## Reproducing

```bash
# N=220 cost-versus-uncertainty sweep on the synthetic surrogate (Figure 3 analogue).
# Shard freely: each (compound) job re-seeds with 1000+i and shares no state.
julia --project=. cns220_worker_synth.jl 1 220 results.csv 2000 10

# Search-budget sensitivity test (ESI). Shardable via NITER_ROWS / NITER_BUDGETS.
# ~2.8 core-hours in total; the n_itr=20,000 arm dominates (2.5 of those hours).
julia --project=. niter_equiv_test.jl
python3 niter_equiv_report.py            # recompute the ESI figures from the shipped CSV

# Kernel-locality diagnostics quoted in the ESI (N_eff, weight mass, prefactor).
julia --project=. kernel_diagnostics.jl              # CNS coefficients (λ_w λ_k = 25 / 100)
julia --project=. kernel_diagnostics.jl --unit       # λ_w λ_k = 1
julia --project=. kernel_diagnostics.jl --gamma 0.5  # the synthetic-benchmark value

# Synthetic policy-recovery benchmark, Table 1 (IBMDP vs VI-Theo vs VI-Sim).
julia --project=. benchmark/route1_benchmark.jl
julia --project=. benchmark/score_table1.jl

# Regenerate / re-verify the surrogate. verify_surrogate.py is a RELEASE GATE:
# it exits non-zero if any real measured value would be disclosed.
python3 synthesize_cns_surrogate.py     # requires the proprietary source data; not released
python3 verify_surrogate.py             # must print "VERDICT: PASS" and exit 0
```

`benchmark/route1_results/route1_summary.csv` is the pre-computed output backing
Table 1, so Table 1 can be checked without re-running the benchmark.
