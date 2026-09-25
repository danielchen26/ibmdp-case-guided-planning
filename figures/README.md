# Figure generators

Every figure in the paper and the ESI is produced by a script in this directory
(or, for main Figure 3, by `../cns220_figure.jl`), from inputs that are shipped
here. No figure in the manuscript carries a hand-entered number.

Run every command from the **repository root**, not from this directory.

## What is needed

| Generator | Needs |
|---|---|
| `*.jl` | Julia 1.11 with this repository's environment (`julia --project=. -e 'using Pkg; Pkg.instantiate()'`); plots via `Plots.jl`/GR |
| `*.py` | Python ≥ 3.9 with **matplotlib ≥ 3.5**; `fig1_scheme.py` also needs **numpy** |
| `inkmeasure.py` | additionally **Pillow**, **numpy**, and **Ghostscript** (`gs`) — it rasterises through Ghostscript because GR converts text to paths; set `GS=/path/to/gs` if `gs` is not on `PATH` |

`pandas` is *not* required — every script reads CSV through the standard-library
`csv` module, because on many systems the interpreter that has matplotlib is not
the one that has pandas. If `python3` on your machine lacks matplotlib, name the
interpreter that has it explicitly (e.g. `python3.12 figures/…`).

## The figures

### Main Figure 1 — the IBMDP workflow schematic

```bash
python3 figures/fig1_scheme.py --art figures/art --out IBMDP_scheme
```

Draws the four-panel workflow cycle at exactly 469.755 × 322 pt, the width it is
included at, so `\includegraphics` applies no scale and every font size in the
script is a literal on-page point size. Prints the effective dpi of each piece of
reused line art and the smallest type on the canvas (6.3 pt).

The four pieces of line art in `art/` (molecule, petri dish, mouse, monkey) are
the authors' own, extracted from `art/TOCScheme.svg`. That extraction is itself
reproducible:

```bash
python3 figures/extract_art.py --svg figures/art/TOCScheme.svg --out figures/art
```

It reports that the SVG contains five embedded rasters and **zero** `<text>`
elements — which is why the original scheme could not simply be re-exported at a
larger size, and why Figure 1 is re-laid-out rather than rescaled.

### Main Figure 2 — four-panel ensemble Pareto fronts

```bash
julia --project=. figures/fig2_fronts.jl \
    --s1 figures/data/fig2_panel1_ensemble_tau0.9.csv --k1 0.529 \
    --s2 figures/data/fig2_panel2_ensemble_tau0.9.csv --k2 0.534 \
    --s3 figures/data/fig2_panel3_ensemble_tau0.9.csv --k3 0.541 \
    --s4 figures/data/fig2_panel4_ensemble_tau0.9.csv --k4 0.640 \
    --out replotted.pdf
```

Prints `N_e` and the vote-share range for each panel; all four report
`N_e = 50`. The `--k` values are panel-title labels only; they enter no
computation. They are the four compounds' measured k_puu to three decimals, the
form the manuscript itself prints on these panel titles. Table 1 of the
manuscript reports the same four values rounded to two decimals.

### Main Figure 3 — the N = 220 cohort sweep

```bash
julia --project=. cns220_figure.jl \
    --results data/cns220_synthetic_ibmdp_results.csv \
    --data data/cns220_synthetic.csv --out cns220
```

Add `--no-plot` for the numbers only. It prints the per-tolerance-level median
first-batch cost with its IQR, the fraction of compounds measuring k_puu, the
number of compounds contributing at each level, and the per-quadrant counts.
**Read the per-level medians, not the band-pooled ones** — see the note in the
root `README.md`.

Figure 3 in the manuscript is this same script run on the real cohort, which is
proprietary and not released. On the shipped surrogate it reproduces the paper's
eleven per-level medians exactly (\$4,000 at ε = 0, \$800 at each of ε = 0.1 …
1.0) but not the per-quadrant counts, which depend on the real QSAR–k_puu joint
structure.

### The RSC table-of-contents graphical abstract

```bash
python3 figures/toc_graphical_abstract.py \
    --results data/cns220_synthetic_ibmdp_results.csv --out TOCScheme
```

Drawn at exactly 8 × 4 cm, the RSC TOC box, so again no scaling is applied and
the sizes in the script are on-page points (smallest 6.0 pt). Reads the released
surrogate sweep, so the graphic is reproducible from public files alone.

### ESI Figure S1 — ensemble vote histogram at one planning state

```bash
python3 figures/esi_figS1_votes.py \
    --csv figures/data/fig2_panel3_ensemble_tau0.9.csv \
    --eps 0.0 --kpuu 0.541 --label "compound 3" --tau 0.9 --out esi_ensemble_votes
```

Prints `winner $P_1\,k$ at 27/50, objective 36377.6` across 7 distinct action
sets — the numbers the ESI caption quotes, and the objective annotated in panel 3
of main Figure 2. The script's own docstring records that the figure it replaces
carried bar heights read off an earlier image by visual inspection, and that two
ensembles exist at this state; only the one shipped here matches the paper.

### ESI Figure S2 — the two goal-likelihood floors

```bash
julia --project=. figures/esi_figS2_tau.jl \
    --tau06 figures/data/esi_figS2_ensemble_tau0.6.csv \
    --tau09 figures/data/esi_figS2_ensemble_tau0.9.csv --out esi_figS2
```

Prints both MLASP loci level by level and the two crossovers the ESI text
discusses (at ε = 0.3 and ε = 0.4), plus the modal-plan support range.

### ESI Figure S3 — MLASP loci on the public clearance dataset

```bash
python3 figures/esi_figS3_adme.py --data figures/data --out adme_tau_comparison.png
```

Reads `figures/data/ADME_ensemble_results_tau_0.{6,9}.csv` and prints both loci
with each modal batch's vote count *and* how many ensemble members returned any
plan at that tolerance — the "2–30 of 30" figures the ESI reports.

## Shipped inputs

`data/` holds planner output only: the columns are exactly
`Threshold, Action_Set, Average_Utility, Frequency` — a tolerance, a candidate
first batch, that batch's ensemble mean objective, and how many ensemble members
chose it. **No measured compound property appears in any of these files.**

The six CNS files are named for the panel they feed rather than for the run
directory they came from, because the planner writes its output directories under
the compound's measured k_puu at full precision, and that is proprietary (see the
data policy in the root `README.md`). The two `ADME_*` files come from the public
clearance dataset and keep their original names.

## The measurement tool

```bash
python3 figures/inkmeasure.py <pdf> --dpi 1200 [--scale 0.8496]
```

Rasterises a PDF at high resolution and measures the actual height of capital
ink, which is how the on-page font sizes cited in the generators' headers were
established rather than assumed. It is what showed that GR renders text at
1.551× the nominal size it is handed — the factor `fig2_fronts.jl` and
`esi_figS2_tau.jl` correct for, and the reason a figure that looks fine on screen
can still reach the page below a publisher's 6 pt floor.
