#!/usr/bin/env python3
"""
Generate a DISTRIBUTION-MATCHED SYNTHETIC SURROGATE of the proprietary CNS
brain-penetration dataset (multitier.csv, N=220), for public release alongside
the IBMDP code so that reviewers/readers can run the pipeline and reproduce the
METHOD (not the proprietary numbers).

Method: Gaussian-copula resampling.
  1. Log-transform each (strictly positive, heavy-tailed) column.
  2. Map each column to standard-normal via its empirical rank (Gaussian copula
     marginal transform), estimate the 15x15 correlation matrix.
  3. Draw N fresh multivariate-normal samples with that correlation.
  4. Invert: normal -> uniform (rank) -> back to the log-marginal via empirical
     quantiles -> exp back to original scale. Every value is NEW (interpolated
     from the empirical quantile function), not a copy of a real row.
  5. Clip kpuu to [0,1] and rank-calibrate it within each class (good/bad) using
     the SMOOTHED quantile map of smoothed_quantile_map(), so the 88/132 split is
     exact while every emitted value falls strictly inside a gap between two real
     order statistics.  Assign fresh synthetic IDs SYN-0001...

Leakage guarantee and how it is enforced: no *individually identifying* real
measurement is emitted.  A naive rank match would violate this -- with
n_out == n the interpolation weight is identically zero and the "calibrated"
column is a verbatim permutation of the real multiset -- so the calibration
draws a random interior point of each inter-order-statistic gap instead.  Values
that many real compounds share (assay ceilings/floors) can still appear, since
such a level characterizes the marginal rather than any one compound.
verify_surrogate.py asserts this per column, at the level of VALUES, not rows:
row-wise nearest-neighbor distance alone cannot detect a permuted column.

The surrogate preserves: marginal shapes, the 15x15 correlation structure
(incl. the QSAR<->assay +0.75/+0.79 correlations and the weak PgP/BCRP<->kpuu
negative correlations that IBMDP relies on), and the good/bad class balance.
It contains NO real Merck compound IDs and NO real measured values.

NO SEED FROM Date/random module state that could leak; a fixed seed is passed
for reproducibility of the surrogate itself.
"""
import csv, sys
import numpy as np

SRC = "data/multitier.csv"
OUT = "data/cns220_synthetic.csv"
SEED = 20260717
N_OUT = 220
CLASS_THRESHOLD = 0.5      # kpuu > 0.5 == "good" (brain-penetrant)

def load(path):
    rows = list(csv.DictReader(open(path)))
    cols = [c for c in rows[0].keys() if c != "compound_id"]
    X = np.array([[float(r[c]) for c in cols] for r in rows], dtype=float)
    return cols, X, rows

def gaussian_copula_sample(X, n_out, rng):
    """Return n_out synthetic rows matching X's marginals + correlation."""
    n, d = X.shape
    # log-transform positive heavy-tailed columns (all columns here are > 0)
    shift = np.where(X.min(axis=0) <= 0, 1e-6 - X.min(axis=0), 0.0)
    L = np.log(X + shift)
    # marginal -> normal scores via empirical ranks (copula transform)
    from scipy.stats import norm
    ranks = np.argsort(np.argsort(L, axis=0), axis=0)          # 0..n-1 per column
    U = (ranks + 0.5) / n                                       # (0,1), no 0/1
    Z = norm.ppf(U)                                             # normal scores
    C = np.corrcoef(Z, rowvar=False)                            # 15x15 corr
    # sample new normal scores with that correlation
    Lchol = np.linalg.cholesky(C + 1e-8 * np.eye(d))
    Znew = rng.standard_normal((n_out, d)) @ Lchol.T
    Unew = norm.cdf(Znew)                                       # (0,1)
    # invert marginal: empirical quantile of the log-column, then exp back
    Xnew = np.empty((n_out, d))
    for j in range(d):
        Lsorted = np.sort(L[:, j])
        q = Unew[:, j] * (n - 1)                                # position in sorted
        lo = np.floor(q).astype(int); hi = np.ceil(q).astype(int)
        frac = q - lo
        Lval = Lsorted[lo] * (1 - frac) + Lsorted[hi] * frac    # interpolate
        Xnew[:, j] = np.exp(Lval) - shift[j]
    return Xnew

def smoothed_quantile_map(target, x, rng):
    """Monotone rank-calibrate x onto target's marginal WITHOUT copying values.

    Each synthetic rank is placed at a *random interior point* of the
    corresponding inter-order-statistic gap of `target`, so the returned value
    lies strictly between two consecutive real order statistics rather than on
    one of them.  (A plain rank match -- q = rank/(n_out-1)*(n-1) -- is a
    verbatim copy of the real multiset whenever n_out == n, because then q is
    integral and the interpolation weight is exactly 0.  That is the defect this
    function exists to avoid.)  Ties in `target`, e.g. an assay ceiling shared by
    many compounds, are reproduced as that tied value: a level shared by dozens
    of compounds is an aggregate feature of the marginal, not an individual
    measurement.
    """
    t = np.sort(np.asarray(target, dtype=float))
    n, n_out = len(t), len(x)
    order = np.argsort(np.argsort(x))                          # 0..n_out-1
    u = (order + rng.uniform(0.0, 1.0, size=n_out)) / n_out    # strictly in (0,1)
    q = u * (n - 1)
    lo = np.clip(np.floor(q).astype(int), 0, n - 1)
    hi = np.clip(lo + 1, 0, n - 1)
    frac = q - lo
    return t[lo] * (1.0 - frac) + t[hi] * frac

def main():
    cols, X, rows = load(SRC)
    rng = np.random.default_rng(SEED)
    Xnew = gaussian_copula_sample(X, N_OUT, rng)
    # enforce known bounds / consistency
    if "kpuu" in cols:
        j = cols.index("kpuu")
        Xnew[:, j] = np.clip(Xnew[:, j], 0.0, 1.0)
        # Class-balance correction: rank-calibrate synthetic kpuu onto the real
        # kpuu marginal so the good/bad split (kpuu>0.5) exactly matches the real
        # 88/132, while preserving the copula's cross-column rank ordering (each
        # synthetic compound keeps its relative kpuu rank; only the values are
        # re-mapped).  The calibration is done SEPARATELY WITHIN the two classes,
        # which pins the split exactly by construction, and uses the smoothed
        # (gap-interior) quantile map so no real measurement is reproduced.
        real_kpuu = np.sort(X[:, j])
        bad_ref = real_kpuu[real_kpuu <= CLASS_THRESHOLD]
        good_ref = real_kpuu[real_kpuu > CLASS_THRESHOLD]
        order = np.argsort(np.argsort(Xnew[:, j]))              # 0..N_OUT-1 ranks
        n_bad = int(round(len(bad_ref) / len(real_kpuu) * N_OUT))
        is_bad = order < n_bad                                 # lowest-ranked -> bad
        out = np.empty(N_OUT)
        out[is_bad] = smoothed_quantile_map(bad_ref, Xnew[is_bad, j], rng)
        out[~is_bad] = smoothed_quantile_map(good_ref, Xnew[~is_bad, j], rng)
        Xnew[:, j] = np.clip(out, 0.0, 1.0)
    Xnew = np.clip(Xnew, 0.0, None)                             # all cols non-negative

    # Emitted text, not the float, is what ships: round FIRST, then assert on the
    # rounded values.  At 6 significant figures a synthetic value can coincide
    # with a singleton real measurement by chance; nudge any such collision into
    # its neighbouring gap so the released file provably contains none.
    #
    # The guarantee must hold at every precision a DOWNSTREAM consumer might round
    # to, not only at the 6 significant figures written here.  A value that is
    # safe at 6 sf can collide with a real measurement once coarsened: the released
    # sweep driver emits `round(kpuu, digits=4)`, and a 6-sf-safe value rounded to
    # 4 decimals landed back on a real singleton.  So de-collide against the union
    # of the real singleton sets taken over all of these renderings.
    # Only the precisions actually emitted by released code are enforced.  Going
    # coarser (2 dp) would demand nudges above assay precision in the wide columns
    # and would visibly distort those marginals, so it is deliberately not done;
    # any new consumer that coarsens further must be added here and the surrogate
    # regenerated.
    ROUNDINGS = [lambda v: float(f"{v:.6g}"),      # this file
                 lambda v: round(v, 4)]            # cns220_worker*.jl, niter_equiv_test.jl

    def collides(v, singleton_sets):
        return any(r(v) in s for r, s in zip(ROUNDINGS, singleton_sets))

    Sround = np.array([[float(f"{v:.6g}") for v in row] for row in Xnew])
    for j, c in enumerate(cols):
        col = X[:, j]
        singleton_sets = []
        for r in ROUNDINGS:
            vals, counts = np.unique(np.array([r(v) for v in col]), return_counts=True)
            singleton_sets.append(set(vals[counts == 1].tolist()))
        for i in range(N_OUT):
            tries = 0
            while collides(Sround[i, j], singleton_sets) and tries < 256:
                # jitter within the local gap: 1e-3 relative is far below assay
                # precision yet always changes the 6th significant figure
                Sround[i, j] = float(f"{Sround[i, j] * (1.0 + 1e-3 * (1 + tries)):.6g}")
                tries += 1
            assert not collides(Sround[i, j], singleton_sets), \
                f"could not de-collide {c} row {i} at all reported precisions"
    if "kpuu" in cols:
        j = cols.index("kpuu")
        Sround[:, j] = np.clip(Sround[:, j], 0.0, 1.0)
        good = int((Sround[:, j] > CLASS_THRESHOLD).sum())
        real_good = int((X[:, j] > CLASS_THRESHOLD).sum())
        assert good == real_good, f"class balance drifted: {good} vs {real_good}"

    # write with fresh synthetic IDs
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["compound_id"] + cols)
        for i in range(N_OUT):
            w.writerow([f"SYN-{i+1:04d}"] + [f"{v:.6g}" for v in Sround[i]])
    print(f"wrote {OUT}: {N_OUT} synthetic compounds, {len(cols)} columns")
    return cols, X, Sround

if __name__ == "__main__":
    main()
