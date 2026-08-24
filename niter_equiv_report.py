#!/usr/bin/env python3
"""Recompute every figure the ESI quotes for the MCTS-budget equivalence test.

Reads niter_equiv_results.csv -- the raw output of niter_equiv_test.jl -- and prints
the statistics that appear in the ESI's hyperparameter and complexity sections, so
that paragraph can be checked against the shipped file without re-running the planner.

    python3 niter_equiv_report.py [niter_equiv_results.csv]

The cohort here is the synthetic surrogate, so all values are safe to print.
"""
import csv
import os
import re
import statistics as st
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "niter_equiv_results.csv")
with open(path) as f:
    rows = list(csv.DictReader(f))

BUDGETS = (2000, 20000)
TIGHT = 0.1          # epsilon <= TIGHT is the "tight" band
LOOSE = 0.5          # epsilon >= LOOSE is the "loose" band


def batch(mlasp):
    """MLASP text -> frozenset of assay names (the batch is unordered)."""
    return frozenset(x.strip() for x in re.findall(r'"([^"]+)"', mlasp))


solved = [r for r in rows if r["eps"] != "NA" and r["mlasp"] not in ("ERRORED", "<none>")]
print(f"records: {len(rows)}   with a recommendation: {len(solved)}")

# (compound, eps) -> {budget: record}
cell = defaultdict(dict)
for r in solved:
    cell[(r["compound_id"], float(r["eps"]))][int(r["n_iter"])] = r
paired = {k: v for k, v in cell.items() if all(b in v for b in BUDGETS)}
print(f"cells with a recommendation under at least one budget: {len(cell)}")
print(f"cells paired under both budgets                      : {len(paired)}")

# ---- agreement between the two budgets on the paired cells -------------------------
n = len(paired)
same_batch = same_cost = same_kpuu = 0
perm, costly, kpuu_flips = [], [], []
for key, v in sorted(paired.items()):
    a, b = v[2000], v[20000]
    sa, sb = batch(a["mlasp"]), batch(b["mlasp"])
    ca, cb = float(a["cost"]), float(b["cost"])
    same_batch += sa == sb
    same_cost += ca == cb
    same_kpuu += ("kpuu" in sa) == ("kpuu" in sb)
    if sa != sb:
        (perm if ca == cb else costly).append((key, sorted(sa), sorted(sb), ca, cb))
    if ("kpuu" in sa) != ("kpuu" in sb):
        kpuu_flips.append((key, sorted(sa), sorted(sb), ca, cb))

pct = lambda c: f"{c}/{n} ({round(100 * c / n)}%)"
print(f"\nexact MLASP batch agreement : {pct(same_batch)}")
print(f"total plan cost agreement   : {pct(same_cost)}")
print(f"kpuu-in-batch agreement     : {pct(same_kpuu)}")
print(f"\nbatch disagreements: {len(perm) + len(costly)}"
      f"  ({len(perm)} equal-cost permutations, {len(costly)} change the cost)")
# Where the permutations sit: the ESI claims 9 of 10 are at loose tolerance.
hi_perm = sum(1 for (_, eps), *_ in perm if eps >= 0.3)
print(f"   permutations at eps>=0.3: {hi_perm} of {len(perm)}"
      f"   (eps values: {sorted(eps for (_, eps), *_ in perm)})")
# Cost-changing cells: how many differ by exactly one $400 proxy, and how many are
# nested (one batch = the other plus a proxy) rather than differing in identity too.
one_proxy = nested = 0
for (cid, eps), sa, sb, ca, cb in costly:
    d = abs(ca - cb)
    nest = set(sa) <= set(sb) or set(sb) <= set(sa)
    one_proxy += d == 400
    nested += nest
    print(f"   {cid} eps={eps}: {sa} ${ca:,.0f}  vs  {sb} ${cb:,.0f}   "
          f"(d=${d:,.0f}, nested={nest})")
print(f"   of the {len(costly)} cost-changing cells: {one_proxy} differ by exactly $400, "
      f"{nested} are nested")
print(f"kpuu flips: {len(kpuu_flips)}")
for (cid, eps), sa, sb, ca, cb in kpuu_flips:
    print(f"   {cid} eps={eps}: {BUDGETS[0]}={sa} ${ca:,.0f} | {BUDGETS[1]}={sb} ${cb:,.0f}")

# ---- band-pooled medians and per-band kpuu agreement -------------------------------
BANDS = (("tight (eps<=%.1f)" % TIGHT, lambda e: e <= TIGHT),
         ("loose (eps>=%.1f)" % LOOSE, lambda e: e >= LOOSE))
# Reported on BOTH bases: all cells a budget solved, and only the paired cells.
# They agree on the two headline band medians; they do NOT agree on every per-eps
# median (see the per-eps table below), so the basis is stated explicitly.
print("\nband-pooled median first-batch cost")
for basis_name, basis in (("all cells", cell), ("paired only", paired)):
    for ni in BUDGETS:
        for name, keep in BANDS:
            c = [float(v[ni]["cost"]) for (_, eps), v in basis.items() if ni in v and keep(eps)]
            if c:
                print(f"   [{basis_name:11s}] n_iter={ni:<6} {name:18s} "
                      f"median=${st.median(c):>9,.0f}  n={len(c)}")
print("\nkpuu-criterion agreement by band")
for name, keep in BANDS:
    cells = [v for (_, eps), v in paired.items() if keep(eps)]
    ok = sum(1 for v in cells
             if ("kpuu" in batch(v[2000]["mlasp"])) == ("kpuu" in batch(v[20000]["mlasp"])))
    print(f"   {name:18s} {ok}/{len(cells)}")

# ---- the compound-level criterion behind the main text's count ---------------------
# It reads kpuu status at the TIGHTEST solved tolerance, not a disjunction over levels.
print("\ncompound-level criterion (kpuu at each compound's tightest solved tolerance)")
for cid in sorted({r["compound_id"] for r in solved}):
    line = [cid]
    for ni in BUDGETS:
        lv = sorted((float(r["eps"]), r) for r in solved
                    if r["compound_id"] == cid and int(r["n_iter"]) == ni)
        e, r = lv[0]
        line.append(f"{ni}: eps={e} kpuu={'kpuu' in batch(r['mlasp'])}")
    print("   " + "   ".join(line))

# ---- per-epsilon medians: how many of the levels differ? ---------------------------
# The basis MATTERS here, unlike for the band medians. On the paired cells 3 of 11
# levels differ; pooling every solved cell adds a 4th (eps=0.5), which is a dropout
# artifact, not a plan change: one compound solves that level under the reduced
# budget only, and its unpaired cheap cell pulls that arm's median down. The ESI
# quotes the paired figure and explains the discrepancy, so both are printed.
levels = sorted({eps for (_, eps) in cell})
for basis_name, basis in (("paired only", paired), ("all cells", cell)):
    print(f"\nper-eps median first-batch cost by budget  [{basis_name}]")
    ndiff = 0
    for eps in levels:
        m, n = {}, {}
        for ni in BUDGETS:
            c = [float(v[ni]["cost"]) for (_, e), v in basis.items() if e == eps and ni in v]
            m[ni] = st.median(c) if c else None
            n[ni] = len(c)
        ndiff += m[BUDGETS[0]] != m[BUDGETS[1]]
        f = lambda x: "n/a" if x is None else f"${x:,.0f}"
        flag = "" if m[BUDGETS[0]] == m[BUDGETS[1]] else "   <-- differs"
        print(f"   eps={eps:<5} {f(m[BUDGETS[0]]):>10} (n={n[BUDGETS[0]]})  "
              f"{f(m[BUDGETS[1]]):>10} (n={n[BUDGETS[1]]}){flag}")
    print(f"   levels differing: {ndiff} of {len(levels)}")

# ---- timings ----------------------------------------------------------------------
# One elapsed time per (compound, budget); it covers that compound's whole 11-level sweep.
t = {}
for r in rows:
    t.setdefault((r["compound_id"], int(r["n_iter"])), float(r["elapsed_s"]))
print("\nwall-clock per compound (whole 11-level sweep) and the budget ratio")
ratios = []
for cid in sorted({c for c, _ in t}):
    a, b = t.get((cid, 2000)), t.get((cid, 20000))
    nlv = [sum(1 for r in solved if r["compound_id"] == cid and int(r["n_iter"]) == ni)
           for ni in BUDGETS]
    if a and b:
        ratios.append(b / a)
        print(f"   {cid}  {a:>8.1f}s ({nlv[0]:>2}/11 solved)   {b:>8.1f}s ({nlv[1]:>2}/11)"
              f"   ratio={b / a:>6.1f}x")
for ni in BUDGETS:
    v = sorted(x for (_, b), x in t.items() if b == ni)
    print(f"   n_iter={ni:<6} median {st.median(v):>7.1f}s   mean {st.mean(v):>7.1f}s   "
          f"range {min(v):.1f}-{max(v):.1f}s   total {sum(v) / 3600:.2f} core-h")
    print(f"      cohort projection for 220 compounds: median-scaled "
          f"{220 * st.median(v) / 3600:.1f} core-h, mean-scaled {220 * st.mean(v) / 3600:.1f} core-h")
    # The right-skew figures the ESI quotes: the n-1 fastest, and the outlier's margin.
    print(f"      all but the slowest: {v[0]:.1f}-{v[-2]:.1f}s;  slowest {v[-1]:.1f}s "
          f"= {v[-1] / v[-2]:.1f}x the next slowest;  spread {v[-1] / v[0]:.1f}x")
if ratios:
    # Exact endpoints as well as rounded: a rounded endpoint cannot bound a range
    # (7.2727 rounds to 7.3 but is not >= 7.3).
    print(f"   budget ratio: median {st.median(ratios):.1f}x   "
          f"range {min(ratios):.1f}-{max(ratios):.1f}x   "
          f"(exact {min(ratios):.4f}-{max(ratios):.4f}x)")
    print(f"   both budgets together: {sum(t.values()) / 3600:.2f} core-h")
    print(f"   nominal gap to the four-compound budget: 10x iterations x 5x ensemble = 50x")
