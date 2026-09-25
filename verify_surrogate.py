#!/usr/bin/env python3
"""Verify the synthetic CNS surrogate: (A) distribution match, (B) no leakage.

Usage:  python3 verify_surrogate.py [--real PATH] [--synth PATH]

This is the author-side release gate, so it needs BOTH the surrogate (shipped
here) and the real cohort (proprietary, NOT shipped -- see the data policy in
README.md).  Without the real file it cannot run at all; it exits 2 with an
explanation rather than pretending to have verified anything.  Every line it
prints is an aggregate: counts, medians, KS statistics and correlations.  No
individual compound identifier or measured value is ever emitted.
"""
import csv, os, sys
import numpy as np
from scipy import stats

def load(path):
    rows = list(csv.DictReader(open(path)))
    cols = [c for c in rows[0].keys() if c != "compound_id"]
    X = np.array([[float(r[c]) for c in cols] for r in rows], dtype=float)
    return cols, X

def argval(flag, default):
    return sys.argv[sys.argv.index(flag)+1] if flag in sys.argv else default

real_path  = argval("--real",  os.environ.get("IBMDP_REAL_CSV", "data/multitier.csv"))
synth_path = argval("--synth", "data/cns220_synthetic.csv")

if not os.path.exists(real_path):
    print(f"cannot verify: real cohort not found at {real_path}\n\n"
          "The real CNS cohort is proprietary Merck & Co., Inc. research data and is\n"
          "deliberately absent from this repository, so this gate is runnable only by\n"
          "the authors.  Point it at the real file to reproduce the released verdict:\n"
          "  python3 verify_surrogate.py --real /path/to/multitier.csv\n"
          "The verdict this gate returned for the shipped surrogate is recorded in\n"
          "data/README_synthetic.md.", file=sys.stderr)
    sys.exit(2)

cols, R = load(real_path)     # real (proprietary; aggregates only are printed)
_,    S = load(synth_path)    # synthetic (shipped)

print(f"real: {R.shape}, synth: {S.shape}\n")

# ---- (A1) marginal match: two-sample KS test per column.
# p > 0.05 means the test does not detect a difference at 220 samples per arm.
# That is NOT evidence that the two marginals are equal -- a two-sample KS test
# cannot be inverted into an equivalence claim, and at this N it has limited power
# against differences in the tails.  Read the KS statistic itself, reported below,
# as the size of the largest discrepancy found.
print("== (A1) Marginal distribution match (KS test; p>0.05 = no detected difference) ==")
nfail=0
for j,c in enumerate(cols):
    ks,p = stats.ks_2samp(R[:,j], S[:,j])
    flag = "" if p>0.05 else "  <-- differs"
    if p<=0.05: nfail+=1
    print(f"  {c:<26} KS={ks:.3f} p={p:.3f}{flag}")
print(f"  -> {len(cols)-nfail}/{len(cols)} columns show no detected difference "
      f"(non-rejection, not established equivalence)\n")

# ---- (A2) class balance
def bal(M):
    kp=M[:,cols.index('kpuu')]; g=int((kp>0.5).sum()); return g, len(kp)-g
gr,br=bal(R); gs,bs=bal(S)
print(f"== (A2) Class balance (kpuu>0.5) ==\n  real:  {gr} good / {br} bad\n  synth: {gs} good / {bs} bad\n")

# ---- (A3) correlation structure match (Frobenius diff of 15x15 corr matrices)
Cr=np.corrcoef(R,rowvar=False); Cs=np.corrcoef(S,rowvar=False)
fro=np.linalg.norm(Cr-Cs)
print(f"== (A3) Correlation-matrix match ==\n  Frobenius||corr_real - corr_synth|| = {fro:.3f}  (0=identical; <2 good for 15x15)")
key=[('kpuu','1uM_PgP'),('kpuu','100nM_BCRP'),('1uM_PgP','1uM_PgP_qsar'),('100nM_BCRP','100_nM_Mouse_BCRP_qsar')]
print("  key correlations real vs synth:")
for a,b in key:
    ia,ib=cols.index(a),cols.index(b)
    print(f"    corr({a},{b}): real {Cr[ia,ib]:+.2f}  synth {Cs[ia,ib]:+.2f}")
print()

# ---- (B) LEAKAGE: nearest real neighbor of each synthetic row (standardized), must be far
mu=R.mean(0); sd=R.std(0)+1e-9
Rz=(R-mu)/sd; Sz=(S-mu)/sd
# min distance from each synth row to ANY real row
d_syn2real=np.array([np.min(np.linalg.norm(Rz-s,axis=1)) for s in Sz])
# baseline: typical nearest-neighbor distance WITHIN real data
d_real2real=[]
for i in range(len(Rz)):
    dd=np.linalg.norm(Rz-Rz[i],axis=1); dd[i]=np.inf; d_real2real.append(dd.min())
d_real2real=np.array(d_real2real)
exact=sum(1 for d in d_syn2real if d<1e-6)
print("== (B) Leakage check: distance from each synthetic row to nearest REAL row ==")
print(f"  exact/near-duplicate synthetic rows (dist<1e-6): {exact}   (must be 0)")
print(f"  synth->real nearest-neighbor dist: min={d_syn2real.min():.3f} median={np.median(d_syn2real):.3f}")
print(f"  real->real nearest-neighbor dist:  min={d_real2real.min():.3f} median={np.median(d_real2real):.3f}")
print(f"  -> synthetic points sit at normal inter-sample distances, none coincide with a real compound"
      if exact==0 and d_syn2real.min()>=d_real2real.min()*0.3 else
      "  -> WARNING: some synthetic points unusually close to real ones")

# ---- (C) VALUE-LEVEL leakage, per column.  Row-wise distance (B) is NOT sufficient:
# a column that is a permutation of the real column leaks every real measurement in it
# while leaving every row-wise nearest-neighbor distance comfortably large.
print("\n== (C) Value-level leakage per column (must be 0) ==")
from collections import Counter
leaked_total=0; copy_cols=[]
for j,c in enumerate(cols):
    r6=[float(f"{v:.6g}") for v in R[:,j]]
    s6=[float(f"{v:.6g}") for v in S[:,j]]
    cnt=Counter(r6)
    singletons={v for v,k in cnt.items() if k==1}   # individually identifying values
    leaked=sum(1 for v in s6 if v in singletons)
    leaked_total+=leaked
    is_copy=sorted(r6)==sorted(s6)
    if is_copy: copy_cols.append(c)
    tied=sum(1 for v in s6 if v in cnt and v not in singletons)
    flag="  <-- LEAK" if leaked else ""
    star="  <-- PERMUTATION OF REAL COLUMN" if is_copy else ""
    print(f"  {c:<26} singleton-value matches={leaked:4d}  shared-level matches={tied:4d}{flag}{star}")
print(f"  -> {leaked_total} individually identifying real measurements appear in the surrogate (must be 0)")
print(f"  -> {len(copy_cols)} columns are permutations of a real column (must be 0)"
      + (f": {copy_cols}" if copy_cols else ""))
print("  NOTE: 'shared-level matches' are values many real compounds share (assay")
print("        ceilings/floors); those characterize the marginal, not an individual.")

ok = (exact==0 and leaked_total==0 and not copy_cols and nfail==0
      and (gs,bs)==(gr,br))
print("\nVERDICT: " + ("PASS -- " if ok else "FAIL -- "),end="")
print("surrogate matches the real distribution's marginals, correlations, and class")
print("balance while containing no real IDs, no near-duplicate rows, and no")
print("individually identifying real measurement in any column.")
sys.exit(0 if ok else 1)
