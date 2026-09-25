#!/usr/bin/env python3
r"""ESI Figure S3 -- MLASP loci for the public PK clearance compound at two goal-likelihood floors.

Re-render of IBMDP_arxiv/pub_figures/adme_tau_comparison.py with three changes and no
others, so that every claim the ESI caption makes about the figure's geometry stays true:

  1. The axis labels named the wrong quantities.  The script set xlabel 'Cost' and ylabel
     'State Uncertainty', while the ESI caption for this very figure states in bold that
     "The horizontal axis is the planner's own objective value, not a dollar spend."  A
     figure whose axis contradicts its own caption is read off the axis, not the caption.
     They now name what is actually plotted: the ensemble mean planner objective, and the
     tolerance epsilon the design was solved at.  This is the same correction already
     applied to main Figure 2 (see replot_fig2.jl).
  2. pandas is not installed on the interpreter that has matplotlib here, so the two
     dataframe calls are replaced by the standard-library csv module.  The MLASP extraction
     is a faithful port: per threshold, descending, the row with the largest Frequency.
  3. `import numpy as np` is dropped.  Nothing in the script referenced it, and a released
     script should not fail on an interpreter that lacks a package it never uses.  This
     leaves matplotlib as the only third-party requirement.

Run with an interpreter that has matplotlib, against the two CSVs shipped here:
  python3 figures/esi_figS3_adme.py --data figures/data --out adme_tau_comparison.png
The two input CSVs (figures/data/ADME_ensemble_results_tau_0.{6,9}.csv) carry planner
outputs for a PUBLIC clearance dataset only -- tolerance, action set, ensemble mean
objective, vote count -- and no proprietary measurement, so they are released in full.
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams


def read_rows(path):
    """The two columns used downstream, typed; Action_Set stays a raw string."""
    out = []
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            out.append({
                'Threshold': float(r['Threshold']),
                'Average_Utility': float(r['Average_Utility']),
                'Frequency': float(r['Frequency']),
                'Action_Set': (r.get('Action_Set') or '').strip(),
            })
    return out


def notna(v):
    """Stand-in for pd.notna on the Action_Set field: an empty batch is 'missing'."""
    return bool(v) and v.lower() != 'nan'

# Set publication-quality parameters with LARGER fonts
# Emit TrueType (42) rather than matplotlib's default Type 3 (3) fonts.
# Type 3 subsets propagate into the compiled ESI PDF and are flagged by
# publisher preflight; the glyphs drawn are identical either way.
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 18  # Increased from 14
rcParams['axes.linewidth'] = 2.0
rcParams['xtick.major.size'] = 10
rcParams['ytick.major.size'] = 10
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['lines.linewidth'] = 3.2
rcParams['lines.markersize'] = 13
rcParams['mathtext.fontset'] = 'cm'  # Computer Modern for proper tau rendering
rcParams['mathtext.rm'] = 'serif'

def get_mlasp_path(rows):
    """Extract the MLASP: per threshold, descending, the plurality-vote batch."""
    by = defaultdict(list)
    for r in rows:
        by[r['Threshold']].append(r)
    mlasp_points = []
    for threshold in sorted(by, reverse=True):
        threshold_data = by[threshold]
        if len(threshold_data) > 0:
            max_point = max(threshold_data, key=lambda r: r['Frequency'])
            mlasp_points.append({
                'threshold': max_point['Threshold'],
                'cost': max_point['Average_Utility'],
                'action_set': max_point['Action_Set'],
                # how many ensemble members returned this batch, and how many returned
                # anything at all at this tolerance -- printed for the record because at
                # loose tolerance the second number is far below the nominal N_e
                'freq': max_point['Frequency'],
                'responding': sum(r['Frequency'] for r in threshold_data),
            })
    return mlasp_points


ap = argparse.ArgumentParser()
ap.add_argument('--data', default='.', help='directory holding the two ADME_ensemble_results_tau_*.csv')
ap.add_argument('--out', default='adme_tau_comparison.png')
args = ap.parse_args()

# Load both tau datasets
df_06 = read_rows(os.path.join(args.data, 'ADME_ensemble_results_tau_0.6.csv'))
df_09 = read_rows(os.path.join(args.data, 'ADME_ensemble_results_tau_0.9.csv'))

# Get MLASP paths
path_06 = get_mlasp_path(df_06)
path_09 = get_mlasp_path(df_09)

# Create the figure
fig, ax = plt.subplots(figsize=(14, 8))

# Plot tau = 0.6 path
costs_06 = [p['cost'] for p in path_06]
thresholds_06 = [p['threshold'] for p in path_06]
color_06 = '#D2691E'  # Darker chocolate/orange

ax.plot(costs_06, thresholds_06, 
        color=color_06,
        linestyle='--',
        linewidth=3.2,
        alpha=0.9,
        label=r'Maximum Likelihood Action Sets Path, $\tau$ = 0.6')

ax.scatter(costs_06, thresholds_06,
          color=color_06,
          s=180,
          marker='o',
          edgecolor='black',
          linewidth=0.8,
          alpha=0.7,
          zorder=5)

# Plot tau = 0.9 path
costs_09 = [p['cost'] for p in path_09]
thresholds_09 = [p['threshold'] for p in path_09]
color_09 = '#1E3A8A'  # Darker blue

ax.plot(costs_09, thresholds_09, 
        color=color_09,
        linestyle='--',
        linewidth=3.2,
        alpha=0.9,
        label=r'Maximum Likelihood Action Sets Path, $\tau$ = 0.9')

ax.scatter(costs_09, thresholds_09,
          color=color_09,
          s=180,
          marker='s',
          edgecolor='black',
          linewidth=0.8,
          alpha=0.7,
          zorder=5)

# NO OVERLAP ANNOTATION LOGIC - tau 0.6 on LEFT, tau 0.9 on RIGHT
def add_inflection_annotations_no_overlap(path, color, tau):
    prev_action = None
    annotation_y_positions = []  # Track y positions to avoid vertical overlaps
    
    for i, point in enumerate(path):
        current_action = str(point['action_set'])
        
        # Only annotate when action set changes from previous
        if current_action != prev_action and notna(point['action_set']):
            action_text = current_action.replace('[', '').replace(']', '').replace('"', '').replace("'", '')
            action_text = action_text.replace('CL_total_', '').replace('_exp', '')
            
            # Calculate y position with adjustment if too close to previous annotations
            y_pos = point['threshold']
            min_spacing = 0.06
            for existing_y in annotation_y_positions:
                if abs(y_pos - existing_y) < min_spacing:
                    y_pos = existing_y + min_spacing if y_pos > existing_y else existing_y - min_spacing
            annotation_y_positions.append(y_pos)
            
            # TAU 0.6 - ALWAYS ON LEFT
            if tau == 0.6:
                xytext = (point['cost'] - 2000, y_pos)  # Left side
                ha = 'right'
            # TAU 0.9 - ALWAYS ON RIGHT  
            else:  
                xytext = (point['cost'] + 2000, y_pos)  # Right side
                ha = 'left'
            
            # Add annotation with clean styling
            ax.annotate(action_text, 
                       xy=(point['cost'], point['threshold']),
                       xytext=xytext,
                       fontsize=14,
                       color=color, 
                       alpha=1.0,
                       ha=ha,
                       va='center',
                       weight='normal',
                       bbox=dict(boxstyle='round,pad=0.2', 
                               facecolor='white', 
                               edgecolor=color,
                               alpha=0.95,
                               linewidth=1.0),
                       arrowprops=dict(arrowstyle='->', 
                                     connectionstyle='arc3,rad=0.2',  # Curved arrows
                                     color=color, 
                                     alpha=0.6,
                                     lw=1.0))
        
        prev_action = current_action

# Add annotations for both paths
add_inflection_annotations_no_overlap(path_06, color_06, 0.6)
add_inflection_annotations_no_overlap(path_09, color_09, 0.9)

# NO TITLE (as requested)

# Customize axes.  These labels used to read 'Cost' and 'State Uncertainty', which the ESI
# caption for this figure explicitly contradicts: the abscissa is the planner's objective and
# not a spend, and the ordinate is the tolerance the design was solved at rather than a
# measured uncertainty.
ax.set_xlabel('Ensemble mean planner objective', fontsize=20, fontweight='bold')
ax.set_ylabel(r'Tolerance $\epsilon$', fontsize=20, fontweight='bold')

# Set axis limits with more space for annotations
max_cost = max(max(costs_06), max(costs_09))
ax.set_xlim(-3500, max_cost + 3500)  # More space on sides for annotations
ax.set_ylim(-0.05, 1.10)

# Customize ticks
if max_cost > 20000:
    ax.set_xticks([0, 5000, 10000, 15000, 20000, 25000, 30000])
    ax.set_xticklabels(['0', '5,000', '10,000', '15,000', '20,000', '25,000', '30,000'], fontsize=16)
else:
    ax.set_xticks([0, 5000, 10000, 15000, 20000])
    ax.set_xticklabels(['0', '5,000', '10,000', '15,000', '20,000'], fontsize=16)

ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=16)

# Add subtle grid
ax.grid(True, linestyle='--', alpha=0.2, linewidth=0.5)
ax.set_axisbelow(True)

# Create professional legend in UPPER RIGHT
legend = ax.legend(loc='upper right',
                  frameon=True,
                  fancybox=False,
                  shadow=False,
                  framealpha=0.95,
                  edgecolor='#666666',
                  fontsize=16)

# Style the legend
legend.get_frame().set_facecolor('white')
legend.get_frame().set_linewidth(1.0)

# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Make remaining spines slightly thicker
ax.spines['left'].set_linewidth(2.0)
ax.spines['bottom'].set_linewidth(2.0)

# Adjust layout
plt.tight_layout()

# Save the figure as both a vector PDF (what the manuscript includes) and a PNG for review
stem = os.path.splitext(args.out)[0]
for ext in ('.pdf', '.png'):
    plt.savefig(stem + ext, dpi=300, bbox_inches='tight')
    print(f"Figure saved as: {stem + ext}")

# The full loci, with the vote counts behind each point.  These are printed because the ESI
# caption quotes the abscissae and the batch identities, and because the "responding" column
# is the honest denominator: the nominal ensemble is 30 runs, but at loose tolerance only a
# handful of runs return any plan at all, so a locus there rests on very few votes.
PRICE = {'rat': 400, 'dog': 800, 'human': 4000}


def batch(p):
    s = str(p['action_set']).replace('CL_total_', '').replace('_exp', '')
    for ch in '[]"\'':
        s = s.replace(ch, '')
    return [t.strip() for t in s.split(',') if t.strip()]


for name, path in (('tau = 0.6', path_06), ('tau = 0.9', path_09)):
    print(f"\n=== {name}: {len(path)} tolerance levels solved ===")
    print("     eps   objective  votes/responding  price  batch")
    for p in path:
        b = batch(p)
        print(f"  {p['threshold']:6.2f} {p['cost']:11.1f} {p['freq']:8.0f}/{p['responding']:<8.0f}"
              f" ${sum(PRICE.get(x, 0) for x in b):>5} {b}")
    lo = min(p['responding'] for p in path)
    hi = max(p['responding'] for p in path)
    loose = [p for p in path if p['threshold'] > 0.1]
    print(f"  responding members range {lo:.0f}-{hi:.0f} of 30; "
          f"at eps>0.1 the modal batch carries "
          f"{min(p['freq'] for p in loose):.0f}-{max(p['freq'] for p in loose):.0f} votes")