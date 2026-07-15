"""
Author: Pavan Kumar

Restyled 4x4 grid of Response Spectra: Mean Prediction vs Recorded vs Std Dev band.

This keeps the same layout/content as your original (Standard Deviation band,
Recorded Response Spectra, Mean Prediction) but with a cleaner, modern style:
  - a single shared legend at the top instead of one per panel
  - a softer, colorblind-friendly palette
  - decluttered spines/gridlines
  - consistent fonts/ticks across panels
  - panel sub-titles showing event context (Mw, distance) instead of blank axes

HOW TO USE WITH YOUR REAL MODEL:
  Replace the `mean_pred` / `std_pred` placeholder block (marked below) with your
  actual `predictions = make_predictions(model, X_scaler, Y_scaler, testing_sample_x)`
  output (mean = predictions.mean(axis=0), std = predictions.std(axis=0)).
  Everything else (styling, layout) stays the same.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv('final_data.csv')

time_periods = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                0.15, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0]
cols = [f'T{tp:.3f}S' if tp not in (0.01,) else 'T0.010S' for tp in time_periods]
# use the exact column names from the csv
col_map = ['T0.010S','T0.020S','T0.030S','T0.040S','T0.050S','T0.060S','T0.070S',
           'T0.080S','T0.090S','T0.150S','T0.200S','T0.300S','T0.500S','T0.600S',
           'T0.700S','T0.800S','T0.900S','T1.000S','T1.200S','T1.500S','T2.000S',
           'T3.000S','T4.000S']

# pick 16 diverse, well-populated records for the grid
rng = np.random.default_rng(7)
sample_idx = rng.choice(df.index, size=16, replace=False)

# ------------------------------------------------------------------
# 2. Styling
# ------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.edgecolor': '#444444',
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.color': '#e6e6e6',
    'grid.linewidth': 0.7,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
})

BAND_COLOR = '#5B8FD9'      # soft blue for std-dev band
MEAN_COLOR = '#1B7A5A'      # deep teal-green for mean prediction
RECORD_COLOR = '#E0578C'    # muted magenta/pink for recorded spectra

fig, axes = plt.subplots(4, 4, figsize=(16, 12), dpi=150)
axes = axes.flatten()

for ax_i, (ax, idx) in enumerate(zip(axes, sample_idx)):
    row = df.loc[idx]
    true_values = row[col_map].values.astype(float)

    # --------------------------------------------------------------
    # PLACEHOLDER mean/std — swap this block for your real model output
    # --------------------------------------------------------------
    spline = make_interp_spline(time_periods, true_values, k=3)
    xnew = np.linspace(min(time_periods), max(time_periods), 200)
    mean_smooth = spline(xnew)
    # small illustrative uncertainty band (replace with real std from your CGAN)
    std_smooth = 0.12 * np.abs(mean_smooth) + 0.15 * np.abs(mean_smooth) * \
                 np.abs(np.sin(np.linspace(0, 3, 200)))
    # --------------------------------------------------------------

    ax.fill_between(xnew, mean_smooth - std_smooth, mean_smooth + std_smooth,
                     color=BAND_COLOR, alpha=0.55, linewidth=0, zorder=1)
    ax.plot(time_periods, true_values, color=RECORD_COLOR, linewidth=1.6,
            zorder=3)
    ax.plot(xnew, mean_smooth, color=MEAN_COLOR, linewidth=1.8, zorder=2)

    ax.set_xscale('log')
    ax.set_xlim([0.0099, 4])
    ax.grid(True, which='major', axis='both')
    ax.grid(False, which='minor')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title(f"Mw {row['eqm']:.1f} · R$_{{JB}}$ {row['dist']:.0f} km",
                 fontsize=9.5, color='#333333', pad=4)

    if ax_i % 4 == 0:
        ax.set_ylabel('$S_a$ (g)', fontsize=10)
    if ax_i >= 12:
        ax.set_xlabel('Time Period (s)', fontsize=10)

# ------------------------------------------------------------------
# 3. Single shared legend at the top
# ------------------------------------------------------------------
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor=BAND_COLOR, alpha=0.55, label='Standard Deviation'),
    Line2D([0], [0], color=RECORD_COLOR, lw=1.8, label='Recorded Response Spectra'),
    Line2D([0], [0], color=MEAN_COLOR, lw=1.8, label='Mean Prediction'),
]
fig.legend(handles=legend_elements, loc='upper center', ncol=3,
           bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=12)

fig.suptitle('') # keep clean, legend acts as the header
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('response_spectra_grid_restyled.png',
            dpi=200, bbox_inches='tight')
print("Saved.")
