#!/usr/bin/env python3
"""
Free-energy profiles for Li_all with per-side alignment:

For EACH temperature curve:
  • Left side:
      - In a specified z-range [LEFT_MIN_SEARCH_MIN, LEFT_MIN_SEARCH_MAX],
        find the minimum of F(z) (Li_all).
      - Shift energies so that this minimum is at 0.
      - Shift z so that this minimum coincides with the left-minimum of the
        reference curve (800 K).
      - Plot only the left side (z <= LEFT_PLOT_MAX_Z).

  • Right side:
      - In [RIGHT_MIN_SEARCH_MIN, RIGHT_MIN_SEARCH_MAX], find the minimum.
      - Shift energies so that this minimum is at 0.
      - Shift z so that this minimum coincides with the right-minimum of the
        reference curve (800 K).
      - Plot only the right side (z >= RIGHT_PLOT_MIN_Z).

Outputs
-------
PNG:
    freeEnergy_left_side_aligned.png
    freeEnergy_right_side_aligned.png
CSV:
    freeEnergy_side_minima_and_shifts.csv
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# ── INPUTS ─────────────────────────────────────────────────────
free_energy_files = {
    "80 LiTFSI (500K)": "500/results/free_energy/free_energy.csv",
    "80 LiTFSI (600K)": "600/results2/free_energy/free_energy.csv",
    "80 LiTFSI (700K)": "700/results2/free_energy/free_energy.csv",
    "80 LiTFSI (800K)": "800/results2/free_energy/free_energy.csv",
}
selected_columns = ["Li_all"]     # columns to use from each CSV
smoothing_window = 4              # centered rolling window; set 1 to disable

OUT_PNG_LEFT  = "freeEnergy_left_side_aligned.png"
OUT_PNG_RIGHT = "freeEnergy_right_side_aligned.png"
OUT_CSV       = "freeEnergy_side_minima_and_shifts.csv"

# Ranges where we SEARCH for the minima (Å).
# Adjust these if needed.
LEFT_MIN_SEARCH_MIN  = -32.0
LEFT_MIN_SEARCH_MAX  = -22.0
RIGHT_MIN_SEARCH_MIN =  22.0
RIGHT_MIN_SEARCH_MAX =  32.0

# Ranges that we actually PLOT (Å).
LEFT_PLOT_MIN_Z  = -40.0
LEFT_PLOT_MAX_Z  = -22.0      # left panel will show z in this range (before shift)
RIGHT_PLOT_MIN_Z =  22.0      # right panel will show z in this range (before shift)
RIGHT_PLOT_MAX_Z =  40.0

kb = 1.380649E-23

# ── HOUSE STYLE ────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":         14,
    "font.weight":       "bold",
    "axes.linewidth":    2,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.major.size":  10,
    "ytick.major.size":  10,
    "xtick.minor.width": 1,
    "ytick.minor.width": 1,
    "xtick.minor.size":  4,
    "ytick.minor.size":  4,
})
CMAP    = plt.get_cmap("tab10")
MARKERS = ["o","s","^","v","D","P","X","<",">","*","h","H"]

# ── helpers ────────────────────────────────────────────────────
def pick_zcol(df):
    for cand in ("z_value", "z", "Z", "z(Å)", "z_A"):
        if cand in df.columns:
            return cand
    raise KeyError("No z column found (looked for: z_value, z, Z, z(Å), z_A).")

def parse_temperature(label):
    """
    Extract temperature in K from label like '80 LiTFSI (700K)'.
    Returns int or None if not found.
    """
    m = re.search(r"\((\d+)\s*K\)", label)
    if m:
        return int(m.group(1))
    return None

def find_min_in_range(z, y, zmin, zmax):
    """Return (z_min, y_min) for y inside [zmin, zmax]."""
    mask = (z >= zmin) & (z <= zmax) & np.isfinite(y)
    if not np.any(mask):
        return np.nan, np.nan
    idxs = np.where(mask)[0]
    local_idx = int(np.nanargmin(y[mask]))
    idx = idxs[local_idx]
    return float(z[idx]), float(y[idx])

# ── first pass: load curves, smooth, find minima ──────────────
curves = []

for label, filepath in free_energy_files.items():
    df = pd.read_csv(filepath)
    zcol = pick_zcol(df)

    # smoothing
    df_s = df.copy()
    if smoothing_window and smoothing_window > 1:
        for col in selected_columns:
            if col in df_s.columns:
                df_s[col] = df_s[col].rolling(window=smoothing_window, center=True).mean()

    z = df_s[zcol].to_numpy()
    if np.all(~np.isfinite(z)):
        print(f"[warn] all z are NaN in '{filepath}'.")
        continue

    for col in selected_columns:
        if col not in df_s.columns:
            print(f"[skip] '{col}' not in '{filepath}'. Available: {list(df_s.columns)}")
            continue

        y_sm = df_s[col].to_numpy()
        if np.all(~np.isfinite(y_sm)):
            print(f"[skip] '{label}' ({col}) has no finite values.")
            continue

        # left & right minima in specified ranges
        z_min_L, F_min_L = find_min_in_range(z, y_sm,
                                             LEFT_MIN_SEARCH_MIN,
                                             LEFT_MIN_SEARCH_MAX)
        z_min_R, F_min_R = find_min_in_range(z, y_sm,
                                             RIGHT_MIN_SEARCH_MIN,
                                             RIGHT_MIN_SEARCH_MAX)

        # if missing, we can widen the window a bit as fallback
        if np.isnan(z_min_L):
            print(f"[warn] no left minimum in [{LEFT_MIN_SEARCH_MIN},{LEFT_MIN_SEARCH_MAX}] Å "
                  f"for {label} ({col}); widening to [-60,-10] Å.")
            z_min_L, F_min_L = find_min_in_range(z, y_sm, -60.0, -10.0)
        if np.isnan(z_min_R):
            print(f"[warn] no right minimum in [{RIGHT_MIN_SEARCH_MIN},{RIGHT_MIN_SEARCH_MAX}] Å "
                  f"for {label} ({col}); widening to [10,60] Å.")
            z_min_R, F_min_R = find_min_in_range(z, y_sm, 10.0, 60.0)

        if np.isnan(z_min_L) or np.isnan(z_min_R):
            print(f"[skip] could not find minima for {label} ({col}).")
            continue

        T = parse_temperature(label)

        curves.append({
            "label":    label,
            "filepath": filepath,
            "column":   col,
            "T":        T,
            "z":        z,
            "y_sm":     y_sm,
            "z_min_L":  z_min_L,
            "F_min_L":  F_min_L,
            "z_min_R":  z_min_R,
            "F_min_R":  F_min_R,
        })

if not curves:
    raise RuntimeError("No valid curves found.")

# ── choose reference curve (highest T, i.e. 800 K) ────────────
valid_T = [c for c in curves if c["T"] is not None]
if valid_T:
    ref_curve = max(valid_T, key=lambda c: c["T"])
else:
    ref_curve = curves[0]

T_ref = ref_curve["T"]
z_min_L_ref = ref_curve["z_min_L"]
z_min_R_ref = ref_curve["z_min_R"]

print(f"Reference curve: '{ref_curve['label']}' "
      f"(T_ref={T_ref} K) with z_min_L = {z_min_L_ref:.3f} Å, "
      f"z_min_R = {z_min_R_ref:.3f} Å")

# ── second pass: apply per-side shifts & plot ─────────────────
figL, axL = plt.subplots(figsize=(10.5, 6.2))
figR, axR = plt.subplots(figsize=(10.5, 6.2))

rows = []
curve_idx = 0

for c in curves:
    label = c["label"]
    z     = c["z"]
    y_sm  = c["y_sm"]
    T     = c["T"]

    z_min_L = c["z_min_L"]
    F_min_L = c["F_min_L"]
    z_min_R = c["z_min_R"]
    F_min_R = c["F_min_R"]

    # x-shifts to align minima with reference (per side)
    if T is not None and T_ref is not None:
        delta_z_L = z_min_L - z_min_L_ref
        delta_z_R = z_min_R - z_min_R_ref
    else:
        delta_z_L = 0.0
        delta_z_R = 0.0

    # LEFT SIDE PLOT: use data in [LEFT_PLOT_MIN_Z, LEFT_PLOT_MAX_Z]
    mask_left = (z >= LEFT_PLOT_MIN_Z) & (z <= LEFT_PLOT_MAX_Z) & np.isfinite(y_sm)
    if np.any(mask_left):
        zL = z[mask_left].copy()
        yL = y_sm[mask_left].copy()

        # vertical shift so minimum becomes 0
        yL_shifted = (yL - F_min_L) / kb
        # horizontal shift to align minima
        zL_shifted = zL - delta_z_L

        axL.scatter(
            zL_shifted*0.1, yL_shifted,
            s=22,
            marker=MARKERS[curve_idx % len(MARKERS)],
            c=[CMAP(curve_idx % 10)],
            edgecolors="none",
            label=f"{label} ({c['column']})"
        )

    # RIGHT SIDE PLOT: use data in [RIGHT_PLOT_MIN_Z, RIGHT_PLOT_MAX_Z]
    mask_right = (z >= RIGHT_PLOT_MIN_Z) & (z <= RIGHT_PLOT_MAX_Z) & np.isfinite(y_sm)
    if np.any(mask_right):
        zR = z[mask_right].copy()
        yR = y_sm[mask_right].copy()

        yR_shifted = (yR - F_min_R) / kb
        zR_shifted = zR - delta_z_R

        axR.scatter(
            zR_shifted*0.1, yR_shifted,
            s=22,
            marker=MARKERS[curve_idx % len(MARKERS)],
            c=[CMAP(curve_idx % 10)],
            edgecolors="none",
            label=f"{label} ({c['column']})"
        )

    # Store info for CSV
    rows.append({
        "label":          label,
        "column":         c["column"],
        "T[K]":           T,
        "z_min_left[Å]":  z_min_L,
        "F_min_left[J]":  F_min_L,
        "x_shift_left[Å]":  float(delta_z_L),
        "z_min_right[Å]": z_min_R,
        "F_min_right[J]": F_min_R,
        "x_shift_right[Å]": float(delta_z_R),
    })

    curve_idx += 1

# ── styling: LEFT ─────────────────────────────────────────────
axL.set_xlabel("z (nm)", fontsize=18, fontweight="bold")
axL.set_ylabel("(F(z) − F_min,left)/k$_B$ (K)", fontsize=18, fontweight="bold")
axL.xaxis.set_minor_locator(AutoMinorLocator())
axL.yaxis.set_minor_locator(AutoMinorLocator())
axL.tick_params(axis='both', which='major',
                bottom=True, left=True, top=True, right=True,
                direction='in', width=2, length=8, labelsize=16)
axL.tick_params(axis='both', which='minor',
                bottom=True, left=True, top=True, right=True,
                direction='in', width=1, length=4)
for s in axL.spines.values():
    s.set_visible(True)
    s.set_linewidth(2)
axL.legend(frameon=False, fontsize=12, loc="upper left")
figL.tight_layout()
figL.savefig(OUT_PNG_LEFT, dpi=300)
plt.close(figL)
print(f"Saved left-side figure → {OUT_PNG_LEFT}")

# ── styling: RIGHT ────────────────────────────────────────────
axR.set_xlabel("z (nm)", fontsize=18, fontweight="bold")
axR.set_ylabel("(F(z) − F_min,right)/k$_B$ (K)", fontsize=18, fontweight="bold")
axR.xaxis.set_minor_locator(AutoMinorLocator())
axR.yaxis.set_minor_locator(AutoMinorLocator())
axR.tick_params(axis='both', which='major',
                bottom=True, left=True, top=True, right=True,
                direction='in', width=2, length=8, labelsize=16)
axR.tick_params(axis='both', which='minor',
                bottom=True, left=True, top=True, right=True,
                direction='in', width=1, length=4)
for s in axR.spines.values():
    s.set_visible(True)
    s.set_linewidth(2)
axR.legend(frameon=False, fontsize=12, loc="upper right")
figR.tight_layout()
figR.savefig(OUT_PNG_RIGHT, dpi=300)
plt.close(figR)
print(f"Saved right-side figure → {OUT_PNG_RIGHT}")

# ── CSV ───────────────────────────────────────────────────────
if rows:
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Saved minima/shift table → {OUT_CSV}")
