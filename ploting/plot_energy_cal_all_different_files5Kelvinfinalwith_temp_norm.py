#!/usr/bin/env python3
"""
Free-energy profiles for Li_all with per-side alignment.

• Y-axis:      (F(z) - F(0))/(k_B T)  for each curve (same reference F(0) per curve).
• Left side:   find F_min in z ∈ [-32, -22] Å, then shift z so that all
               left minima lie on one vertical line (reference = highest-T curve).
• Right side:  same, but z ∈ [22, 32] Å.
• X-axis in nm.

UPGRADED STYLE:
Uses the same style as compute_Rz_firstlayer_chunks.py:
  - font.size 22, bold
  - axes.linewidth 3
  - thick lines default
  - inward ticks, top/right on
  - style_axes() helper

MODIFICATION:
Produces TWO separate figures:
  1) freeEnergy_left_side_aligned.png
  2) freeEnergy_right_side_aligned.png

Also writes:
  freeEnergy_side_minima_and_shifts.csv
"""

import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# ── INPUTS ─────────────────────────────────────────────────────
free_energy_files = {
   # "500K": "500/results/free_energy/free_energy.csv",
    "600K": "600/results2/free_energy/free_energy.csv",
    "700K": "700/results2/free_energy/free_energy.csv",
    "800K": "800/results2/free_energy/free_energy.csv",
}
selected_columns = ["Li_all"]
smoothing_window = 4  # centered rolling window; set 1 to disable

OUT_LEFT_PNG  = "testfreeEnergy_left_side_aligned.png"
OUT_RIGHT_PNG = "testfreeEnergy_right_side_aligned.png"
OUT_CSV       = "testfreeEnergy_side_minima_and_shifts.csv"

# Windows where we SEARCH for the minima (Å)
LEFT_MIN_SEARCH_MIN  = -32.0
LEFT_MIN_SEARCH_MAX  = -22.0
RIGHT_MIN_SEARCH_MIN =  22.0
RIGHT_MIN_SEARCH_MAX =  32.0

# Ranges that we actually PLOT (Å) BEFORE shifting
LEFT_PLOT_MIN_Z  = -50.0
LEFT_PLOT_MAX_Z  = -20.0
RIGHT_PLOT_MIN_Z =  20.0
RIGHT_PLOT_MAX_Z =  50.0

# Å → nm conversion
X_SCALE = 0.1

kb = 1.380649E-23

# ── HOUSE STYLE (from compute_Rz_firstlayer_chunks.py) ─────────
def apply_house_style():
    plt.rcParams.update({
        "font.size":         25,
        "font.weight":       "bold",
        "axes.linewidth":    3,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.major.width": 8,
        "ytick.major.width": 8,
        "xtick.major.size":  10,
        "ytick.major.size":  10,
        "xtick.minor.width": 4,
        "ytick.minor.width": 4,
        "xtick.minor.size":  4,
        "ytick.minor.size":  4,
        "lines.linewidth":   4,
        "legend.frameon":    False,
    })

def style_axes(ax, x_major=None, x_minor_div=5, y_minor=True):
    if x_major is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
        ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor_div))
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if y_minor:
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(axis='both', which='major',
                   bottom=True, left=True, top=True, right=True,
                   direction='in', width=2, length=8, labelsize=22)
    ax.tick_params(axis='both', which='minor',
                   bottom=True, left=True, top=True, right=True,
                   direction='in', width=2, length=4, labelsize=22)

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)

CMAP    = plt.get_cmap("tab10")
MARKERS = ["o","s","^","v","D","P","X","<",">","*","h","H"]

# ── helpers ────────────────────────────────────────────────────
def pick_zcol(df):
    for cand in ("z_value", "z", "Z", "z(Å)", "z_A"):
        if cand in df.columns:
            return cand
    raise KeyError("No z column found (looked for: z_value, z, Z, z(Å), z_A).")

def parse_temperature(label):
    m = re.search(r"(\d+)\s*K", label)
    if m:
        return int(m.group(1))
    return None

def find_min_in_range(z, y, zmin, zmax):
    mask = (z >= zmin) & (z <= zmax) & np.isfinite(y)
    if not np.any(mask):
        return np.nan, np.nan
    idxs = np.where(mask)[0]
    local_idx = int(np.nanargmin(y[mask]))
    idx = idxs[local_idx]
    return float(z[idx]), float(y[idx])

def ref_at_zero(z, y, window=0.5):
    mask = (z >= -window) & (z <= window) & np.isfinite(y)
    if np.any(mask):
        return float(np.nanmean(y[mask]))
    idx = int(np.nanargmin(np.abs(z)))
    return float(y[idx])

def smooth_series(y, window):
    if window is None or window <= 1:
        return y
    s = pd.Series(y)
    return s.rolling(window=window, center=True).mean().to_numpy()

def main():
    apply_house_style()

    # ── load curves, smooth, compute F0 and minima ──────────────
    curves = []

    for label, filepath in free_energy_files.items():
        df = pd.read_csv(filepath)
        zcol = pick_zcol(df)

        z = df[zcol].to_numpy(float)
        if np.all(~np.isfinite(z)):
            print(f"[warn] all z are NaN in '{filepath}'.")
            continue

        for col in selected_columns:
            if col not in df.columns:
                print(f"[skip] '{col}' not in '{filepath}'. Available: {list(df.columns)}")
                continue

            y_raw = df[col].to_numpy(float)
            if np.all(~np.isfinite(y_raw)):
                print(f"[skip] '{label}' ({col}) has no finite values.")
                continue

            y_sm = smooth_series(y_raw, smoothing_window)

            F0 = ref_at_zero(z, y_sm, window=0.5)

            z_min_L, F_min_L = find_min_in_range(z, y_sm, LEFT_MIN_SEARCH_MIN, LEFT_MIN_SEARCH_MAX)
            z_min_R, F_min_R = find_min_in_range(z, y_sm, RIGHT_MIN_SEARCH_MIN, RIGHT_MIN_SEARCH_MAX)

            if np.isnan(z_min_L):
                print(f"[warn] no left minimum in [{LEFT_MIN_SEARCH_MIN},{LEFT_MIN_SEARCH_MAX}] Å "
                      f"for {label} ({col}); widening to [-50,-20] Å.")
                z_min_L, F_min_L = find_min_in_range(z, y_sm, -50.0, -20.0)
            if np.isnan(z_min_R):
                print(f"[warn] no right minimum in [{RIGHT_MIN_SEARCH_MIN},{RIGHT_MIN_SEARCH_MAX}] Å "
                      f"for {label} ({col}); widening to [20,50] Å.")
                z_min_R, F_min_R = find_min_in_range(z, y_sm, 20.0, 50.0)

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
                "F0":       F0,
                "z_min_L":  z_min_L,
                "F_min_L":  F_min_L,
                "z_min_R":  z_min_R,
                "F_min_R":  F_min_R,
            })

    if not curves:
        raise RuntimeError("No valid curves found.")

    # ── choose reference curve (highest T) ──────────────────────
    valid_T = [c for c in curves if c["T"] is not None]
    ref_curve = max(valid_T, key=lambda c: c["T"]) if valid_T else curves[0]

    T_ref = ref_curve["T"]
    z_min_L_ref = ref_curve["z_min_L"]
    z_min_R_ref = ref_curve["z_min_R"]

    print(f"Reference curve: '{ref_curve['label']}' "
          f"(T_ref={T_ref} K) with z_min_L = {z_min_L_ref:.3f} Å, "
          f"z_min_R = {z_min_R_ref:.3f} Å")

    # ── prepare CSV rows ────────────────────────────────────────
    rows = []
    print("\n=== Alignment summary (per curve) ===")

    # ── LEFT FIGURE ─────────────────────────────────────────────
    figL, axL = plt.subplots(figsize=(10, 6))
    # ── RIGHT FIGURE ────────────────────────────────────────────
    figR, axR = plt.subplots(figsize=(10, 6))

    curve_idx = 0

    for c in curves:
        label = c["label"]
        z     = c["z"]      # Å
        y_sm  = c["y_sm"]
        T     = c["T"]

        F0      = c["F0"]
        z_min_L = c["z_min_L"]
        F_min_L = c["F_min_L"]
        z_min_R = c["z_min_R"]
        F_min_R = c["F_min_R"]

        # energy relative to F(0), normalized by temperature: (F(z)-F(0))/(k_B T)
        if T is None:
            raise ValueError(f"Could not parse temperature from label '{label}'")
        y_rel = (y_sm - F0) / (kb * T)

        # x-shifts (Å) so minima align with ref
        delta_z_L = z_min_L - z_min_L_ref
        delta_z_R = z_min_R - z_min_R_ref

        # LEFT side data
        mask_left = (z >= LEFT_PLOT_MIN_Z) & (z <= LEFT_PLOT_MAX_Z) & np.isfinite(y_rel)
        if np.any(mask_left):
            zL = z[mask_left]
            yL = y_rel[mask_left]

            zL_shifted_nm = (zL - delta_z_L) * X_SCALE
            axL.scatter(
                zL_shifted_nm, yL,
                s=90,
                marker=MARKERS[curve_idx % len(MARKERS)],
                c=[CMAP(curve_idx % 10)],
                edgecolors="none",
                label=f"{label}"
            )

        # RIGHT side data
        mask_right = (z >= RIGHT_PLOT_MIN_Z) & (z <= RIGHT_PLOT_MAX_Z) & np.isfinite(y_rel)
        if np.any(mask_right):
            zR = z[mask_right]
            yR = y_rel[mask_right]

            zR_shifted_nm = (zR - delta_z_R) * X_SCALE
            axR.scatter(
                zR_shifted_nm, yR,
                s=90,
                marker=MARKERS[curve_idx % len(MARKERS)],
                c=[CMAP(curve_idx % 10)],
                edgecolors="none",
                label=f"{label}"
            )

        print(
            f"{label}: F(0) = {F0: .3e} J (= {F0/(kb*T):8.3f} in units of k_B T);  "
            f"left min at z = {z_min_L:7.3f} Å (Δz_L = {delta_z_L:7.3f} Å),  "
            f"right min at z = {z_min_R:7.3f} Å (Δz_R = {delta_z_R:7.3f} Å)"
        )

        rows.append({
            "label":                    label,
            "column":                   c["column"],
            "T[K]":                     T,
            "F0[J]":                    F0,
            "F0_over_kBT[-]":           F0 / (kb * T),
            "z_min_left[Å]":            z_min_L,
            "F_min_left[J]":            F_min_L,
            "F_min_left_over_kBT[-]":   F_min_L / (kb * T),
            "x_shift_left[Å]":          float(delta_z_L),
            "z_min_right[Å]":           z_min_R,
            "F_min_right[J]":           F_min_R,
            "F_min_right_over_kBT[-]":  F_min_R / (kb * T),
            "x_shift_right[Å]":         float(delta_z_R),
        })

        curve_idx += 1

    # ── style LEFT plot ─────────────────────────────────────────
    axL.set_xlabel("z (nm)", fontsize=25, fontweight="bold")
    axL.set_ylabel(r"$(F(z)-F(0))/(k_B T)$", fontsize=25, fontweight="bold")
    axL.set_xlim(LEFT_PLOT_MIN_Z * X_SCALE, LEFT_PLOT_MAX_Z * X_SCALE)
    style_axes(axL, x_major=0.5, x_minor_div=5, y_minor=True)
    axL.legend(frameon=False, fontsize=22, loc="upper left")
    figL.tight_layout()
    figL.savefig(OUT_LEFT_PNG, dpi=300)
    plt.close(figL)
    print(f"\nSaved LEFT figure → {OUT_LEFT_PNG}")

    # ── style RIGHT plot ────────────────────────────────────────
    axR.set_xlabel("z (nm)", fontsize=25, fontweight="bold")
    axR.set_ylabel(r"$(F(z)-F(0))/(k_B T)$", fontsize=25, fontweight="bold")
    axR.set_xlim(RIGHT_PLOT_MIN_Z * X_SCALE, RIGHT_PLOT_MAX_Z * X_SCALE)
    style_axes(axR, x_major=0.5, x_minor_div=5, y_minor=True)
    axR.legend(frameon=False, fontsize=22, loc="upper right")
    figR.tight_layout()
    figR.savefig(OUT_RIGHT_PNG, dpi=300)
    plt.close(figR)
    print(f"Saved RIGHT figure → {OUT_RIGHT_PNG}")

    # ── CSV ─────────────────────────────────────────────────────
    if rows:
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"Saved minima/shift table → {OUT_CSV}")

if __name__ == "__main__":
    main()
