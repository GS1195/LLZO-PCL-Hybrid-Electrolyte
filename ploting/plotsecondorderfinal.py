#!/usr/bin/env python3
"""
Multi-system P2(Δz) plotter from anisotropy CSV files (anisotropy_bins.py output).

Upgraded with the SAME plotting style as in compute_Rz_firstlayer_chunks.py:
  - font.size 22, bold
  - axes.linewidth 3
  - thick lines (default 4) (relevant for reference lines)
  - inward ticks, top/right on
  - style_axes() helper

For each system:
  • Wrap z_center around slab COM → Δz in (−Lz/2, +Lz/2]
  • Convert Δz from Å → nm
  • Compute:
        <r^2>  = <x^2> + <y^2> + <z^2>
        <cos^2θ> ≈ <z^2> / <r^2>
        P2      = 0.5 * (3 <cos^2θ> − 1)
    with error propagation from rx_unc, ry_unc, rz_unc.
  • Exclude bins whose original z_center lies inside the slab (no P2 in solid).
  • Shade slab region in Δz (nm).
  • Plot P2(Δz) as scatter with error bars (no lines).

Output:
  P2_multisystem_DeltaZ_with_errors_nm.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from pathlib import Path

# ================================================================
# 1.  LIST YOUR SYSTEMS HERE  (LABEL : CSV PATH)
# ================================================================
p2_files = {
   # "500K": "500/monomer_anisotropy_profile.csv",
    "600K": "600/monomer_anisotropy_profile.csv",
    "700K": "700/monomer_anisotropy_profile.csv",
    "800K": "800/monomer_anisotropy_profile.csv",
}
# ================================================================

X_SCALE = 0.1  # Å → nm for Δz

# ---------- house-style (from compute_Rz_firstlayer_chunks.py) ----------
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

def style_axes(ax, x_major=None, x_minor_div=5):
    if x_major is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
        ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor_div))
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Keep these at width=2/length=8 like your script (rcParams tick widths are huge otherwise)
    ax.tick_params(axis='both', which='major',
                   bottom=True, left=True, top=True, right=True,
                   direction='in', width=2, length=8, labelsize=22)
    ax.tick_params(axis='both', which='minor',
                   bottom=True, left=True, top=True, right=True,
                   direction='in', width=2, length=4, labelsize=22)

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)

# ---------- plot palettes ----------
COLORS  = plt.get_cmap("tab10")
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "*"]

def main():
    apply_house_style()

    fig, ax = plt.subplots(figsize=(10, 6))  # match your Rz plot footprint
    curve_idx = 0
    did_slab_shading = False
    all_x_nm = []

    # ---------- Loop through each system ----------
    for label, path in p2_files.items():
        path = Path(path)
        if not path.exists():
            print(f"[warn] File not found → {path}")
            continue

        df = pd.read_csv(path)
        if df.empty:
            print(f"[warn] Empty CSV → {path}")
            continue

        # --- box & COM ---
        Lz   = float(df["z_max"].iloc[0])
        half = 0.5 * Lz
        COM  = float(df["solid_COM_z"].iloc[0])

        # wrap function (Å) : (z − COM) → (−Lz/2, +Lz/2]
        wrap = lambda z: (z - COM + half) % Lz - half

        # Δz in Å (use z_center for plotting), then convert to nm
        Delta_z_A = wrap(df["z_center"])
        x_nm = Delta_z_A * X_SCALE
        all_x_nm.append(x_nm.to_numpy())

        # --- slab bounds in ORIGINAL z (for masking) ---
        solid_zmin = float(df["solid_zmin"].iloc[0])
        solid_zmax = float(df["solid_zmax"].iloc[0])

        # mask: bins whose original z_center lies inside the slab
        is_inside_slab = (df["z_center"] >= solid_zmin) & (df["z_center"] <= solid_zmax)

        # --- slab shading (only once, using this system as reference) ---
        if not did_slab_shading:
            slab_lo = wrap(solid_zmin)
            slab_hi = wrap(solid_zmax)
            wraps = slab_hi < slab_lo

            slab_lo_nm = slab_lo * X_SCALE
            slab_hi_nm = slab_hi * X_SCALE
            half_nm    = half * X_SCALE

            if wraps:
                ax.axvspan(-half_nm, slab_hi_nm, color="grey", alpha=0.15, zorder=0)
                ax.axvspan(slab_lo_nm,  half_nm, color="grey", alpha=0.15, zorder=0)
            else:
                ax.axvspan(slab_lo_nm, slab_hi_nm, color="grey", alpha=0.15, zorder=0)

            # COM at Δz = 0 nm
            ax.axvline(0.0, color="grey", linestyle="--", linewidth=3)
            did_slab_shading = True

        # --- component means and uncertainties ---
        x2  = df["rx_mean"].to_numpy()
        y2  = df["ry_mean"].to_numpy()
        z2  = df["rz_mean"].to_numpy()

        sx2 = df["rx_unc"].to_numpy()
        sy2 = df["ry_unc"].to_numpy()
        sz2 = df["rz_unc"].to_numpy()

        # <r^2> = <x^2> + <y^2> + <z^2>
        r2 = x2 + y2 + z2

        # base "good" mask + exclude slab region
        good = np.isfinite(r2) & (r2 > 0.0) & (~is_inside_slab.to_numpy())

        cos2 = np.full_like(r2, np.nan, dtype=float)

        # --- error propagation for cos^2θ = z2 / r2 ---
        dcos_dx2 = np.zeros_like(r2, dtype=float)
        dcos_dy2 = np.zeros_like(r2, dtype=float)
        dcos_dz2 = np.zeros_like(r2, dtype=float)

        dcos_dx2[good] = -z2[good] / (r2[good] ** 2)
        dcos_dy2[good] = -z2[good] / (r2[good] ** 2)
        dcos_dz2[good] = (x2[good] + y2[good]) / (r2[good] ** 2)

        var_cos2 = np.zeros_like(r2, dtype=float)
        var_cos2[good] = (
            (dcos_dx2[good] ** 2) * (sx2[good] ** 2) +
            (dcos_dy2[good] ** 2) * (sy2[good] ** 2) +
            (dcos_dz2[good] ** 2) * (sz2[good] ** 2)
        )
        sigma_cos2 = np.sqrt(var_cos2)

        cos2[good] = z2[good] / r2[good]

        # P2 and its uncertainty
        P2 = 0.5 * (3.0 * cos2 - 1.0)
        sigma_P2 = 1.5 * sigma_cos2

        # --- scatter only with error bars (no lines) ---
        color = COLORS(curve_idx % 10)
        mk = MARKERS[curve_idx % len(MARKERS)]

        ax.errorbar(
            x_nm[good], P2[good],
            yerr=sigma_P2[good],
            fmt=mk,
            mfc=color,
            mec=color,
            ecolor=color,
            elinewidth=2,
            capsize=4,
            linestyle="none",
            markersize=10,   # upgraded to match your Rz style
            label=label,
            zorder=3
        )

        curve_idx += 1

    # ---------- Axes styling ----------
    ax.set_xlabel(r"$ z$ (nm)", fontsize=25, fontweight="bold")
    ax.set_ylabel(r"$\langle P_2(\cos\theta)\rangle$", fontsize=25, fontweight="bold")
    ax.set_ylim(-1.05, 1.05)

    ax.axhline(0.0, color="black", linestyle=":", linewidth=3)

    # Use your style_axes helper (major tick every 1 nm, 5 minors)
    style_axes(ax, x_major=1.0, x_minor_div=5)

    # sensible x-limits from all data
    if all_x_nm:
        all_x_nm_concat = np.concatenate(all_x_nm)
        finite_x = all_x_nm_concat[np.isfinite(all_x_nm_concat)]
        if finite_x.size > 0:
            xmin, xmax = finite_x.min(), finite_x.max()
            margin = 0.05 * (xmax - xmin) if xmax > xmin else 0.5
            ax.set_xlim(xmin - margin, xmax + margin)

    # Legend: outside (clean, no overlap)
    ax.legend(frameon=False, fontsize=22, loc="upper right")

    fig.tight_layout()
    out_png = "P2_multisystem_DeltaZ_with_errors_nm.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"✔ Saved P2 multi-system Δz plot → {out_png}")

if __name__ == "__main__":
    main()
