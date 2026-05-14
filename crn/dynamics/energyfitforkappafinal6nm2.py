#!/usr/bin/env python3
"""
Free-energy profile with TWO local curvature fits:
(1) upward parabola around minimum z0:      F(z) ≈ F(z0) + k_min (z-z0)^2
(2) downward parabola around barrier zb:    F(z) ≈ F(zb) - k_top (z-zb)^2

Y-axis shown in Kelvin: ΔF(z)/kB relative to F(z0) (PCCP-style).

Modified:
- removed the text block from the plot
- x-axis is shown in nm
- top/right ticks added
- fit curves in legend now use short names only
- full fitted functions are printed in terminal for later use in text
"""

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# =========================
# USER SETTINGS
# =========================
CSV_FILE = "free_energy.csv"

PLOT_Z_MIN, PLOT_Z_MAX = 10.0, 35.0   # Å (internal)

# --- region to locate the WELL MINIMUM (z0) ---
MIN_WIN_Z_MIN, MIN_WIN_Z_MAX = 22.0, 35.0   # Å 22 35

# --- region to locate and fit the BARRIER TOP (downward parabola) ---
TOP_WIN_Z_MIN, TOP_WIN_Z_MAX = 21.8, 23.8   # Å

# Fit half-width (use FIXED for clarity)
H_MIN_FIT = 1.25   # Å around z0
H_TOP_FIT = 1   # Å around zb (barrier top)

SMOOTH_N = 1

kB  = 1.380649e-23
mLi = 1.15e-26

ANG_TO_NM = 0.1

OUT_PNG = "fit_free_energy_min_and_top_parabola_nm_1_25ztop1.png"
# =========================

# -------- HOUSE STYLE --------
plt.rcParams.update({
    "font.size": 22,
    "font.weight": "bold",
    "axes.linewidth": 3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 8,
    "ytick.major.width": 8,
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "xtick.minor.width": 4,
    "ytick.minor.width": 4,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 4,
})

def moving_average(y, n):
    if n <= 1:
        return y.copy()
    kernel = np.ones(n) / n
    ypad = np.pad(y, (n//2, n-1-n//2), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")

def lsq_k_from_y_equals_kx2(x, y):
    """Least-squares for y ≈ k x^2 (one-parameter)."""
    a = x**2
    denom = np.sum(a*a)
    if denom <= 0:
        return np.nan
    return np.sum(a*y) / denom

def main():
    df = pd.read_csv(CSV_FILE)
    z = df["z_value"].to_numpy(float)     # Å
    F = df["Li_all"].to_numpy(float)

    srt = np.argsort(z)
    z, F = z[srt], F[srt]

    # plot window
    mp = (z >= PLOT_Z_MIN) & (z <= PLOT_Z_MAX)
    zP, FP = z[mp], F[mp]   # zP in Å

    # ---------------------------
    # (A) Find minimum z0 in MIN window and fit UP parabola
    # ---------------------------
    mmin = (z >= MIN_WIN_Z_MIN) & (z <= MIN_WIN_Z_MAX)
    zMin, FMin = z[mmin], F[mmin]
    FMin_s = moving_average(FMin, SMOOTH_N)

    i0 = np.argmin(FMin_s)
    z0, F0 = zMin[i0], FMin[i0]   # z0 in Å

    # fit around z0 (upward): F - F0 ≈ k_min (z-z0)^2
    mf_minfit = np.abs(zP - z0) <= H_MIN_FIT
    x_min = zP[mf_minfit] - z0
    y_min = FP[mf_minfit] - F0
    k_min = lsq_k_from_y_equals_kx2(x_min, y_min)  # J/Å^2

    # attempt frequency from k_min
    k_min_SI = k_min * 1e20  # J/Å^2 -> J/m^2
    nu_min = (1/(2*np.pi)) * np.sqrt((2*k_min_SI) / mLi)

    # ---------------------------
    # (B) Find barrier top zb in TOP window and fit DOWN parabola
    # ---------------------------
    mtop = (z >= TOP_WIN_Z_MIN) & (z <= TOP_WIN_Z_MAX)
    zTop, FTop = z[mtop], F[mtop]
    FTop_s = moving_average(FTop, SMOOTH_N)

    ib = np.argmax(FTop_s)
    zb, Fb = zTop[ib], FTop[ib]   # zb in Å

    # fit around zb (downward): Fb - F ≈ k_top (z-zb)^2
    mf_topfit = np.abs(zP - zb) <= H_TOP_FIT
    x_top = zP[mf_topfit] - zb
    y_top = Fb - FP[mf_topfit]
    k_top = lsq_k_from_y_equals_kx2(x_top, y_top)  # J/Å^2 (positive)

    # barrier (imaginary) frequency magnitude from |curvature|
    k_top_SI = k_top * 1e20
    nu_top = (1/(2*np.pi)) * np.sqrt((2*k_top_SI) / mLi)

    # barrier height relative to well minimum
    dF = Fb - F0
    dF_K = dF / kB

    z0_nm = z0 * ANG_TO_NM
    zb_nm = zb * ANG_TO_NM

    print("[RESULT] Minimum (upward fit)")
    print(f"z0 = {z0:.3f} Å  ({z0_nm:.3f} nm)")
    print(f"k_min = {k_min:.3e} J Å^-2")
    print(f"nu_min = {nu_min:.3e} s^-1")

    print("[RESULT] Barrier top (downward fit)")
    print(f"zb = {zb:.3f} Å  ({zb_nm:.3f} nm)")
    print(f"k_top = {k_top:.3e} J Å^-2")
    print(f"nu_top = {nu_top:.3e} s^-1")

    print("[RESULT] Barrier height")
    print(f"ΔF‡/kB = {dF_K:.0f} K   (Fb - F0)")

    # full fitted functions for later text
    print("\n[FULL FIT FUNCTIONS]")
    print(f"Minimum well fit:   F(z) = {F0:.6e} + ({k_min:.6e}) * (z - {z0:.6f})^2    [z in Å, F in J]")
    print(f"Barrier-top fit:    F(z) = {Fb:.6e} - ({k_top:.6e}) * (z - {zb:.6f})^2    [z in Å, F in J]")

    # ---------------------------
    # PLOT: ΔF(z)/kB relative to F0
    # X-axis displayed in nm
    # ---------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    zP_nm = zP * ANG_TO_NM

    ax.plot(
        zP_nm, (FP - F0)/kB,
        marker="o", markersize=4,
        label=r"Free-energy"
    )

    # markers
    ax.scatter([z0_nm], [0], s=90, marker="s") #, label=r"Minimum $z_0$")
    ax.scatter([zb_nm], [(Fb-F0)/kB], s=90, marker="^") #label=r"Barrier top $z_b$")

    # fitted parabolas
    zfit_min = np.linspace(z0-H_MIN_FIT, z0+H_MIN_FIT, 200)
    Ffit_min = F0 + k_min*(zfit_min-z0)**2
    ax.plot(
        zfit_min*ANG_TO_NM, (Ffit_min - F0)/kB, "--",color='red',lw=4,
        #label="Minimum-fit function"
    )

    zfit_top = np.linspace(zb-H_TOP_FIT, zb+H_TOP_FIT, 200)
    Ffit_top = Fb - k_top*(zfit_top-zb)**2
    ax.plot(
        zfit_top*ANG_TO_NM, (Ffit_top - F0)/kB, "--"
        #label="Barrier-top fit"
    )

    ax.set_xlabel("z (nm)", fontsize=25, fontweight="bold")
    ax.set_ylabel(r"$\Delta F(z)/k_B$ (K)", fontsize=25, fontweight="bold")

    # 5 Å = 0.5 nm
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(
        axis="both", which="major",
        direction="in", top=True, right=True,
        width=2, length=8, labelsize=25
    )
    ax.tick_params(
        axis="both", which="minor",
        direction="in", top=True, right=True,
        width=2, length=4
    )

    for s in ax.spines.values():
        s.set_visible(True)

    ax.legend(frameon=False, fontsize=22, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300)
    print(f"[OK] Saved plot: {OUT_PNG}")

if __name__ == "__main__":
    main()
