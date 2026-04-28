#!/usr/bin/env python3
"""
Density-profile plotter with SAME LEFT/RIGHT z-ranges as the free-energy plot.
Produces ONE FIGURE containing two subplots:
    • Left panel  : z in [–50 Å, –20 Å] → [–5.0 nm, –2.0 nm]
    • Right panel : z in [+20 Å, +50 Å] → [+2.0 nm, +5.0 nm]

MODIFICATION:
    Li_total = ("LiTFSI Li" + "LLZO Li")
    Only Li_total is plotted for each temperature.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator, NullFormatter

#density_files = {
    #"80 LiTFSI (500K)": "500/results/mean_density.csv",
#    "600K, N(TFSI)": "600/results2/mean_density.csv",
#    "700K, N(TFSI)": "700/results2/mean_density.csv",
#    "800K, N(TFSI)": "800/results2/mean_density.csv",
#}

#LI_COMPONENTS = ["N_TFSI"]    # you may add more columns
#SMOOTH = 2

#X_SCALE = 0.1      # Å → nm
#Y_SCALE = 1000     # atoms/Å³ → atoms/nm³

# SAME plot window as energy plot
#LEFT_MIN  = -65.0 * X_SCALE
#LEFT_MAX  = -20.0 * X_SCALE
#RIGHT_MIN =  20.0 * X_SCALE
#RIGHT_MAX =  65.0 * X_SCALE

#OUT_PNG = "NTFSIdensity_left_right_panels.png"



#density_files = {
    #"80 LiTFSI (500K)": "500/results/mean_density.csv",
#    "600K, O(PCL)": "600/results2/mean_density.csv",
#    "700K, O(PCL)": "700/results2/mean_density.csv",
#    "800K, O(PCL)": "800/results2/mean_density.csv",
#}

#LI_COMPONENTS = ["ODB"]    # you may add more columns
#SMOOTH = 2

#X_SCALE = 0.1      # Å → nm
#Y_SCALE = 1000     # atoms/Å³ → atoms/nm³

# SAME plot window as energy plot
#LEFT_MIN  = -65.0 * X_SCALE
#LEFT_MAX  = -20.0 * X_SCALE
#RIGHT_MIN =  20.0 * X_SCALE
#RIGHT_MAX =  65.0 * X_SCALE

#OUT_PNG = "OPCLdensity_left_right_panels.png"


density_files = {
    #"80 LiTFSI (500K)": "500/results/mean_density.csv",
    "600K, OH": "600/results2/mean_density.csv",
    "700K, OH": "700/results2/mean_density.csv",
    "800K, OH": "800/results2/mean_density.csv",
}

LI_COMPONENTS = ["OH"]    # you may add more columns
SMOOTH = 4

X_SCALE = 0.1      # Å → nm
Y_SCALE = 1000     # atoms/Å³ → atoms/nm³

LEFT_MIN  = -65.0 * X_SCALE
LEFT_MAX  = -20.0 * X_SCALE
RIGHT_MIN =  20.0 * X_SCALE
RIGHT_MAX =  65.0 * X_SCALE

OUT_PNG = "OHdensity_left_right_panels.png"


#density_files = {
    #"80 LiTFSI (500K)": "500/results/mean_density.csv",
#    "600K, CH$_3$": "600/results2/mean_density.csv",
#    "700K, CH$_3$": "700/results2/mean_density.csv",
#    "800K, CH$_3$": "800/results2/mean_density.csv",
#}

#LI_COMPONENTS = ["CH3"]    # you may add more columns
#SMOOTH = 2

#X_SCALE = 0.1      # Å → nm
#Y_SCALE = 1000     # atoms/Å³ → atoms/nm³

# SAME plot window as energy plot
#LEFT_MIN  = -65.0 * X_SCALE
#LEFT_MAX  = -20.0 * X_SCALE
#RIGHT_MIN =  20.0 * X_SCALE
#RIGHT_MAX =  65.0 * X_SCALE

#OUT_PNG = "CH3density_left_right_panels.png"


# ───── SYSTEM PATHS ────────────────────────────────────────────
#density_files = {
    # "80 LiTFSI (500K)": "500/results/mean_density.csv",
#    "600K, Li$_{total}$": "600/results2/mean_density.csv",
#    "700K, Li$_{total}$": "700/results2/mean_density.csv",
#    "800K, Li$_{total}$": "800/results2/mean_density.csv",
#}

#LI_COMPONENTS = ["LiTFSI Li", "LLZO Li"]

#SMOOTH = 1

#X_SCALE = 0.1      # Å → nm
#Y_SCALE = 1000     # atoms/Å³ → atoms/nm³

# SAME plot window as energy plot
#LEFT_MIN  = -65.0 * X_SCALE
#LEFT_MAX  = -20.0 * X_SCALE
#RIGHT_MIN =  20.0 * X_SCALE
#RIGHT_MAX =  65.0 * X_SCALE

#OUT_PNG = "Litotaldensity_left_right_panels.png"

# ───── HOUSE STYLE ─────────────────────────────────────────────
plt.rcParams.update({
    "font.size":         20,
    "font.weight":       "bold",
    "axes.linewidth":    3,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.major.size":  10,
    "ytick.major.size":  10,
    "xtick.minor.width": 4,
    "ytick.minor.width": 4,
    "xtick.minor.size":  4,
    "ytick.minor.size":  4,
    "lines.linewidth":   4,
})

CMAP    = plt.get_cmap("tab10")
MARKERS = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>'] * 10

# ───── FIGURE WITH TWO SUBPLOTS ────────────────────────────────
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
curve_idx = 0

for sys_label, csv_path in density_files.items():
    df = pd.read_csv(csv_path)
    x_all = df["z_value"] * X_SCALE

    # Check required columns
    missing = [col for col in LI_COMPONENTS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {csv_path}: {missing}")

    # Build Li_total first
    df["Li_total"] = df[LI_COMPONENTS].sum(axis=1)

    # Smooth only Li_total
    if SMOOTH > 1:
        df["Li_total"] = df["Li_total"].rolling(window=SMOOTH, center=True).mean()

    y_all = df["Li_total"] * Y_SCALE

    # LEFT side mask
    maskL = (x_all >= LEFT_MIN) & (x_all <= LEFT_MAX)
    xL = x_all[maskL]
    yL = y_all[maskL]

    axL.plot(
        xL, yL,
        linestyle='-',
        color=CMAP(curve_idx % 10),
        marker=MARKERS[curve_idx % len(MARKERS)],
        markersize=4,
        label=sys_label
    )

    # RIGHT side mask
    maskR = (x_all >= RIGHT_MIN) & (x_all <= RIGHT_MAX)
    xR = x_all[maskR]
    yR = y_all[maskR]

    axR.plot(
        xR, yR,
        linestyle='-',
        color=CMAP(curve_idx % 10),
        marker=MARKERS[curve_idx % len(MARKERS)],
        markersize=4,
        label=sys_label
    )

    curve_idx += 1

# ───── AXES & STYLING — LEFT PANEL ─────────────────────────────
axL.set_title("(a) Left side ", fontsize=20, fontweight="bold")
axL.set_xlabel("z (nm)", fontsize=20, fontweight="bold")
axL.set_ylabel("Number density (atoms / nm$^{3}$)", fontsize=20, fontweight="bold")
axL.set_yscale("log")
axL.set_ylim(0.001, 50)
axL.xaxis.set_major_locator(MultipleLocator(1))
axL.xaxis.set_minor_locator(AutoMinorLocator(5))
axL.yaxis.set_major_locator(LogLocator(base=10.0))
axL.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
axL.yaxis.set_minor_formatter(NullFormatter())

axL.tick_params(axis='both', which='major',
                bottom=True, left=True, top=True, right=True,
                direction='in', width=2, length=8, labelsize=20)
axL.tick_params(axis='both', which='minor',
                bottom=True, left=True, top=True, right=True,
                direction='in', width=2, length=4)

# ───── AXES & STYLING — RIGHT PANEL ────────────────────────────
axR.set_title("(b) Right side", fontsize=20, fontweight="bold")
axR.set_xlabel("z (nm)", fontsize=20, fontweight="bold")
axR.set_yscale("log")
axR.set_ylim(0.001, 50)

axR.xaxis.set_major_locator(MultipleLocator(1))
axR.xaxis.set_minor_locator(AutoMinorLocator(5))
axR.yaxis.set_major_locator(LogLocator(base=10.0))
axR.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
axR.yaxis.set_minor_formatter(NullFormatter())

axR.tick_params(axis='both', which='major',
                bottom=True, left=True, top=True, right=True,
                direction='in', width=2, length=8, labelsize=20)
axR.tick_params(axis='both', which='minor',
                bottom=True, left=True, top=True, right=True,
                direction='in', width=2, length=4)

# Spines
for ax in [axL, axR]:
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)

axR.legend(frameon=False, fontsize=20, loc="upper right")

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300)

print("Final combined plot saved →", OUT_PNG)
