#!/usr/bin/env python3
"""
Density-profile plotter with house style (bold fonts, 2-pt spines,
ticks inside on bottom/left only, unique markers per curve).

Now: "LiTFSI Li" + "LLZO Li" → combined into one curve "Li".
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# ───── USER SETTINGS ────────────────────────────────────────────
CSV_FILE = "results/mean_density.csv"

# First load these columns. Two Li columns will be merged later.
SELECTED_INPUT = ["ODB", "LiTFSI Li", "LLZO Li", "N_TFSI"]

SMOOTH   = 1            # rolling-window; set 1 for no smoothing
X_SCALE  = 0.1          # Å → nm
Y_SCALE  = 1000         # atoms/Å³ → atoms/nm³
OUT_PNG  = "700LZOdensity_profiles_plot10with2s.png"
# ────────────────────────────────────────────────────────────────


# -------- 1. HOUSE STYLE ---------------------------------------
plt.rcParams.update({
    "font.size":        20,
    "font.weight":      "bold",
    "axes.linewidth":   2,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.width":2,
    "ytick.major.width":2,
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "xtick.minor.width":2,
    "ytick.minor.width":2,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,
    "lines.linewidth":  3,
})

CMAP = plt.get_cmap('tab10')

# -------- 2. LOAD DATA -----------------------------------------
df = pd.read_csv(CSV_FILE)
print("Loaded:", CSV_FILE)

# -------- 3. Combine Li columns --------------------------------
missing = [col for col in ["LiTFSI Li", "LLZO Li"] if col not in df.columns]
if missing:
    print("[ERROR] Missing Li columns:", missing)
    raise SystemExit

# Create new column "Li"
df["Li"] = df["LiTFSI Li"] + df["LLZO Li"]

# Now create the final SELECTED list with "Li" replacing both originals
SELECTED = ["ODB", "Li", "N_TFSI"]

# -------- 4. SMOOTHING -----------------------------------------
if SMOOTH > 1:
    for col in SELECTED:
        if col in df.columns:
            df[col] = df[col].rolling(window=SMOOTH, center=True).mean()


# -------- 5. PLOT ----------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

x = df["z_value"] * X_SCALE

for idx, col in enumerate(SELECTED):
    if col not in df.columns:
        print(f"[skip] '{col}' not found")
        continue

    y = df[col] * Y_SCALE

    ax.plot(
        x, y,
        linestyle='-',
        color=CMAP(idx),
        markersize=8,
        label=col
    )


# -------- 6. AXES & TICKS --------------------------------------
ax.set_xlabel("z (nm)", fontsize=22, fontweight='bold')
ax.set_ylabel("Number density (atoms / nm$^{3}$)", fontsize=22, fontweight='bold')
ax.set_yscale("log")

ax.xaxis.set_major_locator(MultipleLocator(1.0))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator())

ax.tick_params(axis='both', which='major',
               bottom=True, left=True,
               top=True, right=True,
               direction='in',
               width=2, length=8, labelsize=18)
ax.tick_params(axis='both', which='minor',
               bottom=True, left=True,
               top=True, right=True,
               direction='in',
               width=2, length=4, labelsize=18)

for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_visible(True)

ax.legend(frameon=False, fontsize=20, loc="lower right")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300)

print("Plot saved →", OUT_PNG)
