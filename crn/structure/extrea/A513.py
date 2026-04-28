#!/usr/bin/env python3
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import csv

# ──────────────────────────────────────────────────────────────
# Files & trajectory timing 123 
# ──────────────────────────────────────────────────────────────
u = mda.Universe("combine_system.dat", "position.lammpstrj",
                 topology_format="DATA", format="LAMMPSDUMP", dt=20)

# ──────────────────────────────────────────────────────────────
# Atom types (your mapping)
# ──────────────────────────────────────────────────────────────
base_types = {
    'Li (PCL)': 16,         # Li assigned as PCL Li in topology
    'Li (LLZO)': 17,        # Li in LLZO
    '=O (Free PCL)': 3,     # Carbonyl O in free PCL
    'N  (TFSI)': 14,        # Nitrogen in TFSI
}

# We will also output a derived "Li (Total)" by summing the two Li histograms
derived_names = ['Li (Total)']

# ──────────────────────────────────────────────────────────────
# Solid selection used for centering by COM (z only) — EDIT if needed
# ──────────────────────────────────────────────────────────────
solid_selection = "type 18 or type 19 or type 20"

# ──────────────────────────────────────────────────────────────
# Frame selection & binning
# ──────────────────────────────────────────────────────────────
frames_to_analyze = range(6000, len(u.trajectory), 1)   # increase the step for speed if needed
bin_width_A = 0.5                                   # Å

# Prepare fixed bins over [0, Lz)
u.trajectory[0]
Lz = float(u.dimensions[2])
if not np.isfinite(Lz) or Lz <= 0:
    raise RuntimeError("Invalid Lz from trajectory dimensions.")

bins = np.arange(0.0, Lz + 1e-6, bin_width_A)       # [0, Lz] stepping by bin_width_A
bin_centers_A = 0.5 * (bins[:-1] + bins[1:])

# Solid group (for COM centering)
solid_atoms = u.select_atoms(solid_selection)
if solid_atoms.n_atoms == 0:
    raise RuntimeError("Solid selection returned 0 atoms. Edit 'solid_selection' to match your solid.")

# Build selection objects for base types (faster than re-parsing each frame)
base_groups = {k: u.select_atoms(f"type {v}") for k, v in base_types.items()}

# Setup histogram accumulators
summed_hists = {k: np.zeros(len(bins) - 1, dtype=float) for k in base_types.keys()}

# ──────────────────────────────────────────────────────────────
# Pass over frames — center by COM(z), fold into [0, Lz), bin per group
# ──────────────────────────────────────────────────────────────
for frame in frames_to_analyze:
    u.trajectory[frame]
    u.atoms.wrap()

    COMz = float(solid_atoms.center_of_mass()[2])
    z_all = (u.atoms.positions[:, 2] - COMz) % Lz  # now in [0, Lz)

    for name, ag in base_groups.items():
        if ag.n_atoms:
            z_species = z_all[ag.indices]
            hist, _ = np.histogram(z_species, bins=bins, density=False)
        else:
            hist = np.zeros(len(bins) - 1, dtype=float)
        summed_hists[name] += hist

# Average across frames
nframes = len(frames_to_analyze)
avg_counts = {k: summed_hists[k] / nframes for k in summed_hists.keys()}

# Add derived Li (Total)
avg_counts['Li (Total)'] = avg_counts['Li (PCL)'] + avg_counts['Li (LLZO)']

# ──────────────────────────────────────────────────────────────
# Print a compact table
# ──────────────────────────────────────────────────────────────
print("\nAveraged counts per bin (z-centered by solid COM; bins fixed over [0, Lz))")
for name in ['Li (PCL)', 'Li (LLZO)', 'Li (Total)', '=O (Free PCL)', 'N  (TFSI)']:
    print(f"\n{name}")
    print(f"{'Bin Center (Å)':>15} {'Avg Count':>12}")
    y = avg_counts[name]
    for zc, cnt in zip(bin_centers_A, y):
        print(f"{zc:15.2f} {cnt:12.2f}")

# ──────────────────────────────────────────────────────────────
# Save CSV: per-bin average counts
# ──────────────────────────────────────────────────────────────
with open("zbin_counts.csv", "w", newline="") as f:
    w = csv.writer(f, delimiter=",")
    header = ["bin_center_A", "Li(PCL)", "Li(LLZO)", "Li(Total)", "O_FreePCL", "N_TFSI"]
    w.writerow(header)
    for i in range(len(bin_centers_A)):
        w.writerow([f"{bin_centers_A[i]:.3f}",
                    f"{avg_counts['Li (PCL)'][i]:.6g}",
                    f"{avg_counts['Li (LLZO)'][i]:.6g}",
                    f"{avg_counts['Li (Total)'][i]:.6g}",
                    f"{avg_counts['=O (Free PCL)'][i]:.6g}",
                    f"{avg_counts['N  (TFSI)'][i]:.6g}"])

# ──────────────────────────────────────────────────────────────
# Plot: grouped bars (x in nm). Bars: Li(PCL), Li(LLZO), Li(Total), O_FreePCL, N_TFSI
# ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))

species_order = ['Li (PCL)', 'Li (LLZO)', 'Li (Total)', '=O (Free PCL)', 'N  (TFSI)']
n_types = len(species_order)
bin_centers_nm = bin_centers_A * 0.1
bar_group_width = (bin_width_A * 0.1)

for idx, name in enumerate(species_order):
    y = avg_counts[name]
    offset = (-0.5 + (idx + 0.5) / n_types) * bar_group_width
    bars = ax.bar(bin_centers_nm + offset,
                  y,
                  width=bar_group_width / n_types,
                  align='center',
                  label=name)
    # numeric labels on bars (no ratios)
    bump = 0.01 * (np.nanmax(y) if np.any(np.isfinite(y)) else 1.0)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + bump,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)

ax.axvline(x=0.0, color='black', linewidth=1)
ax.set_xlabel('z (nm)')
ax.set_ylabel('Average number of atoms per bin')
ax.legend(ncol=2)
plt.tight_layout()
plt.savefig('average_counts_Li_split_plus_O_TFSI.png', dpi=300)
# plt.show()
