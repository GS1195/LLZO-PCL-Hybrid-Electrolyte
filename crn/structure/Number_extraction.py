#!/usr/bin/env python3
# A513_3.py
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import csv

# ──────────────────────────────────────────────────────────────
# Files & trajectory timing
# ──────────────────────────────────────────────────────────────
u = mda.Universe(
    "combine_system.dat",
    "position.lammpstrj",
    topology_format="DATA",
    format="LAMMPSDUMP",
    dt=20
)

# ──────────────────────────────────────────────────────────────
# Atom types
# ──────────────────────────────────────────────────────────────
base_types = {
    'Li (PCL)': 16,
    'Li (LLZO)': 17,
    '=O (Free PCL)': 3,
    'N  (TFSI)': 14,
}

# ──────────────────────────────────────────────────────────────
# Solid selection used for centering by COM (z only)
# ──────────────────────────────────────────────────────────────
solid_selection = "type 18 or type 19 or type 20"

# ──────────────────────────────────────────────────────────────
# Frame selection & binning
# ──────────────────────────────────────────────────────────────
frames_to_analyze = range(6000, len(u.trajectory), 1)
bin_width_A = 0.5  # Å

# Use first frame to define fixed bins over [-Lz/2, +Lz/2]
u.trajectory[0]
Lz0 = float(u.dimensions[2])
if not np.isfinite(Lz0) or Lz0 <= 0:
    raise RuntimeError("Invalid Lz from trajectory dimensions.")

zmin = -0.5 * Lz0
zmax =  0.5 * Lz0

bins = np.arange(zmin, zmax, bin_width_A)
if len(bins) == 0 or bins[0] > zmin:
    bins = np.insert(bins, 0, zmin)
if bins[-1] < zmax:
    bins = np.append(bins, zmax)

bin_centers_A = 0.5 * (bins[:-1] + bins[1:])

# Solid group
solid_atoms = u.select_atoms(solid_selection)
if solid_atoms.n_atoms == 0:
    raise RuntimeError("Solid selection returned 0 atoms. Edit 'solid_selection'.")

# Build selection objects once
base_groups = {k: u.select_atoms(f"type {v}") for k, v in base_types.items()}

# Print actual group sizes
print("\n=== Static selection sizes from topology ===")
for name, ag in base_groups.items():
    print(f"{name:16s}: {ag.n_atoms}")

# Setup histogram accumulators
summed_hists = {k: np.zeros(len(bins) - 1, dtype=float) for k in base_types.keys()}

# ──────────────────────────────────────────────────────────────
# Pass over frames
# ──────────────────────────────────────────────────────────────
warned_box_change = False

for frame in frames_to_analyze:
    u.trajectory[frame]
    u.atoms.wrap()

    Lz = float(u.dimensions[2])
    if not np.isfinite(Lz) or Lz <= 0:
        raise RuntimeError(f"Invalid Lz at frame {frame}: {Lz}")

    # For NVT this should remain constant
    if abs(Lz - Lz0) > 1e-6 and not warned_box_change:
        print(f"[WARN] Lz differs from first frame: frame {frame}, Lz={Lz:.6f}, Lz0={Lz0:.6f}")
        warned_box_change = True

    COMz = float(solid_atoms.center_of_mass()[2])

    # Center by solid COM and wrap into [-Lz0/2, +Lz0/2)
    z_all = (u.atoms.positions[:, 2] - COMz + 0.5 * Lz0) % Lz0 - 0.5 * Lz0

    for name, ag in base_groups.items():
        if ag.n_atoms > 0:
            z_species = z_all[ag.indices]
            hist, _ = np.histogram(z_species, bins=bins, density=False)

            # Conservation check
            if hist.sum() != ag.n_atoms:
                print(
                    f"[WARN] frame={frame}, {name}: "
                    f"hist sum={hist.sum()} != selected atoms={ag.n_atoms}"
                )
        else:
            hist = np.zeros(len(bins) - 1, dtype=float)

        summed_hists[name] += hist

# ──────────────────────────────────────────────────────────────
# Average across frames
# ──────────────────────────────────────────────────────────────
nframes = len(frames_to_analyze)
avg_counts = {k: summed_hists[k] / nframes for k in summed_hists.keys()}
avg_counts['Li (Total)'] = avg_counts['Li (PCL)'] + avg_counts['Li (LLZO)']

# Sanity totals
sum_li_pcl = avg_counts['Li (PCL)'].sum()
sum_li_llzo = avg_counts['Li (LLZO)'].sum()
sum_li_total = avg_counts['Li (Total)'].sum()

print("\n=== Averaged total counts from histogram ===")
print(f"Li (PCL)   : {sum_li_pcl:.6f}")
print(f"Li (LLZO)  : {sum_li_llzo:.6f}")
print(f"Li (Total) : {sum_li_total:.6f}")
print(f"Closure    : {sum_li_total - (sum_li_pcl + sum_li_llzo):.10f}")

# ──────────────────────────────────────────────────────────────
# Print compact table
# ──────────────────────────────────────────────────────────────
print("\nAveraged counts per bin (z-centered by solid COM; bins fixed over [-Lz/2, +Lz/2])")
for name in ['Li (PCL)', 'Li (LLZO)', 'Li (Total)', '=O (Free PCL)', 'N  (TFSI)']:
    print(f"\n{name}")
    print(f"{'Bin Center (Å)':>15} {'Avg Count':>12}")
    y = avg_counts[name]
    for zc, cnt in zip(bin_centers_A, y):
        print(f"{zc:15.3f} {cnt:12.6f}")

# ──────────────────────────────────────────────────────────────
# Save CSV
# ──────────────────────────────────────────────────────────────
with open("zbin_counts_centered.csv", "w", newline="") as f:
    w = csv.writer(f, delimiter=",")
    header = ["bin_center_A", "Li(PCL)", "Li(LLZO)", "Li(Total)", "O_FreePCL", "N_TFSI"]
    w.writerow(header)
    for i in range(len(bin_centers_A)):
        w.writerow([
            f"{bin_centers_A[i]:.6f}",
            f"{avg_counts['Li (PCL)'][i]:.10f}",
            f"{avg_counts['Li (LLZO)'][i]:.10f}",
            f"{avg_counts['Li (Total)'][i]:.10f}",
            f"{avg_counts['=O (Free PCL)'][i]:.10f}",
            f"{avg_counts['N  (TFSI)'][i]:.10f}",
        ])

print("\nSaved: zbin_counts_centered.csv")

# ──────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))

species_order = ['Li (PCL)', 'Li (LLZO)', 'Li (Total)', '=O (Free PCL)', 'N  (TFSI)']
n_types = len(species_order)
bin_centers_nm = bin_centers_A * 0.1
bar_group_width = bin_width_A * 0.1

for idx, name in enumerate(species_order):
    y = avg_counts[name]
    offset = (-0.5 + (idx + 0.5) / n_types) * bar_group_width
    ax.bar(
        bin_centers_nm + offset,
        y,
        width=bar_group_width / n_types,
        align='center',
        label=name
    )

ax.axvline(x=0.0, color='black', linewidth=1)
ax.set_xlabel('z (nm)')
ax.set_ylabel('Average number of atoms per bin')
ax.legend(ncol=2)
plt.tight_layout()
plt.savefig('average_counts_centered_Li_split_plus_O_TFSI.png', dpi=300)
print("Saved: average_counts_centered_Li_split_plus_O_TFSI.png")
