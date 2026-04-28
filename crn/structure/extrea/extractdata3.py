#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import MDAnalysis as mda

# ============================================================
# SETTINGS
# ============================================================
top = "combine_system.dat"
traj = "position.lammpstrj"
dt_ps = 20

solid_selection = "type 18 or type 19 or type 20"
LI_TOTAL_SEL = "type 16 or type 17"

start_frame = 0
stride = 1

# Density histogram resolution (Å)
bin_width_A = 0.5

# Output folder
outdir = "layer_check_outputs"
# ============================================================

os.makedirs(outdir, exist_ok=True)

# ---- Universe ----
u = mda.Universe(top, traj, topology_format="DATA", format="LAMMPSDUMP", dt=dt_ps)

solid = u.select_atoms(solid_selection)
if solid.n_atoms == 0:
    raise RuntimeError("Solid selection returned 0 atoms. Edit solid_selection.")

li = u.select_atoms(LI_TOTAL_SEL)
if li.n_atoms == 0:
    raise RuntimeError("Li(total) selection returned 0 atoms. Check types 16/17.")

frames = list(range(start_frame, len(u.trajectory), stride))
if len(frames) < 5:
    raise RuntimeError("Too few frames. Decrease start_frame or stride.")

# Box length (Å) from first frame
u.trajectory[frames[0]]
Lz = float(u.dimensions[2])
if not np.isfinite(Lz) or Lz <= 0:
    raise RuntimeError("Invalid Lz from trajectory dimensions.")

# ============================================================
# OUTPUT FILENAMES
# ============================================================
out_li = os.path.join(outdir, "zc_LiTotal.csv")
out_meta = os.path.join(outdir, "zc_meta.csv")
out_density_csv = os.path.join(outdir, "LiTotal_density_zcentered.csv")
out_density_png = os.path.join(outdir, "LiTotal_density_zcentered.png")

# ============================================================
# 1) Extract centered z(t) for ALL Li(total) -> wide CSV
# ============================================================
zc = np.empty((len(frames), li.n_atoms), dtype=np.float32)

# 2) Accumulate density histogram of Li(total) along centered z
z_edges = np.arange(-0.5 * Lz, 0.5 * Lz + bin_width_A, bin_width_A)
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
hist_sum = np.zeros(len(z_edges) - 1, dtype=np.float64)

for k, fr in enumerate(frames):
    u.trajectory[fr]
    u.atoms.wrap()

    COMz = float(solid.center_of_mass()[2])

    # centered coordinate in [-Lz/2, Lz/2)
    zc_all = (u.atoms.positions[:, 2] - COMz + 0.5 * Lz) % Lz - 0.5 * Lz

    zc_li = zc_all[li.indices]
    zc[k, :] = zc_li

    h, _ = np.histogram(zc_li, bins=z_edges)
    hist_sum += h

# Write wide CSV: frame + one column per Li id
cols = ["frame"] + [f"Li_id{aid}" for aid in li.ids]
df = pd.DataFrame(np.column_stack([frames, zc]), columns=cols)
df.to_csv(out_li, index=False)

# ============================================================
# Density CSV (average count per bin per frame)
# ============================================================
nframes = len(frames)
avg_counts = hist_sum / nframes

dens_df = pd.DataFrame({
    "z_center_A": z_centers,
    "avg_Li_count_per_bin_per_frame": avg_counts
})
dens_df.to_csv(out_density_csv, index=False)

# Make a quick plot (matplotlib only)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4.5))
plt.plot(z_centers, avg_counts, linewidth=1.5)
plt.xlabel("z_centered (Å)")
plt.ylabel("avg Li count per bin per frame")
plt.title("Li(total) density along centered z (solid COM-centered)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_density_png, dpi=300)
plt.close()

# ============================================================
# Meta CSV
# ============================================================
meta = pd.DataFrame([{
    "top": top,
    "traj": traj,
    "dt_ps": float(u.trajectory.dt),
    "start_frame": int(start_frame),
    "stride": int(stride),
    "n_frames": int(len(frames)),
    "Lz_A": float(Lz),
    "solid_selection": solid_selection,
    "li_selection": LI_TOTAL_SEL,
    "n_Li_total": int(li.n_atoms),
    "density_bin_width_A": float(bin_width_A),
    "density_edges_min_A": float(z_edges[0]),
    "density_edges_max_A": float(z_edges[-1]),
}])
meta.to_csv(out_meta, index=False)

print(f"[OK] Wrote: {out_li}  ({zc.shape[0]} frames x {zc.shape[1]} Li_total)")
print(f"[OK] Wrote: {out_density_csv}")
print(f"[OK] Wrote: {out_density_png}")
print(f"[OK] Wrote: {out_meta}")
print(f"[OK] Output folder: {outdir}")
