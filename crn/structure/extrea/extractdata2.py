#!/usr/bin/env python3
import numpy as np
import pandas as pd
import MDAnalysis as mda

# ---- Universe (match your working setup) ----
u = mda.Universe("combine_system.dat", "position.lammpstrj",
                 topology_format="DATA", format="LAMMPSDUMP", dt=20)

solid_selection = "type 18 or type 19 or type 20"

# Li types (your mapping)
LI_TOTAL_SEL = "type 16 or type 17"

# frame range
start_frame = 0
stride = 1

# outputs
out_li = "zc_LiTotal.csv"
out_meta = "zc_meta.csv"

solid = u.select_atoms(solid_selection)
if solid.n_atoms == 0:
    raise RuntimeError("Solid selection returned 0 atoms. Edit solid_selection.")

li = u.select_atoms(LI_TOTAL_SEL)
if li.n_atoms == 0:
    raise RuntimeError("Combined Li selection returned 0 atoms. Check types 16/17.")

frames = list(range(start_frame, len(u.trajectory), stride))
if len(frames) < 5:
    raise RuntimeError("Too few frames. Decrease start_frame or stride.")

u.trajectory[frames[0]]
Lz = float(u.dimensions[2])
if not np.isfinite(Lz) or Lz <= 0:
    raise RuntimeError("Invalid Lz.")

# allocate centered z in [-Lz/2, Lz/2)
zc = np.empty((len(frames), li.n_atoms), dtype=np.float32)

for k, fr in enumerate(frames):
    u.trajectory[fr]
    u.atoms.wrap()
    COMz = float(solid.center_of_mass()[2])
    zc_all = (u.atoms.positions[:, 2] - COMz + 0.5 * Lz) % Lz - 0.5 * Lz
    zc[k, :] = zc_all[li.indices]

# write one wide CSV: frame + one column per Li id
cols = ["frame"] + [f"Li_id{aid}" for aid in li.ids]
df = pd.DataFrame(np.column_stack([frames, zc]), columns=cols)
df.to_csv(out_li, index=False)

meta = pd.DataFrame([{
    "top": "combine_system.dat",
    "traj": "position.lammpstrj",
    "dt_ps": float(u.trajectory.dt),
    "start_frame": int(start_frame),
    "stride": int(stride),
    "n_frames": int(len(frames)),
    "Lz_A": float(Lz),
    "solid_selection": solid_selection,
    "li_selection": LI_TOTAL_SEL,
    "n_Li_total": int(li.n_atoms),
}])
meta.to_csv(out_meta, index=False)

print(f"[OK] Wrote: {out_li}  ({zc.shape[0]} frames x {zc.shape[1]} Li_total)")
print(f"[OK] Wrote: {out_meta}")
