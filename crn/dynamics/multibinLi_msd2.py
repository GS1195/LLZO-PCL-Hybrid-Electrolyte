#!/usr/bin/env python3
"""
msd_binned_with_log.py  (2025‑07 update)
----------------------------------------
• Computes Li⁺ mean‑squared displacement (MSD) **per z‑bin** for:
  1. Li type 16 only
  2. Li type 18 only
  3. Li type 16 + 18 combined
• Writes three separate `*.dat` tables plus a single run‑info log.

Usage example
-------------
    python msd_binned_with_log.py

Settings such as file names, time‑step, and number of bins are hard‑coded
near the top of the script for quick editing.
"""
# ---------------------------------------------------------------------
import time
from pathlib import Path

import numpy as np
import MDAnalysis as mda

# ── user‑editable section ────────────────────────────────────────────
TOPOLOGY_FILE   = "combine_system.dat"
TRAJECTORY_FILE = "position.lammpstrj"
DT_PS           = 20.0        # ps per frame
NUM_BINS        = 25           # number of equal z‑slices

# output root (files are created in the same folder as the trajectory)
OUT_DIR = Path(TRAJECTORY_FILE).resolve().parent

# --------------------------------------------------------------------

def compute_msd_per_bin(pos: np.ndarray, edges: np.ndarray, max_lag: int):
    """Return (msd, count) arrays of shape (max_lag, num_bins)."""
    num_bins = len(edges) - 1

    # assign atoms to bins (wrapped z)
    bin_idx = np.digitize(pos[:, :, 2] % edges[-1], edges) - 1  # shape (T, N)
    bin_idx[bin_idx == num_bins] = num_bins - 1  # safety for z == max

    msd   = np.full((max_lag, num_bins), np.nan)
    count = np.zeros((max_lag, num_bins), dtype=int)

    for lag in range(1, max_lag + 1):
        dr2 = ((pos[lag:] - pos[:-lag]) ** 2).sum(axis=2)  # (origins, N)
        origins = bin_idx[:-lag]
        for b in range(num_bins):
            vals = dr2[origins == b]
            if vals.size:
                msd[lag - 1, b]   = vals.mean()
                count[lag - 1, b] = vals.size
    return msd, count


# --------------------------------------------------------------------

def main():
    print("Loading trajectory …")
    u = mda.Universe(TOPOLOGY_FILE, TRAJECTORY_FILE,
                     topology_format="DATA", format="LAMMPSDUMP", dt=DT_PS)
    n_total = len(u.trajectory)
    print("Total frames:", n_total)

    start_frame, end_frame = 0, n_total - 1
    max_lag = end_frame - start_frame
    print(f"Using frames 0 … {end_frame}   (max lag = {max_lag})")

    # basic geometry
    z_min_box = 0.0
    z_max_box = float(u.dimensions[2])
    edges = np.linspace(z_min_box, z_max_box, NUM_BINS + 1)

    # selections ------------------------------------------------------
    li16 = u.select_atoms("type 16")
    li18 = u.select_atoms("type 17")
    n16, n18 = len(li16), len(li18)
    n_all = n16 + n18
    print(f"Li type 16: {n16}   |   Li type 18: {n18}   |   combined: {n_all}")

    # solid slab COM (for log)
    solid_com_z = u.select_atoms("type 17 or type 19 or type 20").center_of_mass()[2]

    # ---- store positions -------------------------------------------
    pos16 = np.empty((n_total, n16, 3)) if n16 else None
    pos18 = np.empty((n_total, n18, 3)) if n18 else None

    for i, ts in enumerate(u.trajectory):
        if n16:
            pos16[i] = li16.positions
        if n18:
            pos18[i] = li18.positions

    # combined positions (concatenate along atom axis) --------------
    if n16 and n18:
        pos_all = np.concatenate((pos16, pos18), axis=1)
    elif n16:
        pos_all = pos16.copy()
    elif n18:
        pos_all = pos18.copy()
    else:
        raise RuntimeError("No Li type 16 or 18 atoms found!")

    # ---- MSD calculations ------------------------------------------
    tasks = [
        ("type16", pos16) if n16 else None,
        ("type18", pos18) if n18 else None,
        ("combined", pos_all)
    ]
    tasks = [t for t in tasks if t is not None]

    for tag, pos in tasks:
        print(f"Computing MSD for {tag} …")
        msd, count = compute_msd_per_bin(pos, edges, max_lag)

        # ----- write file -------------------------------------------
        dat_name = OUT_DIR / f"msd_{tag}_binned_vs_time.dat"
        with dat_name.open("w") as fh:
            head = ["Time_ps"] + [f"Bin{b+1}_MSD  Bin{b+1}_count" for b in range(NUM_BINS)]
            fh.write("  ".join(head) + "\n")
            for lag in range(1, max_lag + 1):
                t = lag * DT_PS
                row = [f"{t:.6f}"]
                for b in range(NUM_BINS):
                    m = msd[lag - 1, b]
                    row.append(f"{m:.6f}" if not np.isnan(m) else "NaN")
                    row.append(str(count[lag - 1, b]))
                fh.write("  ".join(row) + "\n")
        print("✔ MSD table →", dat_name.name)

    # ---- write common run‑info log ---------------------------------
    log_name = OUT_DIR / "msd_run_info.txt"
    with log_name.open("w") as fh:
        fh.write(f"run date         : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"topology         : {TOPOLOGY_FILE}\n")
        fh.write(f"trajectory       : {TRAJECTORY_FILE}\n")
        fh.write(f"frames           : 0 … {end_frame}   (dt = {DT_PS} ps)\n")
        fh.write(f"bins             : {NUM_BINS}\n")
        fh.write(f"z_min (Å)        : {z_min_box:.4f}\n")
        fh.write(f"z_max (Å)        : {z_max_box:.4f}\n")
        fh.write(f"solid COM z (Å)  : {solid_com_z:.4f}\n")
        fh.write(f"Li type 16 count : {n16}\n")
        fh.write(f"Li type 18 count : {n18}\n")
        fh.write(f"combined count   : {n_all}\n")
    print("✔ log file →", log_name.name)


if __name__ == "__main__":
    main()
