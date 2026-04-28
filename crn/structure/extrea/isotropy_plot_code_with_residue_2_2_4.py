#!/usr/bin/env python3
"""
anisotropy_bins.py  (2025‑07 update)
------------------------------------
• Computes monomer ⟨x²⟩, ⟨y²⟩, ⟨z²⟩ per *z*-slice.
• Accepts **multiple residue ranges** via `--pcl_resids 'LOW‑HIGH …'`.
• You can now **let the script build equal‑width bins** with `--n_bins N`
  *or* pass explicit regions with `--regions 'z_lo,z_hi …'` (mutually exclusive).
• Output CSV contains slab location info to help downstream plotting.

Run examples
------------
# 12 equal slices, analysing three PCL ranges
python anisotropy_bins.py \
       --s combine_system.dat --f position.lammpstrj \
       --pcl_resids '1-100 302-401 603-702' --n_bins 12

# explicit user‑defined regions (old behaviour)
python anisotropy_bins.py \
       --s combine_system.dat --f position.lammpstrj \
       --pcl_resids '1-100' --regions '0,20 20,40 40,60'
"""
# ---------------------------------------------------------------------
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt

# ── helpers -----------------------------------------------------------

def parse_pcl_resids(pcl_string: str, parser: argparse.ArgumentParser):
    """Return list of (lo, hi) tuples from 'LOW-HIGH' tokens."""
    ranges = []
    for token in pcl_string.split():
        m = re.fullmatch(r"(\d+)-(\d+)", token)
        if not m:
            parser.error(
                f"Invalid range '{token}' in --pcl_resids (expected LOW-HIGH)")
        lo, hi = map(int, m.groups())
        if lo > hi:
            lo, hi = hi, lo
        ranges.append((lo, hi))
    return ranges


def parse_regions(txt: str):
    """Return list of (z_lo, z_hi) floats from 'lo,hi lo,hi …'."""
    return [tuple(map(float, p.split(','))) for p in txt.split()]


# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Monomer r² anisotropy per z-slice")
    ap.add_argument("--s", required=True, help="LAMMPS DATA file")
    ap.add_argument("--f", required=True, help="LAMMPS DUMP trajectory file")

    # residue ranges
    ap.add_argument("--pcl_resids", required=True,
                    help="space‑separated residue ranges 'LOW-HIGH …'")

    # z‑slice definition (choose ONE)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--regions", help="explicit z ranges 'lo,hi lo,hi …' (Å)")
    grp.add_argument("--n_bins", type=int,
                     help="number of equal‑width z‑bins spanning the box")

    # misc
    ap.add_argument("--dt", type=float, default=20, help="timestep (ps)")
    args = ap.parse_args()

    # ---- parse residue ranges ---------------------------------------
    res_ranges = parse_pcl_resids(args.pcl_resids, ap)

    # ---- MDAnalysis universe ----------------------------------------
    u = mda.Universe(args.s, args.f,
                     topology_format="DATA", format="LAMMPSDUMP", dt=args.dt)

    # ---- z‑regions (equal bins or explicit) -------------------------
    Lz_max = float(u.dimensions[2])
    z_min_box, z_max_box = 0.0, Lz_max

    if args.regions:
        region_ranges = parse_regions(args.regions)
    else:  # args.n_bins is set
        width = (z_max_box - z_min_box) / args.n_bins
        region_ranges = [
            (z_min_box + i * width, z_min_box + (i + 1) * width)
            for i in range(args.n_bins)
        ]

    n_regions = len(region_ranges)

    # ---- containers --------------------------------------------------
    rx_sq = {i: [] for i in range(n_regions)}
    ry_sq = {i: [] for i in range(n_regions)}
    rz_sq = {i: [] for i in range(n_regions)}
    counts = {i: 0 for i in range(n_regions)}
    n_frames = 0

    # ---- track solid min / max / COM --------------------------------
    solid = u.select_atoms("type 18 or type 19 or type 20")
    solid_com_z_sum = 0.0
    solid_zmin, solid_zmax = np.inf, -np.inf

    # ---- main loop --------------------------------------------------
    for ts in u.trajectory:
        n_frames += 1

        # slab statistics
        z_sol = solid.positions[:, 2]
        solid_com_z_sum += z_sol.mean()
        solid_zmin = min(solid_zmin, z_sol.min())
        solid_zmax = max(solid_zmax, z_sol.max())

        # iterate every bond vector in every *selected* residue
        for res in u.residues:
            if not any(lo <= res.resid <= hi for lo, hi in res_ranges):
                continue

            atoms = res.atoms
            start = atoms.select_atoms("type 1")
            end = atoms.select_atoms("type 7")
            bridge = atoms.select_atoms("type 6")
            if start.n_atoms != 1 or end.n_atoms != 1 or bridge.n_atoms < 1:
                continue
            endpoints = [start.atoms[0]] + \
                        sorted(bridge, key=lambda a: a.id) + \
                        [end.atoms[0]]

            for a, b in zip(endpoints[:-1], endpoints[1:]):
                vec = b.position - a.position
                center_z = 0.5 * (a.position[2] + b.position[2]) % Lz_max

                for idx, (z_lo, z_hi) in enumerate(region_ranges):
                    if z_lo <= center_z <= z_hi:
                        rx_sq[idx].append(vec[0] ** 2)
                        ry_sq[idx].append(vec[1] ** 2)
                        rz_sq[idx].append(vec[2] ** 2)
                        counts[idx] += 1
                        break

    solid_COM_z = solid_com_z_sum / n_frames if n_frames else np.nan

    # ---- assemble tidy rows -----------------------------------------
    def mean_unc(vals):
        vals = np.asarray(vals)
        if vals.size == 0:
            return 0.0, 0.0
        mean = vals.mean()
        unc = vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
        return mean, unc

    rows = []
    for idx, (z_lo, z_hi) in enumerate(region_ranges):
        rx_m, rx_u = mean_unc(rx_sq[idx])
        ry_m, ry_u = mean_unc(ry_sq[idx])
        rz_m, rz_u = mean_unc(rz_sq[idx])

        rows.append(dict(slice_idx=idx,
                         z_start=z_lo,
                         z_end=z_hi,
                         z_center=0.5 * (z_lo + z_hi),
                         rx_mean=rx_m, rx_unc=rx_u,
                         ry_mean=ry_m, ry_unc=ry_u,
                         rz_mean=rz_m, rz_unc=rz_u,
                         monomers=counts[idx] / n_frames if n_frames else 0,
                         frames=n_frames,
                         z_min=z_min_box,
                         z_max=z_max_box,
                         solid_COM_z=solid_COM_z,
                         solid_zmin=solid_zmin,
                         solid_zmax=solid_zmax))

    # ---- write CSV ---------------------------------------------------
    out_csv = Path("monomer_anisotropy_profile.csv")
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("✔", out_csv)

    # ---- (optional) quick‑look figures ------------------------------
    region_centers = [r["z_center"] for r in rows]
    x = np.array(region_centers)
    rx = [r["rx_mean"] for r in rows]
    ry = [r["ry_mean"] for r in rows]
    rz = [r["rz_mean"] for r in rows]
    rx_u = [r["rx_unc"] for r in rows]
    ry_u = [r["ry_unc"] for r in rows]
    rz_u = [r["rz_unc"] for r in rows]
    avg = [r["monomers"] for r in rows]

    def _plot(with_unc, fname):
        fig, ax = plt.subplots(figsize=(8, 5))
        if with_unc:
            ax.errorbar(x, rx, yerr=rx_u, fmt='-o', capsize=4, label=r"$\langle x^2\rangle$")
            ax.errorbar(x, ry, yerr=ry_u, fmt='-s', capsize=4, label=r"$\langle y^2\rangle$")
            ax.errorbar(x, rz, yerr=rz_u, fmt='-^', capsize=4, label=r"$\langle z^2\rangle$")
            title = "Monomer $\\langle r^2\\rangle$ with uncertainty"
        else:
            ax.plot(x, rx, '-o', label=r"$\langle x^2\rangle$")
            ax.plot(x, ry, '-s', label=r"$\langle y^2\rangle$")
            ax.plot(x, rz, '-^', label=r"$\langle z^2\rangle$")
            title = "Monomer $\\langle r^2\\rangle$ (no uncertainty)"

        ax.set_xlabel("Z (Å)")
        ax.set_ylabel("Mean of $r^2$ (Å²)")
        ax.set_ylim(0, 40)
        ax.set_xlim(0, Lz_max)
        ax.set_xticks(x)
        ax.set_title(title)
        ax.legend()
        for i, xp in enumerate(x):
            ax.annotate(f"{avg[i]:.2f}", (xp, max(rx[i], ry[i], rz[i])), xytext=(0, 10),
                        textcoords='offset points', ha='center')
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()

    _plot(True, "monomer_r2_uncert.png")
    _plot(False, "monomer_r2_noerr.png")


if __name__ == "__main__":
    main()
