#!/usr/bin/env python3
"""
python3  chargedistributionfinal.py --topology combine_system.dat  --trajectory position.lammpstrj --dt 20 --nbins 1000 --selection "all" --solid-selection "type 17 or type 19 or type 20"  --prefix test2
charge_centered_avg.py

Center the solid each frame, then compute and average the charge profile.

Steps per frame:
1) Find solid COM (z).
2) Shift all atoms so solid COM → z = 0 (wrap into [-Lz/2, +Lz/2)).
3) Histogram charges along z.
Average histograms over frames and compute cumulative charge.

Outputs:
  <prefix>_charge_profile.csv      (z_center_A, charge_per_bin_e, rho_e_per_A3)
  <prefix>_cumulative_charge.csv   (z_center_A, Q_e, sigma_e_per_A2, sigma_C_per_m2)
  <prefix>_charge_profile.png
  <prefix>_cumulative_charge.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda

def wrap_center(z, shift, Lz):
    """Shift by 'shift' and wrap into [-Lz/2, +Lz/2)."""
    return (z - shift + 0.5 * Lz) % Lz - 0.5 * Lz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology",   required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--dt", type=float, default=20.0)
    ap.add_argument("--nbins", type=int, default=1000)
    ap.add_argument("--selection", default="all",
                    help='Atoms to include (default "all").')
    ap.add_argument("--solid-selection", default="type 11 or type 12",
                    help='Atoms that define the solid slab for centering.')
    ap.add_argument("--prefix", default="charge")
    ap.add_argument("--zero-end", action="store_true",
                    help="Remove a uniform background so the final cumulative is exactly 0.")
    args = ap.parse_args()

    # Load trajectory
    u = mda.Universe(args.topology, args.trajectory,
                     topology_format="DATA", format="LAMMPSDUMP", dt=args.dt)

    atoms = u.select_atoms(args.selection)
    if atoms.n_atoms == 0:
        raise ValueError(f'No atoms match selection: "{args.selection}"')

    solid = u.select_atoms(args.solid_selection)
    if solid.n_atoms == 0:
        raise ValueError(f'No atoms match solid selection: "{args.solid_selection}" '
                         '(did you mean: "type 11 or type 12"?)')

    # Diagnostics: total charge as read (your cumulative end should match this)
    total_Q = float(u.atoms.charges.sum())
    print(f"[Diag] Total charge in DATA (selection=all): {total_Q:.6f} e")

    # Reference axis from first frame
    u.trajectory[0]
    Lx0, Ly0, Lz0 = u.dimensions[:3]
    area0 = Lx0 * Ly0

    # Fixed centered bins on [-Lz0/2, +Lz0/2)
    z_edges   = np.linspace(-0.5 * Lz0, 0.5 * Lz0, args.nbins + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    dz        = z_edges[1] - z_edges[0]

    sum_hist = np.zeros(args.nbins, dtype=float)
    nframes  = 0
    dropped_abs_max = 0.0
    solid_center_after = []

    for ts in u.trajectory:
        Lx, Ly, Lz = u.dimensions[:3]  # NVT → Lz == Lz0, but we read it anyway

        # Center by solid COM each frame
        comz = solid.center_of_mass()[2]

        z = atoms.positions[:, 2].astype(float)
        q = atoms.charges.astype(float)
        m = np.isfinite(q)
        z, q = z[m], q[m]

        # Shift to solid-centered coordinates in current box, then (trivially) in ref box
        zc = wrap_center(z, comz, Lz)  # [-Lz/2, Lz/2)
        # Histogram on reference bins (Lz0). For NVT Lz==Lz0.
        hist, _ = np.histogram(zc, bins=z_edges, weights=q)

        sum_hist += hist
        nframes  += 1

        # Charge conservation diagnostic
        dropped = q.sum() - hist.sum()
        dropped_abs_max = max(dropped_abs_max, abs(dropped))

        # Monitor that the solid is at the center after transform
        zs = wrap_center(solid.positions[:, 2], comz, Lz)
        solid_center_after.append(float(np.mean(zs)))

    if nframes == 0:
        raise RuntimeError("No frames processed.")

    # Average per-slab charge (e per bin)
    avg_hist = sum_hist / nframes

    # Optional: enforce exact neutrality of the averaged profile
    if args.zero_end:
        offset = avg_hist.sum()                    # equals Q_e[-1] without neutralization
        avg_hist -= offset / args.nbins

    # Volumetric charge density (e/Å^3)
    rho_e_A3 = avg_hist / (area0 * dz)

    # Cumulative charge (from left edge)
    Q_e = np.cumsum(avg_hist)                      # e
    sigma_e_A2 = Q_e / area0                       # e/Å^2
    sigma_C_m2 = sigma_e_A2 * 16.02176634          # C/m^2

    # Save CSVs
    pd.DataFrame({
        "z_center_A": z_centers,
        "charge_per_bin_e": avg_hist,
        "rho_e_per_A3": rho_e_A3
    }).to_csv(f"{args.prefix}_charge_profile.csv", index=False)

    pd.DataFrame({
        "z_center_A": z_centers,
        "Q_e": Q_e,
        "sigma_e_per_A2": sigma_e_A2,
        "sigma_C_per_m2": sigma_C_m2
    }).to_csv(f"{args.prefix}_cumulative_charge.csv", index=False)

    # Plots
    plt.rcParams.update({"font.size": 13, "figure.figsize": (8,5), "savefig.dpi": 300})

    fig1, ax1 = plt.subplots()
    ax1.plot(z_centers, avg_hist)
    ax1.axhline(0.0, ls="--", lw=1)
    ax1.set_xlabel("z (Å, centered)"); ax1.set_ylabel("Avg net charge per slab (e)")
    fig1.tight_layout(); fig1.savefig(f"{args.prefix}_charge_profile.png")

    fig2, ax2 = plt.subplots()
    ax2.plot(z_centers, Q_e); ax2.axhline(0.0, ls="--", lw=1)
    ax2.set_xlabel("z (Å, centered)"); ax2.set_ylabel("Cumulative charge Q(z) [e]")
    ax2b = ax2.twinx(); ax2b.plot(z_centers, sigma_C_m2, ls="--")
    ax2b.set_ylabel("σ(z) [C/m²]")
    fig2.tight_layout(); fig2.savefig(f"{args.prefix}_cumulative_charge.png")

    # Console diagnostics
    print(f"[Diag] Frames averaged: {nframes}")
    print(f"[Diag] Final Q(Lz) = {Q_e[-1]:.6f} e "
          f"(≈ total charge; add --zero-end to force exactly 0)")
    print(f"[Diag] Max dropped charge per frame: {dropped_abs_max:.3e} e")
    print(f"[Diag] Solid COM after centering (mean over frames): {np.mean(solid_center_after):.3f} Å (should be ~0)")

if __name__ == "__main__":
    main()
