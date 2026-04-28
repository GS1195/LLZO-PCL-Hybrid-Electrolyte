#!/usr/bin/env python3
"""
binned_msd_cli.py

Compute mean–squared displacements (MSD) of selected ions binned along the z-axis
for one or more independent lag-times, without interactive prompts.

Now supports MULTIPLE atom types via --atom-types "16,18" (or "16 18").
You can also provide an explicit MDAnalysis selection with --atom-sel.

Examples
--------
# single type (backward compatible)
python binned_msd_cli.py --ti 0 --tf 4000 --runs "1000:50" --atom-types 16

# combine Li from PCL(type 16) and LLZO(type 18)
python binned_msd_cli.py --ti 100000 --tf 340000 \
    --runs "1000:25:Li_total.dat" --atom-types 16,18

# explicit selection string (overrides --atom-types if both are given)
python binned_msd_cli.py --ti 0 --tf 4000 --runs "1000:50" \
    --atom-sel "type 16 or type 18"
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import MDAnalysis as mda
import numpy as np

# ------------------------ CLI helpers ------------------------

RunSpec = Tuple[float, int, Optional[str]]  # (lag_time, nbins, outfile)

def _parse_runs(val: str) -> List[RunSpec]:
    """Parse 'lag:nb[:outfile][, ...]' into a list of (lag, nb, outfile)."""
    specs: List[RunSpec] = []
    for item in val.split(','):
        if not item.strip():
            continue
        parts = item.strip().split(':')
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError(
                f"Each run must be 'lag:nb[:outfile]' — got “{item}”"
            )
        lag, nb = float(parts[0]), int(parts[1])
        outfile = parts[2].strip() if len(parts) == 3 and parts[2].strip() else None
        specs.append((lag, nb, outfile))
    if not specs:
        raise argparse.ArgumentTypeError("At least one run must be specified.")
    return specs

def _parse_atom_types(val: Optional[str]) -> List[int]:
    """Parse a list of atom types from comma/space separated string."""
    if val is None:
        return []
    # split by comma and/or whitespace
    tokens = [tok for tok in val.replace(',', ' ').split() if tok.strip()]
    try:
        return [int(t) for t in tokens]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Could not parse --atom-types from “{val}”"
        ) from e

# ----------------------- Core algorithm -----------------------

def compute_binned_msd(
    u: mda.Universe,
    ti: float,
    tf: float,
    runs: List[RunSpec],
    atom_selection: str,
    frame_interval: float = 20.0,
    output_prefix: str = "msd_python_",
) -> None:
    """Perform one or more binned MSD calculations for a given selection."""
    # Convert ti/tf to frame indices
    bf = int(ti / frame_interval)
    ef = int(tf / frame_interval)
    ef = min(ef, len(u.trajectory))

    # Select atoms once (dynamic AtomGroup, positions update with frame)
    atoms = u.select_atoms(atom_selection)
    na = len(atoms)
    print(f"Selected {na} atoms with selection: {atom_selection}")
    if na == 0:
        raise ValueError("Selection returned zero atoms — check --atom-types/--atom-sel.")

    # Box length in z (assuming lower boundary at 0)
    box = float(u.dimensions[2])
    boxo = 0.0

    for idx, (lag_time, nb, outfile) in enumerate(runs, 1):
        print(f"\nRun {idx}: lag={lag_time}, bins={nb}")
        dt = int(lag_time / frame_interval)
        if dt <= 0:
            raise ValueError(f"Lag time {lag_time} must be >= frame interval {frame_interval}.")
        if ef - bf <= dt:
            raise ValueError("Time window too short for given lag; decrease lag or extend tf.")
        db = box / nb

        # Accumulators
        msd_x = np.zeros(nb)
        msd_y = np.zeros(nb)
        msd_z = np.zeros(nb)
        count_b = np.zeros(nb, dtype=int)

        # Traverse trajectory
        for it in range(bf, ef - dt):
            u.trajectory[it]
            pos_initial = atoms.positions.copy()
            initial_ids = atoms.ids.copy()

            u.trajectory[it + dt]
            # Note: 'atoms' is dynamic; its ids are the same set each frame
            pos_final = atoms.positions.copy()
            final_ids = atoms.ids
            final_id_to_index = {aid: j for j, aid in enumerate(final_ids)}

            # For each atom present in both frames
            for i, aid in enumerate(initial_ids):
                j = final_id_to_index.get(aid)
                if j is None:
                    continue  # atom disappeared; skip

                dx, dy, dz = pos_final[j] - pos_initial[i]

                # Bin by initial z coordinate (wrapped into [0, box))
                zw = pos_initial[i, 2] - boxo
                zw -= box * int(zw / box)
                if zw < 0:
                    zw += box
                bin_index = min(int(zw / db), nb - 1)

                msd_x[bin_index] += dx * dx
                msd_y[bin_index] += dy * dy
                msd_z[bin_index] += dz * dz
                count_b[bin_index] += 1

        # Averages
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_msd_x = np.divide(msd_x, count_b, out=np.zeros_like(msd_x), where=count_b > 0)
            avg_msd_y = np.divide(msd_y, count_b, out=np.zeros_like(msd_y), where=count_b > 0)
            avg_msd_z = np.divide(msd_z, count_b, out=np.zeros_like(msd_z), where=count_b > 0)

        # Output
        out_path = Path(outfile) if outfile else Path(f"{output_prefix}{idx}.dat")
        with out_path.open("w") as fh:
            for i in range(nb):
                # column 1 = bin edge (z), then MSDx, MSDy, MSDz
                fh.write(f"{(i + 1) * db} {avg_msd_x[i]} {avg_msd_y[i]} {avg_msd_z[i]}\n")
        print(f"  → {out_path} written.  (non-empty bins: {np.count_nonzero(count_b)}/{nb})")

    print("\nAll runs completed.")

# ------------------------- Entry point -------------------------

def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Binned MSD calculation for one or multiple atom types.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ti", type=float, required=True, help="Initial time")
    p.add_argument("--tf", type=float, required=True, help="Final time")
    p.add_argument("--runs", type=_parse_runs, required=True,
                   help="Comma-separated list lag:nb[:outfile] for each run")

    p.add_argument("-d", "--data", default="combine_system.dat",
                   help="LAMMPS data/topology file")
    p.add_argument("-t", "--traj", default="position.lammpstrj",
                   help="LAMMPS trajectory dump file")

    p.add_argument("--frame-interval", type=float, default=20,
                   help="Time difference between successive frames in the trajectory")

    # New: multiple atom types OR explicit selection
    p.add_argument("--atom-types", type=str, default=None,
                   help="Comma/space-separated atom types to include (e.g., '16,18' or '16 18').")
    p.add_argument("--atom-sel", type=str, default=None,
                   help="MDAnalysis selection string (e.g., 'type 16 or type 18'). Overrides --atom-types if set.")

    # Backward compat (deprecated): single --atom-type
    p.add_argument("--atom-type", type=int, default=None,
                   help="(Deprecated) Single atom type. Use --atom-types instead.")
    p.add_argument("--prefix", default="msd_python_",
                   help="Prefix for automatically generated output files")

    args = p.parse_args(argv)

    # Build selection
    selection: Optional[str] = None
    if args.atom_sel:
        selection = args.atom_sel.strip()
    else:
        types = []
        # include deprecated single type if provided
        if args.atom_type is not None:
            types.append(int(args.atom_type))
        # include multi types if provided
        types += _parse_atom_types(args.atom_types)
        # de-duplicate while preserving order
        seen = set()
        types = [t for t in types if not (t in seen or seen.add(t))]
        if not types:
            raise SystemExit("Please provide --atom-types (e.g., '16,18') or --atom-sel.")
        selection = " or ".join(f"type {t}" for t in types)

    universe = mda.Universe(
        args.data,
        args.traj,
        topology_format="DATA",
        format="LAMMPSDUMP",
        dt=args.frame_interval,
    )

    compute_binned_msd(
        universe,
        ti=args.ti,
        tf=args.tf,
        runs=args.runs,
        atom_selection=selection,
        frame_interval=args.frame_interval,
        output_prefix=args.prefix,
    )

if __name__ == "__main__":
    main()
