#!/usr/bin/env python3

"""binned_msd_cli.py



Compute mean–squared displacements (MSD) of Li ions binned along the *z*‑axis

for one or more independent lag‑times, **without any interactive prompts**.



Example

-------

Run two independent MSD calculations in a single command:



```bash

python binned_msd_cli.py \

       --ti 0        --tf 4000             \

       --runs "1000:50:run1.dat,2000:75:run2.dat"

```



* `ti`, `tf`            – initial/final time (same units as the simulation)

* `runs`                – comma‑separated list of *lag:nb[:outfile]*

  * `lag`      – lag‑time for the MSD (same units as `ti`/`tf`)

  * `nb`       – number of bins along *z*

  * `outfile`  – optional output filename (default is *prefix<idx>.dat*)



You may also specify a different trajectory or data file, change the Li atom

`type`, or adjust the frame‑interval:



```bash

python binned_msd_cli.py \

       -d combine_system2.dat \

       -t nvt_3to11_uw.lammpstrj \

       --frame-interval 20 \

       --atom-type 21 \

       --prefix msd_ \

       --ti 0 --tf 4000 \

       --runs "1000:50,1500:50,2000:50"

```

"""

from __future__ import annotations



import argparse

from pathlib import Path

from typing import List, Tuple, Optional



import MDAnalysis as mda

import numpy as np



###############################################################################

# CLI helpers

###############################################################################



RunSpec = Tuple[float, int, Optional[str]]  # (lag_time, nbins, outfile)



def _parse_runs(val: str) -> List[RunSpec]:

    """Parse *lag:nb[:outfile]*[, ...] into a list of (lag, nb, outfile)."""

    specs: List[RunSpec] = []

    for item in val.split(','):

        if not item.strip():

            continue

        parts = item.strip().split(':')

        if len(parts) not in (2, 3):

            raise argparse.ArgumentTypeError(

                "Each run must be specified as lag:nb[:outfile] — got “{}”".format(item)

            )

        lag, nb = float(parts[0]), int(parts[1])

        outfile = parts[2] if len(parts) == 3 and parts[2].strip() else None

        specs.append((lag, nb, outfile))

    if not specs:

        raise argparse.ArgumentTypeError("At least one run must be specified.")

    return specs



###############################################################################

# Main algorithm

###############################################################################



def compute_binned_msd(

    u: mda.Universe,

    ti: float,

    tf: float,

    runs: List[RunSpec],

    atom_type: int = 16,

    frame_interval: float = 20.0,

    output_prefix: str = "msd_python_",

) -> None:

    """Core routine performing one or more binned MSD calculations."""

    # Convert ti/tf to frame indices

    bf = int(ti / frame_interval)

    ef = int(tf / frame_interval)

    ef = min(ef, len(u.trajectory))



    # Select Li ions once

    li_atoms = u.select_atoms(f"type {atom_type}")

    nl = len(li_atoms)

    print(f"Selected {nl} Li ions (type {atom_type}).")



    # Box length in z (assuming lower boundary at 0)

    box = u.dimensions[2]

    boxo = 0.0



    for idx, (lag_time, nb, outfile) in enumerate(runs, 1):

        print(f"\nRun {idx}: lag={lag_time}, bins={nb}")

        dt = int(lag_time / frame_interval)

        db = box / nb



        # Accumulators

        msd_x = np.zeros(nb)

        msd_y = np.zeros(nb)

        msd_z = np.zeros(nb)

        count_b = np.zeros(nb, dtype=int)



        # Traverse trajectory

        for it in range(bf, ef - dt):

            u.trajectory[it]

            pos_initial = li_atoms.positions.copy()

            initial_ids = li_atoms.ids.copy()



            u.trajectory[it + dt]

            pos_final = li_atoms.positions.copy()

            final_id_to_index = {aid: j for j, aid in enumerate(li_atoms.ids)}



            # For each Li atom present in both frames

            for i, aid in enumerate(initial_ids):

                j = final_id_to_index.get(aid)

                if j is None:

                    continue  # atom disappeared (rare)



                dx, dy, dz = pos_final[j] - pos_initial[i]



                # Bin by initial z coordinate (wrapped)

                zw = pos_initial[i, 2] - boxo

                zw -= box * int(zw / box)

                if zw < 0:

                    zw += box

                bin_index = min(int(zw / db), nb - 1)



                msd_x[bin_index] += dx * dx

                msd_y[bin_index] += dy * dy

                msd_z[bin_index] += dz * dz

                count_b[bin_index] += 1



        # Average

        with np.errstate(divide="ignore", invalid="ignore"):

            avg_msd_x = np.divide(msd_x, count_b, out=np.zeros_like(msd_x), where=count_b>0)

            avg_msd_y = np.divide(msd_y, count_b, out=np.zeros_like(msd_y), where=count_b>0)

            avg_msd_z = np.divide(msd_z, count_b, out=np.zeros_like(msd_z), where=count_b>0)



        # Output

        out_path = Path(outfile) if outfile else Path(f"{output_prefix}{idx}.dat")

        with out_path.open("w") as fh:

            for i in range(nb):

                fh.write(f"{(i + 1) * db} {avg_msd_x[i]} {avg_msd_y[i]} {avg_msd_z[i]}\n")

        print(f"  → {out_path} written.  (non‑empty bins: {np.count_nonzero(count_b)}/{nb})")



    print("\nAll runs completed.")



###############################################################################

# Entry point

###############################################################################



def main(argv: List[str] | None = None) -> None:

    p = argparse.ArgumentParser(

        description="Binned Li‑ion MSD calculation with command‑line control.",

        formatter_class=argparse.ArgumentDefaultsHelpFormatter,

    )

    p.add_argument("--ti", type=float, required=True, help="Initial time")

    p.add_argument("--tf", type=float, required=True, help="Final time")

    p.add_argument("--runs", type=_parse_runs, required=True,

                   help="Comma‑separated list lag:nb[:outfile] for each run")



    p.add_argument("-d", "--data", default="combine_system.dat",

                   help="LAMMPS data/topology file")

    p.add_argument("-t", "--traj", default="position.lammpstrj",

                   help="LAMMPS trajectory dump file")



    p.add_argument("--frame-interval", type=float, default=20,

                   help="Time difference between successive frames in the trajectory")

    p.add_argument("--atom-type", type=int, default=16,

                   help="Atom *type* identifier for Li ions in the trajectory")

    p.add_argument("--prefix", default="msd_python_",

                   help="Prefix for automatically generated output files")



    args = p.parse_args(argv)



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

        atom_type=args.atom_type,

        frame_interval=args.frame_interval,

        output_prefix=args.prefix,

    )



if __name__ == "__main__":

    main()

