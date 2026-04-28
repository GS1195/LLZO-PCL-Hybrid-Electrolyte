#!/usr/bin/env python3
import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# USER SETTINGS (edit only these if needed)
# ----------------------------
TOP = "combine_system.dat"
TRJ = "position.lammpstrj"

# Must match your working framework
u = mda.Universe(TOP, TRJ, topology_format="DATA", format="LAMMPSDUMP", dt=20)

solid_selection = "type 18 or type 19 or type 20"  # same as your script

# Li types (your mapping)
LI_TYPES = {
    "Li_PCL": 16,
    "Li_LLZO": 17,
    # optional combined output (done automatically)
}

# Analysis range (like your heatmap: start late)
start_frame = 7000
stride = 1

# First-layer window in centered coordinates (Å)
# Choose ONE side by setting zmin/zmax:
# Right: [23, 32]
# Left:  [-32, -23]
zmin, zmax = 23.0, 32.0

# Max lag (ns) for C(t)
max_lag_ns = 2.0

# Outputs
out_prefix = "Ct_firstlayer_centered"
# ----------------------------


def compute_Ct_for_group(z_centered, in_layer, dt_ps, max_lag_ns):
    """
    z_centered: (nF, nAtoms) centered z in Å for this group
    in_layer:   (nF, nAtoms) bool mask for being in first-layer window
    Returns: dataframe with lag info + C, Ctilde, counts
    """
    nF, nA = z_centered.shape
    dt_ns = dt_ps * 1e-3
    max_lag_frames = int(np.floor(max_lag_ns / dt_ns))
    max_lag_frames = min(max_lag_frames, nF - 1)

    # mean position in-layer per atom -> z0(atom)
    z0 = np.full(nA, np.nan, dtype=float)
    for i in range(nA):
        m = in_layer[:, i]
        if np.any(m):
            z0[i] = np.mean(z_centered[m, i])

    dz = z_centered - z0[None, :]  # (nF, nA), NaN where z0 undefined

    lags = np.arange(0, max_lag_frames + 1, dtype=int)
    C = np.full_like(lags, np.nan, dtype=float)
    counts = np.zeros_like(lags, dtype=np.int64)

    for idx, lag in enumerate(lags):
        if lag == 0:
            valid = in_layer & np.isfinite(dz)
            prod = dz * dz
        else:
            valid = in_layer[:-lag, :] & in_layer[lag:, :] & np.isfinite(dz[:-lag, :]) & np.isfinite(dz[lag:, :])
            prod = dz[lag:, :] * dz[:-lag, :]

        if np.any(valid):
            C[idx] = np.mean(prod[valid])
            counts[idx] = int(np.sum(valid))
        else:
            C[idx] = np.nan
            counts[idx] = 0

    # normalize
    C0 = C[0]
    if not np.isfinite(C0) or C0 == 0:
        raise RuntimeError("C(0) is NaN/0. Likely no Li remained in the layer at any frame for this selection.")
    Ctilde = C / C0

    t_ns = lags * dt_ns
    df = pd.DataFrame({
        "lag_frames": lags,
        "t_ns": t_ns,
        "C": C,
        "Ctilde": Ctilde,
        "pair_count": counts
    })
    return df


def quick_tau_from_logfit(df, tmax_ns=1.0):
    """Simple exp fit: Ctilde ~ exp(-t/tau) using log(Ctilde) where positive."""
    m = (df["t_ns"] > 0) & (df["t_ns"] <= tmax_ns) & np.isfinite(df["Ctilde"]) & (df["Ctilde"] > 0)
    if m.sum() < 5:
        return np.nan
    x = df.loc[m, "t_ns"].to_numpy()
    y = np.log(df.loc[m, "Ctilde"].to_numpy())
    a, b = np.polyfit(x, y, 1)  # y = a x + b
    if a >= 0:
        return np.nan
    return -1.0 / a


def main():
    solid_atoms = u.select_atoms(solid_selection)
    if solid_atoms.n_atoms == 0:
        raise RuntimeError("Solid selection returned 0 atoms. Edit 'solid_selection'.")

    # Build atomgroups for Li types
    li_groups = {name: u.select_atoms(f"type {atype}") for name, atype in LI_TYPES.items()}
    for name, ag in li_groups.items():
        if ag.n_atoms == 0:
            raise RuntimeError(f"{name} selection returned 0 atoms (type {LI_TYPES[name]}).")

    frames = list(range(start_frame, len(u.trajectory), stride))
    if len(frames) < 10:
        raise RuntimeError("Too few frames selected. Decrease start_frame or stride.")

    dt_ps = float(u.trajectory.dt)  # should be 20
    dt_ns = dt_ps * 1e-3

    # allocate z arrays
    z_centered = {}
    for name, ag in li_groups.items():
        z_centered[name] = np.empty((len(frames), ag.n_atoms), dtype=float)

    # gather centered z(t): (z - COMz) folded to [-Lz/2, Lz/2)
    # This matches your approach conceptually but uses a symmetric centered coordinate,
    # which is what you want for +/- layer windows.
    u.trajectory[0]
    Lz = float(u.dimensions[2])
    if not np.isfinite(Lz) or Lz <= 0:
        raise RuntimeError("Invalid Lz from trajectory dimensions.")

    for k, fr in enumerate(frames):
        u.trajectory[fr]
        u.atoms.wrap()
        COMz = float(solid_atoms.center_of_mass()[2])

        # centered coordinate in [-Lz/2, Lz/2)
        # (z - COMz) wrapped to symmetric interval
        zc_all = (u.atoms.positions[:, 2] - COMz + 0.5 * Lz) % Lz - 0.5 * Lz

        for name, ag in li_groups.items():
            z_centered[name][k, :] = zc_all[ag.indices]

    # define in-layer mask for each group
    in_layer = {name: (z_centered[name] >= zmin) & (z_centered[name] <= zmax) for name in li_groups.keys()}

    # compute C(t) for each group + combined
    results = {}

    for name in li_groups.keys():
        results[name] = compute_Ct_for_group(z_centered[name], in_layer[name], dt_ps, max_lag_ns)

    # Combined (Li total): concatenate arrays along atom axis
    zc_tot = np.concatenate([z_centered["Li_PCL"], z_centered["Li_LLZO"]], axis=1)
    in_tot = np.concatenate([in_layer["Li_PCL"], in_layer["Li_LLZO"]], axis=1)
    results["Li_Total"] = compute_Ct_for_group(zc_tot, in_tot, dt_ps, max_lag_ns)

    # save CSVs
    for name, df in results.items():
        df.to_csv(f"{out_prefix}_{name}.csv", index=False)

    # plot
    plt.figure()
    for name, df in results.items():
        plt.plot(df["t_ns"], df["Ctilde"], marker="o", linestyle="-", label=name)
    plt.xlabel("t (ns)")
    plt.ylabel(r"$\tilde{C}(t)$")
    plt.title(f"First-layer $\~C(t)$, centered z in [{zmin},{zmax}] Å; dt={dt_ps:.1f} ps")
    plt.ylim(-1.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_ALL.png", dpi=300)

    # print quick tau estimates
    print(f"[OK] Window: z_centered in [{zmin},{zmax}] Å; frames {frames[0]}..{frames[-1]} (N={len(frames)}), dt={dt_ps} ps")
    for name, df in results.items():
        tau = quick_tau_from_logfit(df, tmax_ns=min(1.0, max_lag_ns))
        if np.isfinite(tau) and tau > 0:
            print(f"{name}: tau ≈ {tau:.4f} ns  ->  nu_eff ≈ {1.0/tau:.3f} ns^-1  (pairs@t0: {int(df.loc[0,'pair_count'])})")
        else:
            print(f"{name}: tau fit not reliable (inspect CSV/plot). (pairs@t0: {int(df.loc[0,'pair_count'])})")


if __name__ == "__main__":
    main()
