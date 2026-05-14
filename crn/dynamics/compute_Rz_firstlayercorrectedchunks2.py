#!/usr/bin/env python3
"""
compute_Rz_firstlayer_chunks.py  (connected correlation + markers + avg Li per t0 in legend)
python3 compute_Rz_firstlayercorrectedchunks.py   --zc_csv zc_LiTotal.csv   --dt_ps 20   --zmin 22.5 --zmax 32.0   --tmax_ns 1.0   --stride_t0 1   --chunks 4   --z0_ref 30.345   --out_prefix Rz_firstlayer_4chunks3
Adds to legend:
- avgLi = average number of Li in [zmin,zmax] per usable time origin (t0)
- unique = number of distinct Li ions contributing in that chunk
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# ---------- house-style ----------
def apply_house_style():
    plt.rcParams.update({
        "font.size":         25,
        "font.weight":       "bold",
        "axes.linewidth":    3,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.major.width": 8,
        "ytick.major.width": 8,
        "xtick.major.size":  10,
        "ytick.major.size":  10,
        "xtick.minor.width": 4,
        "ytick.minor.width": 4,
        "xtick.minor.size":  4,
        "ytick.minor.size":  4,
        "lines.linewidth":   4,
        "legend.frameon":    False,
    })

def style_axes(ax, x_major=None, x_minor_div=5):
    if x_major is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
        ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor_div))
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(axis='both', which='major',
                   bottom=True, left=True, top=True, right=True,
                   direction='in', width=2, length=8, labelsize=22)
    ax.tick_params(axis='both', which='minor',
                   bottom=True, left=True, top=True, right=True,
                   direction='in', width=2, length=4, labelsize=22)

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)

def choose_x_major(tmax_ns: float) -> float:
    if tmax_ns >= 20:
        return 20.0
    if tmax_ns >= 5:
        return 1.0
    if tmax_ns >= 2:
        return 0.5
    if tmax_ns >= 1:
        return 0.2
    return 0.1

def compute_Rz_chunk_connected(Z, t0_list, zmin, zmax, max_lag, z0_ref=None, eps=1e-14):
    """
    Returns:
      R, counts_t0, N(=W), n_unique, avg_inwell, debug
    """
    if len(t0_list) == 0:
        return None, 0, 0, 0, 0.0, None

    A = np.zeros(max_lag + 1, dtype=float)
    B = np.zeros(max_lag + 1, dtype=float)
    D = 0.0
    E = 0.0
    W = 0

    counts_t0 = 0
    used_li = set()
    inwell_sizes = []

    for t0 in t0_list:
        z_t0 = Z[t0, :]
        idx = np.where((z_t0 >= zmin) & (z_t0 <= zmax))[0]
        if idx.size == 0:
            continue

        z0 = float(z0_ref) if (z0_ref is not None) else float(np.mean(z_t0[idx]))

        dz0 = z_t0[idx] - z0
        dz_t = Z[t0:t0 + max_lag + 1, idx] - z0

        A += np.sum(dz_t * dz0[None, :], axis=1)
        B += np.sum(dz_t, axis=1)
        D += np.sum(dz0)
        E += np.sum(dz0 * dz0)
        W += int(idx.size)

        counts_t0 += 1
        inwell_sizes.append(int(idx.size))
        used_li.update(idx.tolist())

    if counts_t0 < 5 or W <= 0:
        return None, counts_t0, W, len(used_li), 0.0, None

    mu0 = D / W
    Ed = E / W
    C0 = Ed - mu0 * mu0
    if C0 < eps:
        return None, counts_t0, W, len(used_li), float(np.mean(inwell_sizes)), None

    C = (A / W) - (B / W) * mu0
    R = C / C0

    avg_inwell = float(np.mean(inwell_sizes)) if inwell_sizes else 0.0
    debug = {"W": W, "mu0": mu0, "C0": C0}
    return R, counts_t0, W, len(used_li), avg_inwell, debug

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zc_csv", default="zc_LiTotal.csv")
    ap.add_argument("--dt_ps", type=float, default=20.0)
    ap.add_argument("--zmin", type=float, required=True)
    ap.add_argument("--zmax", type=float, required=True)
    ap.add_argument("--z0_ref", type=float, default=None)
    ap.add_argument("--tmax_ns", type=float, default=1.0)
    ap.add_argument("--stride_t0", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--out_prefix", default="Rz_firstlayer_4chunks")

    # plotting controls
    ap.add_argument("--markevery", type=int, default=5)
    ap.add_argument("--markersize", type=float, default=7.0)
    args = ap.parse_args()

    if args.zmax <= args.zmin:
        raise RuntimeError("zmax must be > zmin.")
    if args.chunks < 1:
        raise RuntimeError("chunks must be >= 1.")
    if args.stride_t0 < 1:
        raise RuntimeError("stride_t0 must be >= 1.")
    if args.markevery < 1:
        raise RuntimeError("markevery must be >= 1.")

    apply_house_style()

    df = pd.read_csv(args.zc_csv)
    if "frame" not in df.columns:
        raise RuntimeError("Input CSV must contain a 'frame' column.")

    li_cols = [c for c in df.columns if c != "frame"]
    if len(li_cols) < 1:
        raise RuntimeError("No Li columns found.")

    Z = df[li_cols].to_numpy(float)
    n_frames = Z.shape[0]

    dt_ns = args.dt_ps * 1e-3
    max_lag = int(np.floor(args.tmax_ns / dt_ns))
    if max_lag < 2:
        raise RuntimeError("tmax_ns too small relative to dt_ps.")
    if n_frames <= max_lag + 2:
        raise RuntimeError("Trajectory too short for requested tmax_ns.")

    t = np.arange(max_lag + 1) * dt_ns
    edges = np.linspace(0, n_frames, args.chunks + 1, dtype=int)

    curves = []
    summary = []

    for ci in range(args.chunks):
        c0 = int(edges[ci])
        c1 = int(edges[ci + 1])

        if c1 - c0 <= max_lag + 1:
            print(f"[WARN] chunk {ci+1}: too short; skipping (frames {c0}:{c1})")
            continue

        t0_list = np.arange(c0, c1 - max_lag, args.stride_t0, dtype=int)

        R, counts_t0, W, n_unique, avg_inwell, dbg = compute_Rz_chunk_connected(
            Z, t0_list, args.zmin, args.zmax, max_lag, z0_ref=args.z0_ref
        )
        if R is None:
            print(f"[WARN] chunk {ci+1}: no valid Rz (check window or z0_ref).")
            continue

        out_chunk_csv = f"{args.out_prefix}_chunk{ci+1}.csv"
        pd.DataFrame({"t_ns": t, "Rz": R}).to_csv(out_chunk_csv, index=False)

        curves.append((ci + 1, R, W, n_unique, avg_inwell))

        summary.append({
            "chunk": ci + 1,
            "frame_start": c0,
            "frame_end": c1,
            "t0_used": int(counts_t0),
            "ion_samples_N": int(W),
            "unique_Li_used": int(n_unique),
            "avg_Li_in_well_per_t0": float(avg_inwell),
            "mu_dz0": float(dbg["mu0"]),
            "C0": float(dbg["C0"]),
            "chunk_csv": out_chunk_csv
        })

        print(f"[OK] chunk {ci+1}: N={W} unique={n_unique} avgInWell={avg_inwell:.2f} mu0={dbg['mu0']:.3g} C0={dbg['C0']:.3g}")

    if len(curves) == 0:
        raise RuntimeError("No chunk produced valid Rz.")

    out_summary = f"{args.out_prefix}_summary.csv"
    pd.DataFrame(summary).to_csv(out_summary, index=False)

    # ---------- combined plot with markers ----------
    marker_cycle = ["o", "s", "^", "v", "D", "P", "*", "X", "<", ">"]
    fig, ax = plt.subplots(figsize=(12, 8))

    for (ch, R, W, n_unique, avg_inwell) in curves:
        mk = marker_cycle[(ch - 1) % len(marker_cycle)]
        ax.plot(
            t, R,
            marker=mk,
            markevery=args.markevery,
            markersize=10,
            label=f"interval {ch}"
        )

    ax.set_xlabel("t (ns)", fontsize=35, fontweight="bold")
    ax.set_ylabel(r"$R_z(t)$", fontsize=35, fontweight="bold")
    style_axes(ax, x_major=choose_x_major(args.tmax_ns), x_minor_div=5)
    ax.set_xlim(0, args.tmax_ns)
    ax.legend(frameon=False, fontsize=25, loc="upper right")
    fig.tight_layout()

    out_png = f"{args.out_prefix}.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"[OK] Saved: {out_summary}")
    print(f"[OK] Saved: {out_png}")

if __name__ == "__main__":
    main()
