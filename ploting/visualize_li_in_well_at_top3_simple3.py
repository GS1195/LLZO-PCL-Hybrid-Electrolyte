# python3 visualize_li_in_well_at_top3_simple3.py   --csv zc_LiTotal.csv   --dt_ps 20   --crossing_low 21.5   --crossing_high 23.5   --window_ns 1.0   --crosscheck_ns 0.5   --cooldown_ns 1.0   --n_examples 10   --one_per_li   --seed 2026   --zmin_valid -23.0   --z_barrier 22.4     --out_prefix PCL700K_directionalCrossings2
#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


def apply_house_style():
    plt.rcParams.update({
        "font.size": 14,
        "font.weight": "bold",
        "axes.linewidth": 2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "xtick.major.size": 10,
        "ytick.major.size": 10,
        "xtick.minor.width": 1.8,
        "ytick.minor.width": 1.8,
        "xtick.minor.size": 5,
        "ytick.minor.size": 5,
        "lines.linewidth": 2.5,
    })


def style_axes(ax, x_major=None, x_minor_div=5, y_minor=True):
    if x_major is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
        ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor_div))
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    if y_minor:
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(axis="both", which="major",
                   bottom=True, left=True, top=True, right=True,
                   direction="in", width=2, length=8, labelsize=14)
    ax.tick_params(axis="both", which="minor",
                   bottom=True, left=True, top=True, right=True,
                   direction="in", width=2, length=4, labelsize=14)

    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)


def find_directional_candidates(
    z,
    frames,
    time_ns,
    crossing_low,
    crossing_high,
    window_frames,
    check_frames,
    cooldown_frames,
    zmin_valid=None,
):
    """
    Candidate frame i is accepted only if:
    1) z(i) lies in crossing window [crossing_low, crossing_high]
    2) mean z in the previous check window is > crossing_high
    3) mean z in the next     check window is < crossing_low
    4) full plotting window exists
    5) optional: full plotting window never goes below zmin_valid
    """
    candidates = []
    N = len(z)

    start_i = max(window_frames, check_frames)
    end_i = min(N - window_frames - 1, N - check_frames - 1)

    i = start_i
    while i <= end_i:
        z0 = z[i]

        if not (crossing_low <= z0 <= crossing_high):
            i += 1
            continue

        pre_seg = z[i - check_frames:i]
        post_seg = z[i + 1:i + 1 + check_frames]

        if len(pre_seg) < check_frames or len(post_seg) < check_frames:
            i += 1
            continue

        pre_mean = np.mean(pre_seg)
        post_mean = np.mean(post_seg)

        if not (pre_mean > crossing_high and post_mean < crossing_low):
            i += 1
            continue

        i0 = i - window_frames
        i1 = i + window_frames
        zwin = z[i0:i1 + 1]

        if zmin_valid is not None and np.any(zwin < zmin_valid):
            i += 1
            continue

        candidates.append({
            "event_frame_idx": int(i),
            "event_frame": int(frames[i]),
            "event_time_ns": float(time_ns[i]),
            "z_t0": float(z0),
            "pre_mean_z": float(pre_mean),
            "post_mean_z": float(post_mean),
        })

        i += cooldown_frames

    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="zc_LiTotal.csv")
    ap.add_argument("--dt_ps", type=float, default=20.0)

    ap.add_argument("--crossing_low", type=float, default=21.5)
    ap.add_argument("--crossing_high", type=float, default=23.5)

    ap.add_argument("--window_ns", type=float, default=1.0,
                    help="Plot from -window_ns to +window_ns around t0")
    ap.add_argument("--crosscheck_ns", type=float, default=0.2,
                    help="Local directional check window before/after t0")
    ap.add_argument("--cooldown_ns", type=float, default=1.0,
                    help="Minimum separation between candidate events for same Li")

    ap.add_argument("--n_examples", type=int, default=6)
    ap.add_argument("--one_per_li", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)

    ap.add_argument("--zmin_valid", type=float, default=-23.0,
                    help="Reject candidate if plotted window dips below this value")

    ap.add_argument("--z_barrier", type=float, default=22.4)
    ap.add_argument("--z_wmin", type=float, default=24.0)
    ap.add_argument("--z_wmax", type=float, default=34.0)

    ap.add_argument("--out_prefix", default="directionalCrossings")

    args = ap.parse_args()

    apply_house_style()

    df = pd.read_csv(args.csv)
    if "frame" not in df.columns:
        raise RuntimeError("CSV must contain a 'frame' column.")

    li_cols = [c for c in df.columns if c != "frame"]
    if not li_cols:
        raise RuntimeError("No Li columns found.")

    frames = df["frame"].to_numpy(int)
    Z = df[li_cols].to_numpy(float)

    dt_ns = args.dt_ps * 1e-3
    time_ns = np.arange(len(frames)) * dt_ns

    window_frames = max(1, int(round(args.window_ns / dt_ns)))
    check_frames = max(1, int(round(args.crosscheck_ns / dt_ns)))
    cooldown_frames = max(1, int(round(args.cooldown_ns / dt_ns)))

    all_candidates = []
    for j, li_name in enumerate(li_cols):
        z = Z[:, j]
        cand_j = find_directional_candidates(
            z=z,
            frames=frames,
            time_ns=time_ns,
            crossing_low=args.crossing_low,
            crossing_high=args.crossing_high,
            window_frames=window_frames,
            check_frames=check_frames,
            cooldown_frames=cooldown_frames,
            zmin_valid=args.zmin_valid,
        )
        for ev in cand_j:
            ev["Li_col"] = li_name
            ev["Li_col_index0based"] = j
            all_candidates.append(ev)

    all_df = pd.DataFrame(all_candidates, columns=[
        "Li_col",
        "Li_col_index0based",
        "event_frame_idx",
        "event_frame",
        "event_time_ns",
        "z_t0",
        "pre_mean_z",
        "post_mean_z",
    ])

    all_csv = f"{args.out_prefix}_allCandidates.csv"
    all_df.to_csv(all_csv, index=False)

    if len(all_df) == 0:
        print("[WARN] No directional crossing candidates found.")
        print(f"[OK] Saved empty table: {all_csv}")
        return

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(all_df))

    picked_rows = []
    used_li = set()

    for k in perm:
        row = all_df.iloc[k]
        li_name = row["Li_col"]

        if args.one_per_li and li_name in used_li:
            continue

        picked_rows.append(row)
        used_li.add(li_name)

        if len(picked_rows) >= min(args.n_examples, len(all_df)):
            break

    selected_df = pd.DataFrame(picked_rows).reset_index(drop=True)
    selected_df.insert(0, "random_rank", np.arange(1, len(selected_df) + 1))

    sel_csv = f"{args.out_prefix}_selectedCandidates.csv"
    selected_df.to_csv(sel_csv, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    for _, ev in selected_df.iterrows():
        j = int(ev["Li_col_index0based"])
        event_idx = int(ev["event_frame_idx"])
        z = Z[:, j]

        i0 = event_idx - window_frames
        i1 = event_idx + window_frames

        t_rel = time_ns[i0:i1 + 1] - time_ns[event_idx]
        z_win = z[i0:i1 + 1]

        ax.plot(t_rel, z_win, label=f"{ev['Li_col']}")

    ax.axvline(0.0, color="k", linewidth=2)
#    ax.axhline(args.crossing_low, linestyle=":", color="k", linewidth=1.8)
#    ax.axhline(args.crossing_high, linestyle=":", color="k", linewidth=1.8)
    ax.axhline(args.z_barrier, linestyle="-.", color="k", linewidth=2)
#    ax.axhline(args.z_wmin, linestyle="--", color="k", linewidth=1.5)
#    ax.axhline(args.z_wmax, linestyle="--", color="k", linewidth=1.5)

    ax.set_xlabel(r"time relative to $t_0$ (ns)", fontsize=18, fontweight="bold")
    ax.set_ylabel(r"$z(t)$ (Å)", fontsize=18, fontweight="bold")
    ax.set_xlim(-args.window_ns, args.window_ns)
    style_axes(ax, x_major=0.5, x_minor_div=5)

    ax.set_title(
        rf"Random downward crossings with $z(t_0)\in[{args.crossing_low:.1f},{args.crossing_high:.1f}]$ Å",
        fontsize=16, fontweight="bold"
    )

    ax.legend(frameon=False, fontsize=10, ncol=2, loc="best")

    fig.tight_layout()
    out_png = f"{args.out_prefix}_singlePlot.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"[OK] All candidates: {all_csv}")
    print(f"[OK] Selected candidates: {sel_csv}")
    print(f"[OK] Single combined plot: {out_png}")


if __name__ == "__main__":
    main()
