# python3 visualize_li_in_well3.py   --csv zc_LiTotal.csv --dt_ps 20   --chunks 4 --n_li 5   --use_last_ns 200   --z_wmin 24 --z_wmax 34 --z_prod 22   --tcommit_ns 0.5   --out_prefix PCL700K_reselect_last200
#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# -------------------------
# House style
# -------------------------
def apply_house_style():
    plt.rcParams.update({
        "font.size":        14,
        "font.weight":      "bold",
        "axes.linewidth":   2,
        "xtick.direction":  "in",
        "ytick.direction":  "in",
        "xtick.major.width":2,
        "ytick.major.width":2,
        "xtick.major.size": 10,
        "ytick.major.size": 10,
        "xtick.minor.width":1.8,
        "ytick.minor.width":1.8,
        "xtick.minor.size": 5,
        "ytick.minor.size": 5,
        "lines.linewidth":  2.5,
    })

def style_axes(ax, x_major=None, x_minor_div=5, y_minor=True):
    if x_major is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
        ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor_div))
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    if y_minor:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='major',
                   bottom=True, left=True, top=True, right=True,
                   direction='in', width=2, length=8, labelsize=16)
    ax.tick_params(axis='both', which='minor',
                   bottom=True, left=True, top=True, right=True,
                   direction='in', width=2, length=4, labelsize=16)
    for spine in ['top','right','bottom','left']:
        ax.spines[spine].set_visible(True)

# -------------------------
# States
# -------------------------
# 0 = SOLID, 1 = WELL, 2 = POLYMER
def classify_state(z, z_prod, z_wmin, z_wmax):
    st = np.full_like(z, 2, dtype=np.int8)          # POLYMER
    st[(z >= z_wmin) & (z <= z_wmax)] = 1           # WELL
    st[z <= z_prod] = 0                             # SOLID
    return st

def state_name(s):
    return {0:"SOLID", 1:"WELL", 2:"POLYMER"}.get(int(s), "UNK")

# triggering “committed” is optional; keep it simple here:
def find_committed_events(state, n_commit):
    events = []
    N = len(state)
    i = 0
    while i < N-1:
        s0 = state[i]
        j = i + 1
        while j < N and state[j] == s0:
            j += 1
        if j >= N:
            break
        s1 = state[j]
        end = min(N, j + n_commit)
        ok = True
        for kk in range(j, end):
            if state[kk] != s1:
                ok = False
                break
        if ok:
            events.append((j, int(s0), int(s1)))
            i = end
        else:
            i = j
    return events

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="zc_LiTotal.csv",
                    help="Extracted centered-z file: frame + Li columns.")
    ap.add_argument("--dt_ps", type=float, default=20.0)

    # Chunk definition
    ap.add_argument("--chunks", type=int, default=4,
                    help="Number of chunk start points (default 4).")
    ap.add_argument("--use_last_ns", type=float, default=None,
                    help="If set, only consider the last X ns of the trajectory for chunking. "
                         "Chunk starts are inside this last window, but plots go to trajectory end.")
    ap.add_argument("--n_li", type=int, default=5,
                    help="Number of Li to select at each chunk start (must be in well at that time).")

    # Region definitions (Å)
    ap.add_argument("--z_wmin", type=float, default=24.0)
    ap.add_argument("--z_wmax", type=float, default=34.0)
    ap.add_argument("--z_prod", type=float, default=22.0)

    # Commitment time (optional markers)
    ap.add_argument("--tcommit_ns", type=float, default=0.5)

    ap.add_argument("--out_prefix", default="chunkReselect",
                    help="Output prefix.")
    args = ap.parse_args()

    apply_house_style()

    df = pd.read_csv(args.csv)
    if "frame" not in df.columns:
        raise RuntimeError("CSV must contain a 'frame' column.")
    li_cols_all = [c for c in df.columns if c != "frame"]
    if len(li_cols_all) < 1:
        raise RuntimeError("No Li columns found.")

    Z = df[li_cols_all].to_numpy(float)
    frames = df["frame"].to_numpy(int)

    dt_ns = args.dt_ps * 1e-3
    time_ns = np.arange(len(frames)) * dt_ns

    # Determine window for chunk start placement
    if args.use_last_ns is None:
        start_global = 0
    else:
        n_last = int(round(args.use_last_ns / dt_ns))
        n_last = min(n_last, len(frames))
        start_global = len(frames) - n_last

    # Chunk start indices within [start_global, end)
    chunks = max(1, args.chunks)
    idx_edges = np.linspace(start_global, len(frames), chunks+1, dtype=int)
    chunk_starts = idx_edges[:-1]  # 4 start points

    n_commit = max(1, int(round(args.tcommit_ns / dt_ns)))

    summary_rows = []

    for ci, start_idx in enumerate(chunk_starts, start=1):
        # Select Li in WELL at this start time
        z0 = Z[start_idx, :]
        st0 = classify_state(z0, args.z_prod, args.z_wmin, args.z_wmax)
        in_well = np.where(st0 == 1)[0]

        if in_well.size == 0:
            print(f"[WARN] chunk {ci}: no Li in well at start_idx={start_idx} (t={time_ns[start_idx]:.2f} ns)")
            continue

        pick = in_well[:args.n_li] if in_well.size >= args.n_li else in_well
        chosen_cols = [li_cols_all[i] for i in pick]
        chosen_idx  = pick.tolist()

        # Save chosen Li list for this chunk
        sel_df = pd.DataFrame({
            "chunk": [ci]*len(chosen_cols),
            "start_frame": [int(frames[start_idx])]*len(chosen_cols),
            "start_time_ns": [float(time_ns[start_idx])]*len(chosen_cols),
            "Li_col": chosen_cols,
            "Li_col_index0based": chosen_idx
        })
        sel_csv = f"{args.out_prefix}_selectedLi_chunk{ci}.csv"
        sel_df.to_csv(sel_csv, index=False)

        # Build trajectory slice from chunk start -> end
        Zc = Z[start_idx:, :][:, chosen_idx]          # (T_end, n_li)
        tc = time_ns[start_idx:]
        frc = frames[start_idx:]

        # Events (optional)
        events_all = []
        for j, name in enumerate(chosen_cols):
            st = classify_state(Zc[:, j], args.z_prod, args.z_wmin, args.z_wmax)
            ev = find_committed_events(st, n_commit)
            for (k, s0, s1) in ev:
                events_all.append({
                    "chunk": ci,
                    "Li": name,
                    "frame": int(frc[k]),
                    "time_ns": float(tc[k]),
                    "from": state_name(s0),
                    "to": state_name(s1),
                })
        ev_df = pd.DataFrame(events_all)
        ev_csv = f"{args.out_prefix}_events_chunk{ci}.csv"
        ev_df.to_csv(ev_csv, index=False)

        # Summary
        summary_rows.append({
            "chunk": ci,
            "start_time_ns": float(time_ns[start_idx]),
            "start_frame": int(frames[start_idx]),
            "n_inwell_at_start": int(in_well.size),
            "n_plotted": int(len(chosen_cols)),
            "events_found": int(len(events_all)),
            "selectedLi_csv": sel_csv,
            "events_csv": ev_csv
        })

        # ----- Plot: z(t) + state timeline -----
        fig = plt.figure(figsize=(13, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.08)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

        # z(t)
        for j, name in enumerate(chosen_cols):
            ax1.plot(tc, Zc[:, j], label=name)

        # region lines
        ax1.axhline(args.z_wmin, linestyle="--", color="k", linewidth=2, label="z_wmin/z_wmax")
        ax1.axhline(args.z_wmax, linestyle="--", color="k", linewidth=2)
        ax1.axhline(args.z_prod, linestyle=":",  color="k", linewidth=2, label="z_prod (solid)")

        ax1.set_ylabel(r"$z_\mathrm{centered}(t)$ (Å)", fontsize=18, fontweight="bold")
        style_axes(ax1)
        ax1.legend(frameon=False, fontsize=10, ncol=2, loc="upper right")

        # state timeline
        S = np.vstack([classify_state(Zc[:, j], args.z_prod, args.z_wmin, args.z_wmax)
                       for j in range(Zc.shape[1])])
        ax2.imshow(S, aspect="auto", interpolation="nearest",
                   extent=[tc[0], tc[-1], -0.5, S.shape[0]-0.5])
        ax2.set_yticks(range(len(chosen_cols)))
        ax2.set_yticklabels(chosen_cols, fontsize=12, fontweight="bold")
        ax2.set_xlabel("time (ns)", fontsize=18, fontweight="bold")
        ax2.set_ylabel("Li", fontsize=18, fontweight="bold")
        style_axes(ax2)

        # committed markers
        if len(events_all) > 0:
            for ev in events_all:
                ax1.axvline(ev["time_ns"], color="k", alpha=0.12, linewidth=2)

        ax1.set_title(
            f"Reselect-in-well @ chunk {ci}: start {time_ns[start_idx]:.2f} ns → end "
            f"| WELL=[{args.z_wmin:.0f},{args.z_wmax:.0f}] Å, commit={args.tcommit_ns:.2f} ns",
            fontsize=18, fontweight="bold"
        )

        out_png = f"{args.out_prefix}_fromChunk{ci}_toEnd.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=300)
        plt.close(fig)

        print(f"[OK] chunk {ci}: {out_png} | selected={sel_csv} | events={ev_csv}")

    # Final summary CSV
    sum_df = pd.DataFrame(summary_rows)
    sum_csv = f"{args.out_prefix}_summary.csv"
    sum_df.to_csv(sum_csv, index=False)
    print(f"[OK] Saved summary: {sum_csv}")


if __name__ == "__main__":
    main()
