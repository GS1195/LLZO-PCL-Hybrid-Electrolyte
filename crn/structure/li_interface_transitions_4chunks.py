#!/usr/bin/env python3
"""
li_interface_transitions_4chunks_70ns.py

Produces exactly 4 chunks:
  C1..C3: fixed 70 ns windows (by frame grid)
  C4: remainder of trajectory

Outputs per chunk counts:
  L(P->S), L(S->P), R(P->S), R(S->P)

Also computes:
  TOT = L(P->S)+L(S->P)+R(P->S)+R(S->P)  (one number per chunk)

Writes ALL outputs into --outdir:
  chunk_transition_counts.csv
  chunk_transition_counts.tex
  li_transitions_events.csv
"""

import argparse
import csv
import os
from math import floor

import MDAnalysis as mda
import numpy as np


# -------------------------
# Helpers
# -------------------------
def centered_z(z_raw, comz, Lz):
    """Center z to solid COM and wrap to (-Lz/2, Lz/2]."""
    zc = z_raw - comz
    return (zc + 0.5 * Lz) % Lz - 0.5 * Lz

def region_labels(z_centered, thresh):
    """Return 1 for SOLID (|z|<=thresh), 0 for POLYMER."""
    return (np.abs(z_centered) <= thresh).astype(np.int8)

def hop_mask(prev_labels, curr_labels, z_prev, z_curr, thresh, hys):
    """
    Robust hop masks with hysteresis.
      P->S: prev polymer & |z_prev| >= (thresh+hys)  AND  curr solid & |z_curr| <= (thresh-hys)
      S->P: prev solid   & |z_prev| <= (thresh-hys)  AND  curr polymer & |z_curr| >= (thresh+hys)
    """
    delta = curr_labels - prev_labels
    p2s_basic = (delta == +1)
    s2p_basic = (delta == -1)
    if hys <= 0.0:
        return p2s_basic, s2p_basic
    p2s_hys = (np.abs(z_prev) >= (thresh + hys)) & (np.abs(z_curr) <= (thresh - hys))
    s2p_hys = (np.abs(z_prev) <= (thresh - hys)) & (np.abs(z_curr) >= (thresh + hys))
    return (p2s_basic & p2s_hys), (s2p_basic & s2p_hys)

def split_frames_3x70_plus_remainder(frames, dt_ps, stride, chunk_ns=70.0, n_fixed=3):
    """
    Returns exactly 4 chunks:
      C1..C3: consecutive windows of chunk_ns (by time grid)
      C4: all remaining frames
    """
    if len(frames) == 0:
        return []

    dt_eff_ps = dt_ps * stride
    chunk_ps  = chunk_ns * 1000.0

    # elapsed time for N frames is (N-1)*dt_eff_ps
    n_frames = int(floor(chunk_ps / dt_eff_ps)) + 1
    n_frames = max(n_frames, 2)

    chunks = []
    start = 0
    for _ in range(n_fixed):
        end = start + n_frames
        chunks.append(frames[start:end])
        start = end

    chunks.append(frames[start:])  # remainder
    return chunks

def safe_mean_time_ns(frames_chunk, dt_ps, stride):
    """Observed time spanned by chunk = (nframes-1)*dt."""
    if len(frames_chunk) < 2:
        return 0.0
    return (len(frames_chunk) - 1) * (dt_ps * stride) / 1000.0


# -------------------------
# Analysis per chunk
# -------------------------
def analyze_chunk(u, solid, groups, frames_chunk, chunk_id, dt_ps, stride,
                  surface_z_A, hysteresis_A, f0_global):
    """
    Returns:
      by_side: direction->side->count summed over selected species
      events: per hop rows
      time_ns: observed chunk time (ns)
    """
    by_side = {
        "P->S": {"-": 0, "+": 0},
        "S->P": {"-": 0, "+": 0},
    }
    events = []

    last_labels = {name: None for name in groups.keys()}
    last_zcent  = {name: None for name in groups.keys()}

    if len(frames_chunk) < 2:
        return by_side, events, 0.0

    dt_eff_ps = dt_ps * stride

    for f in frames_chunk:
        u.trajectory[f]
        Lz   = float(u.dimensions[2])
        COMz = float(solid.center_of_mass()[2])

        time_ps_abs = (f - f0_global) * dt_eff_ps
        time_ns_abs = time_ps_abs / 1000.0

        z_raw_all = u.atoms.positions[:, 2].copy()

        for name, ag in groups.items():
            if ag.n_atoms == 0:
                continue

            z_raw  = z_raw_all[ag.indices]
            z_cent = centered_z(z_raw, COMz, Lz)
            labels = region_labels(z_cent, surface_z_A)

            prev  = last_labels[name]
            zprev = last_zcent[name]
            if prev is not None:
                p2s_mask, s2p_mask = hop_mask(prev, labels, zprev, z_cent, surface_z_A, hysteresis_A)

                if np.any(p2s_mask):
                    idxs = np.where(p2s_mask)[0]
                    for idx in idxs:
                        side = "+" if z_cent[idx] >= 0.0 else "-"
                        by_side["P->S"][side] += 1
                        events.append([
                            chunk_id, name, int(ag.indices[idx]), f,
                            f"{time_ps_abs:.6f}", f"{time_ns_abs:.6f}",
                            "P->S",
                            f"{float(z_cent[idx]):.6f}",
                            f"{float(z_raw[idx]):.6f}",
                            f"{COMz:.6f}", f"{Lz:.6f}",
                            side
                        ])

                if np.any(s2p_mask):
                    idxs = np.where(s2p_mask)[0]
                    for idx in idxs:
                        side = "+" if z_cent[idx] >= 0.0 else "-"
                        by_side["S->P"][side] += 1
                        events.append([
                            chunk_id, name, int(ag.indices[idx]), f,
                            f"{time_ps_abs:.6f}", f"{time_ns_abs:.6f}",
                            "S->P",
                            f"{float(z_cent[idx]):.6f}",
                            f"{float(z_raw[idx]):.6f}",
                            f"{COMz:.6f}", f"{Lz:.6f}",
                            side
                        ])

            last_labels[name] = labels.copy()
            last_zcent[name]  = z_cent.copy()

    time_ns = safe_mean_time_ns(frames_chunk, dt_ps, stride)
    return by_side, events, time_ns


# -------------------------
# LaTeX table writer
# -------------------------
def write_latex_table(path_tex, T_K, chunk_rows, caption_extra=""):
    """
    chunk_rows: list of dicts with keys:
      chunk_label, time_ns, L_P2S, L_S2P, R_P2S, R_S2P, TOT
    """
    times_str = ", ".join([f"{r['time_ns']:.2f}" for r in chunk_rows])
    T_header = f"{T_K} ({times_str})"

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{|c|c|c|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{T (K) [ns/chunk]} & \textbf{Chunk} & "
                 r"\textbf{L (P$\to$S)} & \textbf{L (S$\to$P)} & "
                 r"\textbf{R (P$\to$S)} & \textbf{R (S$\to$P)} & \textbf{TOT} \\ \hline")

    lines.append(rf"\multirow{{{len(chunk_rows)}}}{{*}}{{{T_header}}} "
                 rf"& {chunk_rows[0]['chunk_label']} & {chunk_rows[0]['L_P2S']} & {chunk_rows[0]['L_S2P']} & "
                 rf"{chunk_rows[0]['R_P2S']} & {chunk_rows[0]['R_S2P']} & {chunk_rows[0]['TOT']} \\ \cline{{2-7}}")

    for r in chunk_rows[1:]:
        lines.append(rf" & {r['chunk_label']} & {r['L_P2S']} & {r['L_S2P']} & "
                     rf"{r['R_P2S']} & {r['R_S2P']} & {r['TOT']} \\ \cline{{2-7}}")

    lines[-1] = lines[-1].replace(r"\cline{2-7}", r"\hline")

    lines.append(r"\end{tabular}")
    cap = (r"Chunk-wise transition counts; C1--C3 are fixed 70 ns windows (frame-grid limited), "
           r"C4 is the remainder (time shown in the header parentheses). "
           r"L/R denote left/right interfaces; P$\to$S and S$\to$P are polymer-to-solid and solid-to-polymer transitions. "
           r"TOT is the sum over L/R and both directions.")
    if caption_extra.strip():
        cap += " " + caption_extra.strip()
    lines.append(rf"\caption{{{cap}}}")
    lines.append(rf"\label{{tab:chunk_transitions_{T_K}K}}")
    lines.append(r"\end{table*}")

    with open(path_tex, "w") as f:
        f.write("\n".join(lines) + "\n")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", default="combine_system.dat")
    ap.add_argument("--traj", default="position.lammpstrj")
    ap.add_argument("--dt_ps", type=float, default=20.0)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)

    ap.add_argument("--T", type=int, required=True, help="Temperature in K (for table header/label)")
    ap.add_argument("--chunk_ns", type=float, default=70.0)

    ap.add_argument("--surface_z_A", type=float, default=23.0)
    ap.add_argument("--hysteresis_A", type=float, default=0.5)

    ap.add_argument("--li_types", default="16,17",
                    help="Comma-separated Li atom types to include (sum together)")
    ap.add_argument("--solid_sel", default="type 18 or type 19 or type 20")

    ap.add_argument("--outdir", default="outputs", help="Output directory for all files")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    li_types = [int(x.strip()) for x in args.li_types.split(",") if x.strip()]
    if len(li_types) == 0:
        raise RuntimeError("No Li types provided.")

    u = mda.Universe(args.top, args.traj, topology_format="DATA",
                     format="LAMMPSDUMP", dt=args.dt_ps)
    solid = u.select_atoms(args.solid_sel)
    if solid.n_atoms == 0:
        raise RuntimeError("Solid selection returned 0 atoms. Check --solid_sel.")

    # Build groups
    groups = {}
    for t in li_types:
        ag = u.select_atoms(f"type {t}")
        groups[f"Li(type{t})"] = ag

    # Frame list
    all_frames = list(range(args.start_frame, len(u.trajectory), args.stride))
    if len(all_frames) < 2:
        raise RuntimeError("Not enough frames after start_frame/stride.")

    # Exactly 4 chunks
    chunks = split_frames_3x70_plus_remainder(
        all_frames, args.dt_ps, args.stride, chunk_ns=args.chunk_ns, n_fixed=3
    )

    chunk_rows = []
    event_rows = []

    for cid, fr in enumerate(chunks, start=1):
        if len(fr) < 2:
            chunk_rows.append({
                "chunk_id": cid,
                "chunk_label": f"C{cid}",
                "time_ns": 0.0,
                "L_P2S": 0, "L_S2P": 0, "R_P2S": 0, "R_S2P": 0,
                "TOT": 0
            })
            continue

        by_side, events, time_ns = analyze_chunk(
            u=u, solid=solid, groups=groups,
            frames_chunk=fr, chunk_id=cid,
            dt_ps=args.dt_ps, stride=args.stride,
            surface_z_A=args.surface_z_A,
            hysteresis_A=args.hysteresis_A,
            f0_global=all_frames[0]
        )

        L_P2S = by_side["P->S"]["-"]
        R_P2S = by_side["P->S"]["+"]
        L_S2P = by_side["S->P"]["-"]
        R_S2P = by_side["S->P"]["+"]

        TOT = int(L_P2S + L_S2P + R_P2S + R_S2P)

        chunk_rows.append({
            "chunk_id": cid,
            "chunk_label": f"C{cid}",
            "time_ns": float(time_ns),
            "L_P2S": int(L_P2S), "L_S2P": int(L_S2P),
            "R_P2S": int(R_P2S), "R_S2P": int(R_S2P),
            "TOT": TOT
        })
        event_rows.extend(events)

    # Outputs
    out_csv    = os.path.join(args.outdir, "chunk_transition_counts.csv")
    out_tex    = os.path.join(args.outdir, "chunk_transition_counts.tex")
    out_events = os.path.join(args.outdir, "li_transitions_events.csv")

    # CSV counts
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T_K","chunk","time_ns","L_P2S","L_S2P","R_P2S","R_S2P","TOT"])
        for r in chunk_rows:
            w.writerow([args.T, r["chunk_label"], f"{r['time_ns']:.6f}",
                        r["L_P2S"], r["L_S2P"], r["R_P2S"], r["R_S2P"], r["TOT"]])

    # LaTeX table
    write_latex_table(out_tex, args.T, chunk_rows)

    # Events CSV
    with open(out_events, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id","species","atom_index","frame","time_ps_abs","time_ns_abs",
                    "direction","z_centered_A","z_raw_A","COMz_A","Lz_A","side"])
        w.writerows(event_rows)

    # Console summary
    print("\n=== Chunk transition counts (table format) ===")
    print("T(K)  Chunk  time(ns)   L(P->S)  L(S->P)  R(P->S)  R(S->P)   TOT")
    totals = []
    for r in chunk_rows:
        print(f"{args.T:<5d}  {r['chunk_label']:<5s}  {r['time_ns']:>8.2f}  "
              f"{r['L_P2S']:>7d}  {r['L_S2P']:>7d}  {r['R_P2S']:>7d}  {r['R_S2P']:>7d}  {r['TOT']:>5d}")
        totals.append(r["TOT"])

    print("\n=== 4 numbers (TOT per chunk) ===")
    print("C1  C2  C3  C4")
    print(f"{totals[0]}  {totals[1]}  {totals[2]}  {totals[3]}")

    print("\nWrote:")
    print(f"  {out_csv}")
    print(f"  {out_tex}")
    print(f"  {out_events}")
    print("\nNOTE: Left = side '-' and Right = side '+', using centered z relative to solid COM.\n")


if __name__ == "__main__":
    main()
