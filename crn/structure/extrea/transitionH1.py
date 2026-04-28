#!/usr/bin/env python3
import MDAnalysis as mda
import numpy as np
import csv
from collections import defaultdict
from math import ceil

# ── Inputs ───────────────────────────────────────────────────
topology    = "combine_system.dat"
trajectory  = "position.lammpstrj"
dt_ps       = 20.0        # ps between frames
start_frame = 0
stride      = 1           # increase for speed if needed

# Region definition (2-state model)
surface_z_A  = 23.0       # Å: |z_centered| <= surface_z_A ⇒ SOLID region
hysteresis_A = 1.5        # Å: margin to avoid flicker; set 0 to disable

# Chunking
n_chunks     = 4          # number of chunks to split the analyzed frames into

# Atom type mapping
types = {
    "Li(PCL)" : 16,
    "Li(LLZO)": 17,
}
solid_selection = "type 18 or type 19 or type 20"   # atoms that define solid COM

# ── Load ─────────────────────────────────────────────────────
u = mda.Universe(topology, trajectory,
                 topology_format="DATA", format="LAMMPSDUMP", dt=dt_ps)
solid = u.select_atoms(solid_selection)
if solid.n_atoms == 0:
    raise RuntimeError("Solid selection returned 0 atoms. Check 'solid_selection'.")

groups = {name: u.select_atoms(f"type {t}") for name, t in types.items()}

# ── Helpers ──────────────────────────────────────────────────
def centered_z(z_raw, comz, Lz):
    """Center z to solid COM and wrap to (-Lz/2, Lz/2]."""
    zc = z_raw - comz
    return (zc + 0.5*Lz) % Lz - 0.5*Lz

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

def split_frames_into_chunks(frames, n_chunks):
    """Evenly split a list of frame indices into n_chunks non-empty segments when possible."""
    if n_chunks <= 1:
        return [frames]
    # Use numpy-style split without importing numpy for this tiny task
    total = len(frames)
    base = total // n_chunks
    rem  = total % n_chunks
    chunks = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        if size == 0:  # if too few frames, we may produce fewer than n_chunks
            continue
        end = start + size
        chunks.append(frames[start:end])
        start = end
    return chunks

# ── Analysis per chunk ───────────────────────────────────────
def analyze_chunk(frames_chunk, chunk_id):
    """
    Returns:
      res_summary: dict species -> counts & rates for this chunk
      by_side: dict species -> direction -> {+/-: count}
      events: list of per-hop event rows (with chunk_id)
      time_ns: observed time in ns for this chunk
    """
    res_summary = {}
    by_side = {name: {"polymer→solid": {"+": 0, "-": 0},
                      "solid→polymer": {"+": 0, "-": 0}} for name in types.keys()}
    events = []

    # init
    last_labels = {name: None for name in types.keys()}
    last_zcent  = {name: None for name in types.keys()}

    # prefill counts
    for name, ag in groups.items():
        res_summary[name] = {
            "n_li": ag.n_atoms,
            "solid_to_polymer": 0,
            "polymer_to_solid": 0,
        }

    if len(frames_chunk) < 2:
        return res_summary, by_side, events, 0.0

    for f in frames_chunk:
        u.trajectory[f]
        Lz  = float(u.dimensions[2])
        COMz = float(solid.center_of_mass()[2])
        z_raw_all = u.atoms.positions[:, 2].copy()

        for name, ag in groups.items():
            if ag.n_atoms == 0:
                continue
            z_raw  = z_raw_all[ag.indices]
            z_cent = centered_z(z_raw, COMz, Lz)
            labels = region_labels(z_cent, surface_z_A)

            prev = last_labels[name]
            zprev = last_zcent[name]
            if prev is not None:
                p2s_mask, s2p_mask = hop_mask(prev, labels, zprev, z_cent, surface_z_A, hysteresis_A)

                # polymer → solid
                if np.any(p2s_mask):
                    idxs = np.where(p2s_mask)[0]
                    res_summary[name]["polymer_to_solid"] += int(idxs.size)
                    for idx in idxs:
                        atom_index = int(ag.indices[idx])
                        time_ps = (f - frames_chunk[0]) * dt_ps
                        time_ns = time_ps / 1000.0
                        side = "+" if z_cent[idx] >= 0.0 else "-"
                        by_side[name]["polymer→solid"][side] += 1
                        events.append([
                            chunk_id, name, atom_index, f, f"{time_ps:.6f}", f"{time_ns:.6f}",
                            "polymer→solid",
                            f"{float(z_cent[idx]):.6f}",
                            f"{float(z_raw[idx]):.6f}",
                            f"{COMz:.6f}", f"{Lz:.6f}",
                            side
                        ])

                # solid → polymer
                if np.any(s2p_mask):
                    idxs = np.where(s2p_mask)[0]
                    res_summary[name]["solid_to_polymer"] += int(idxs.size)
                    for idx in idxs:
                        atom_index = int(ag.indices[idx])
                        time_ps = (f - frames_chunk[0]) * dt_ps
                        time_ns = time_ps / 1000.0
                        side = "+" if z_cent[idx] >= 0.0 else "-"
                        by_side[name]["solid→polymer"][side] += 1
                        events.append([
                            chunk_id, name, atom_index, f, f"{time_ps:.6f}", f"{time_ns:.6f}",
                            "solid→polymer",
                            f"{float(z_cent[idx]):.6f}",
                            f"{float(z_raw[idx]):.6f}",
                            f"{COMz:.6f}", f"{Lz:.6f}",
                            side
                        ])

            last_labels[name] = labels.copy()
            last_zcent[name]  = z_cent.copy()

    time_ns = (len(frames_chunk) - 1) * (dt_ps * stride) / 1000.0
    # add rates into res_summary
    for name in res_summary:
        n_li = max(res_summary[name]["n_li"], 1)
        s2p = res_summary[name]["solid_to_polymer"]
        p2s = res_summary[name]["polymer_to_solid"]
        tot = s2p + p2s
        if time_ns > 0:
            res_summary[name]["rate_s2p_per_ns"]      = s2p / time_ns
            res_summary[name]["rate_p2s_per_ns"]      = p2s / time_ns
            res_summary[name]["rate_total_per_ns"]    = tot / time_ns
            res_summary[name]["rate_s2p_per_ns_per_ion"]   = (s2p / time_ns) / n_li
            res_summary[name]["rate_p2s_per_ns_per_ion"]   = (p2s / time_ns) / n_li
            res_summary[name]["rate_total_per_ns_per_ion"] = (tot / time_ns) / n_li
        else:
            for k in ["rate_s2p_per_ns","rate_p2s_per_ns","rate_total_per_ns",
                      "rate_s2p_per_ns_per_ion","rate_p2s_per_ns_per_ion","rate_total_per_ns_per_ion"]:
                res_summary[name][k] = 0.0

    return res_summary, by_side, events, time_ns

# ── Build frame list and split ───────────────────────────────
all_frames = list(range(start_frame, len(u.trajectory), stride))
if len(all_frames) < 2:
    raise RuntimeError("Not enough frames after start_frame/stride to compute transitions.")
chunks = split_frames_into_chunks(all_frames, n_chunks)

# ── Run all chunks ───────────────────────────────────────────
chunk_results = []   # list of (chunk_id, res_summary, by_side, events, time_ns)
for cid, fr in enumerate(chunks, start=1):
    if len(fr) < 2:
        # skip degenerate chunk
        continue
    res_summary, by_side_res, events, t_ns = analyze_chunk(fr, cid)
    chunk_results.append((cid, res_summary, by_side_res, events, t_ns))

# ── Aggregate totals & averages ──────────────────────────────
# Totals: sum counts and time across chunks; rates recomputed from totals/time
totals = {name: {"n_li": groups[name].n_atoms,
                 "solid_to_polymer": 0, "polymer_to_solid": 0} for name in types.keys()}
total_time_ns = 0.0

# per-chunk rows for CSV
rows_chunks = []
rows_by_side = []
event_rows = []

for (cid, res_summary, by_side_res, events, t_ns) in chunk_results:
    total_time_ns += t_ns
    for name in types.keys():
        s2p = res_summary[name]["solid_to_polymer"]
        p2s = res_summary[name]["polymer_to_solid"]
        tot = s2p + p2s
        totals[name]["solid_to_polymer"] += s2p
        totals[name]["polymer_to_solid"] += p2s

        rows_chunks.append([
            cid, name, groups[name].n_atoms, f"{t_ns:.6f}",
            s2p, p2s, tot,
            f"{res_summary[name]['rate_s2p_per_ns']:.6g}",
            f"{res_summary[name]['rate_p2s_per_ns']:.6g}",
            f"{res_summary[name]['rate_total_per_ns']:.6g}",
            f"{res_summary[name]['rate_s2p_per_ns_per_ion']:.6g}",
            f"{res_summary[name]['rate_p2s_per_ns_per_ion']:.6g}",
            f"{res_summary[name]['rate_total_per_ns_per_ion']:.6g}",
            hysteresis_A, surface_z_A, dt_ps, start_frame, stride
        ])

        # by-side rows
        for direction in ["polymer→solid","solid→polymer"]:
            for side in ["+","-"]:
                rows_by_side.append([
                    cid, name, direction, side, by_side_res[name][direction][side], f"{t_ns:.6f}"
                ])

    # append events for this chunk
    event_rows.extend(events)

# Recompute total rates from aggregated counts/time
rows_totals = []
for name in types.keys():
    n_li = max(totals[name]["n_li"], 1)
    s2p = totals[name]["solid_to_polymer"]
    p2s = totals[name]["polymer_to_solid"]
    tot = s2p + p2s
    if total_time_ns > 0:
        r_s2p   = s2p / total_time_ns
        r_p2s   = p2s / total_time_ns
        r_total = tot / total_time_ns
        r_s2p_pi   = r_s2p / n_li
        r_p2s_pi   = r_p2s / n_li
        r_total_pi = r_total / n_li
    else:
        r_s2p = r_p2s = r_total = r_s2p_pi = r_p2s_pi = r_total_pi = 0.0

    rows_totals.append([
        name, n_li, f"{total_time_ns:.6f}",
        s2p, p2s, tot,
        f"{r_s2p:.6g}", f"{r_p2s:.6g}", f"{r_total:.6g}",
        f"{r_s2p_pi:.6g}", f"{r_p2s_pi:.6g}", f"{r_total_pi:.6g}",
        hysteresis_A, surface_z_A, dt_ps, start_frame, stride, n_chunks
    ])

# Averages across chunks (mean of per-chunk metrics)
# For rates this is the simple arithmetic mean of per-chunk rates; for counts it's mean per chunk.
rows_avgs = []
for name in types.keys():
    # collect per-chunk values
    chunk_vals = [r for (cid, res, _, _, _) in chunk_results for k,r in []]  # placeholder
    # prepare accumulators
    cnt_s2p = []; cnt_p2s = []; cnt_tot = []
    rt_s2p  = []; rt_p2s  = []; rt_tot  = []
    rt_s2p_pi = []; rt_p2s_pi = []; rt_tot_pi = []
    valid_chunks = 0
    for (cid, res_summary, _, _, t_ns) in chunk_results:
        valid_chunks += 1
        s2p = res_summary[name]["solid_to_polymer"]
        p2s = res_summary[name]["polymer_to_solid"]
        tot = s2p + p2s
        cnt_s2p.append(s2p); cnt_p2s.append(p2s); cnt_tot.append(tot)
        rt_s2p.append(res_summary[name]["rate_s2p_per_ns"])
        rt_p2s.append(res_summary[name]["rate_p2s_per_ns"])
        rt_tot.append(res_summary[name]["rate_total_per_ns"])
        rt_s2p_pi.append(res_summary[name]["rate_s2p_per_ns_per_ion"])
        rt_p2s_pi.append(res_summary[name]["rate_p2s_per_ns_per_ion"])
        rt_tot_pi.append(res_summary[name]["rate_total_per_ns_per_ion"])

    if valid_chunks > 0:
        mean_counts = (np.mean(cnt_s2p), np.mean(cnt_p2s), np.mean(cnt_tot))
        mean_rates  = (np.mean(rt_s2p), np.mean(rt_p2s), np.mean(rt_tot))
        mean_rates_pi = (np.mean(rt_s2p_pi), np.mean(rt_p2s_pi), np.mean(rt_tot_pi))
    else:
        mean_counts = (0.0, 0.0, 0.0)
        mean_rates  = (0.0, 0.0, 0.0)
        mean_rates_pi = (0.0, 0.0, 0.0)

    rows_avgs.append([
        name, valid_chunks,
        f"{mean_counts[0]:.6g}", f"{mean_counts[1]:.6g}", f"{mean_counts[2]:.6g}",
        f"{mean_rates[0]:.6g}",  f"{mean_rates[1]:.6g}",  f"{mean_rates[2]:.6g}",
        f"{mean_rates_pi[0]:.6g}", f"{mean_rates_pi[1]:.6g}", f"{mean_rates_pi[2]:.6g}",
        hysteresis_A, surface_z_A, dt_ps, start_frame, stride, n_chunks
    ])

# ── Write CSVs ───────────────────────────────────────────────
# per-chunk summary
with open("li_transitions_2state_chunks.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["chunk_id","species","n_Li","time_ns",
                "solid_to_polymer","polymer_to_solid","total_hops",
                "rate_s2p_per_ns","rate_p2s_per_ns","rate_total_per_ns",
                "rate_s2p_per_ns_per_ion","rate_p2s_per_ns_per_ion","rate_total_per_ns_per_ion",
                "hysteresis_A","surface_z_A","dt_ps","start_frame","stride"])
    w.writerows(rows_chunks)

# by-side per chunk
with open("li_transitions_by_side_chunks.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["chunk_id","species","direction","side","count","time_ns"])
    w.writerows(rows_by_side)

# overall totals
with open("li_transitions_2state_totals.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["species","n_Li","time_ns",
                "solid_to_polymer","polymer_to_solid","total_hops",
                "rate_s2p_per_ns","rate_p2s_per_ns","rate_total_per_ns",
                "rate_s2p_per_ns_per_ion","rate_p2s_per_ns_per_ion","rate_total_per_ns_per_ion",
                "hysteresis_A","surface_z_A","dt_ps","start_frame","stride","n_chunks"])
    w.writerows(rows_totals)

# averages over chunks
with open("li_transitions_2state_averages.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["species","n_chunks_used",
                "mean_counts_s2p","mean_counts_p2s","mean_counts_total",
                "mean_rate_s2p_per_ns","mean_rate_p2s_per_ns","mean_rate_total_per_ns",
                "mean_rate_s2p_per_ns_per_ion","mean_rate_p2s_per_ns_per_ion","mean_rate_total_per_ns_per_ion",
                "hysteresis_A","surface_z_A","dt_ps","start_frame","stride","n_chunks_requested"])
    w.writerows(rows_avgs)

# per-hop events with chunk id
with open("li_transitions_events.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["chunk_id","species","atom_index","frame","time_ps","time_ns",
                "direction","z_centered_A","z_raw_A","COMz_A","Lz_A","side"])
    w.writerows(event_rows)

# ── Console summary ──────────────────────────────────────────
print(f"\nΔt = {dt_ps} ps, start_frame = {start_frame}, stride = {stride}")
print(f"Interface threshold: |z_centered| <= {surface_z_A} Å; hysteresis: {hysteresis_A} Å")
print(f"Frames analyzed: {len(all_frames)}; chunks used: {len(chunks)}")
print("Wrote:")
print("  li_transitions_2state_chunks.csv")
print("  li_transitions_by_side_chunks.csv")
print("  li_transitions_2state_totals.csv")
print("  li_transitions_2state_averages.csv")
print("  li_transitions_events.csv")
