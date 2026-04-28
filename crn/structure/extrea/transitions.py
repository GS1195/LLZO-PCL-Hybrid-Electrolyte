#!/usr/bin/env python3
import MDAnalysis as mda
import numpy as np
import csv

# ── Inputs ───────────────────────────────────────────────────
topology    = "combine_system.dat"
trajectory  = "position.lammpstrj"
dt_ps       = 20.0        # ps between frames
start_frame = 6000
stride      = 2           # increase for speed if needed
surface_z_A = 23.0        # Å: |z_centered| <= surface_z_A ⇒ SOLID region

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

# ── Main ─────────────────────────────────────────────────────
results = {}
last_labels = {}
# per-hop event log (destination-frame attributes)
event_rows = []  # each row: (species, atom_index, frame, time_ps, time_ns, direction, z_centered, z_raw, COMz, Lz)

# initialize per-species arrays
for name, ag in groups.items():
    results[name] = {
        "n_li": ag.n_atoms,
        "solid_to_polymer": 0,
        "polymer_to_solid": 0,
    }
    last_labels[name] = None

# iterate frames
frames = list(range(start_frame, len(u.trajectory), stride))
if len(frames) < 2:
    raise RuntimeError("Not enough frames after start_frame/stride to compute transitions.")

for f in frames:
    u.trajectory[f]
    # capture frame box and COM *before* any wrapping/transforms
    Lz  = float(u.dimensions[2])
    COMz = float(solid.center_of_mass()[2])

    # raw z for all atoms (no transform)
    z_raw_all = u.atoms.positions[:, 2].copy()

    for name, ag in groups.items():
        if ag.n_atoms == 0:
            continue

        # raw z of this species (destination frame)
        z_raw = z_raw_all[ag.indices]

        # centered (for classification)
        z_cent = centered_z(z_raw, COMz, Lz)
        labels = region_labels(z_cent, surface_z_A)  # 1=solid, 0=polymer

        prev = last_labels[name]
        if prev is not None:
            # +1 => polymer(0)->solid(1); -1 => solid(1)->polymer(0)
            delta = labels - prev

            # polymer → solid
            p2s_idx = np.where(delta == +1)[0]
            cnt_p2s = int(p2s_idx.size)
            results[name]["polymer_to_solid"] += cnt_p2s
            if cnt_p2s > 0:
                for idx in p2s_idx:
                    atom_index = int(ag.indices[idx])
                    time_ps = (f - frames[0]) * dt_ps
                    time_ns = time_ps / 1000.0
                    event_rows.append([
                        name, atom_index, f, f"{time_ps:.6f}", f"{time_ns:.6f}",
                        "polymer→solid",
                        f"{float(z_cent[idx]):.6f}",   # centered z at destination
                        f"{float(z_raw[idx]):.6f}",   # raw z at destination (no transform)
                        f"{COMz:.6f}", f"{Lz:.6f}"
                    ])

            # solid → polymer
            s2p_idx = np.where(delta == -1)[0]
            cnt_s2p = int(s2p_idx.size)
            results[name]["solid_to_polymer"] += cnt_s2p
            if cnt_s2p > 0:
                for idx in s2p_idx:
                    atom_index = int(ag.indices[idx])
                    time_ps = (f - frames[0]) * dt_ps
                    time_ns = time_ps / 1000.0
                    event_rows.append([
                        name, atom_index, f, f"{time_ps:.6f}", f"{time_ns:.6f}",
                        "solid→polymer",
                        f"{float(z_cent[idx]):.6f}",
                        f"{float(z_raw[idx]):.6f}",
                        f"{COMz:.6f}", f"{Lz:.6f}"
                    ])

        last_labels[name] = labels.copy()

# observation time (ns)
time_ns = (len(frames) - 1) * (dt_ps * stride) / 1000.0

# compute rates
for name in results:
    n_li = max(results[name]["n_li"], 1)  # avoid /0
    s2p = results[name]["solid_to_polymer"]
    p2s = results[name]["polymer_to_solid"]
    tot = s2p + p2s

    if time_ns > 0:
        results[name]["rate_s2p_per_ns"]      = s2p / time_ns
        results[name]["rate_p2s_per_ns"]      = p2s / time_ns
        results[name]["rate_total_per_ns"]    = tot / time_ns
        results[name]["rate_s2p_per_ns_per_ion"]   = (s2p / time_ns) / n_li
        results[name]["rate_p2s_per_ns_per_ion"]   = (p2s / time_ns) / n_li
        results[name]["rate_total_per_ns_per_ion"] = (tot / time_ns) / n_li
    else:
        for k in ["rate_s2p_per_ns","rate_p2s_per_ns","rate_total_per_ns",
                  "rate_s2p_per_ns_per_ion","rate_p2s_per_ns_per_ion","rate_total_per_ns_per_ion"]:
            results[name][k] = 0.0

# ── Print summary ────────────────────────────────────────────
print(f"\nΔt = {dt_ps} ps, start_frame = {start_frame}, stride = {stride}")
print(f"Interface threshold: |z_centered| <= {surface_z_A} Å ⇒ SOLID region.")
print(f"Observed time: {time_ns:.6f} ns")

for name, r in results.items():
    print(f"\n{name}  (n_Li = {r['n_li']})")
    print(f"  solid→polymer: {r['solid_to_polymer']}  |  rate: {r['rate_s2p_per_ns']:.6g} /ns  |  per-ion: {r['rate_s2p_per_ns_per_ion']:.6g} /ns/ion")
    print(f"  polymer→solid: {r['polymer_to_solid']}  |  rate: {r['rate_p2s_per_ns']:.6g} /ns  |  per-ion: {r['rate_p2s_per_ns_per_ion']:.6g} /ns/ion")
    print(f"  total hops    : {r['solid_to_polymer'] + r['polymer_to_solid']}  |  rate: {r['rate_total_per_ns']:.6g} /ns  |  per-ion: {r['rate_total_per_ns_per_ion']:.6g} /ns/ion")

# ── Save CSV (summary) ───────────────────────────────────────
with open("li_transitions_2state.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["species","n_Li","time_ns",
                "solid_to_polymer","polymer_to_solid","total_hops",
                "rate_s2p_per_ns","rate_p2s_per_ns","rate_total_per_ns",
                "rate_s2p_per_ns_per_ion","rate_p2s_per_ns_per_ion","rate_total_per_ns_per_ion"])
    for name, r in results.items():
        s2p = r["solid_to_polymer"]; p2s = r["polymer_to_solid"]; tot = s2p + p2s
        w.writerow([name, r["n_li"], f"{time_ns:.6f}",
                    s2p, p2s, tot,
                    f"{r['rate_s2p_per_ns']:.6g}", f"{r['rate_p2s_per_ns']:.6g}", f"{r['rate_total_per_ns']:.6g}",
                    f"{r['rate_s2p_per_ns_per_ion']:.6g}", f"{r['rate_p2s_per_ns_per_ion']:.6g}", f"{r['rate_total_per_ns_per_ion']:.6g}"])

# ── Save CSV (per-hop events, with raw z) ─────────────────────
with open("li_transitions_events.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["species","atom_index","frame","time_ps","time_ns",
                "direction","z_centered_A","z_raw_A","COMz_A","Lz_A"])
    w.writerows(event_rows)
