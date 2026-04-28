#!/usr/bin/env python3
"""
Surface-relative projection maps for Li, TFSI, and PCL near solid terminations,
with tolerance control, areal & volumetric CSVs, per-species figures (centered
shared colorbar), overlay plots (Li filled + TFSI/PCL contours), and RGB composites.

Outputs go to: ./heatmap_outputs_multi_species heat_map_near_surface_combine2.py
"""

import csv
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import MDAnalysis as mda

# Use constrained layout globally (avoids tight_layout warnings)
mpl.rcParams['figure.constrained_layout.use'] = True

# ─────────────────────────────────────────────────────────────────────────────
# USER INPUTS
# ─────────────────────────────────────────────────────────────────────────────

data_file       = "combine_system.dat"
trajectory_file = "position.lammpstrj"

# Solid atom types (ceramic slab)
solid_types = [18, 19, 20]

# Species definitions
species_types = {
    "Li":   [16, 17],
    "TFSI": [14],
    "PCL":  [3],
}

# Polymer-side slab thicknesses (Å)
z1_bottom = 10.0   # bottom slab: [zmin_ref − z1_bottom, zmin_ref]
z2_top    = 10.0   # top slab:    [zmax_ref, zmax_ref + z2_top]

# Tolerance near the solid surface
#   "into_solid": shift the reference plane by +tol (bottom) / −tol (top) into the solid
#   "polymer_gap": keep surface plane but exclude a tol-thick gap on polymer side
#   "none": no tolerance
surface_tolerance = 0.5
tolerance_mode    = "into_solid"   # "into_solid" | "polymer_gap" | "none"

# 2D grid resolution
n_bins_x = 100
n_bins_y = 100

# Trajectory spacing (ps) for logs only
frame_dt_ps = 20

# Averaging mode:
#   "per_frame"               → divide by total processed frames (solid present)
#   "conditional_on_presence" → divide by frames where the slab had ≥1 atoms for that species
averaging_mode = "per_frame"

# Surface detection at frame 0: "pbc_arc" (robust) or "naive_minmax"
surface_method = "pbc_arc"

# Warn if apparent solid thickness drifts > tol (Å). Set None to disable.
solid_arc_width_tolerance_A = 0.5

# Outputs
out_dir = Path("heatmap_outputs_multi_species2")
out_dir.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def selection_str_from_types(types_list):
    return "type " + " ".join(str(t) for t in types_list)

def minimal_arc_z_range(zvals, Lz):
    if len(zvals) == 0: return (0.0, 0.0)
    z = np.sort(zvals % Lz)
    diffs = np.diff(z, append=z[0] + Lz)
    k = np.argmax(diffs)
    zmin = z[(k + 1) % len(z)]
    zmax = z[k]
    return float(zmin), float(zmax)

def circular_mean_z(zvals, Lz):
    if len(zvals) == 0: return 0.0
    theta = 2.0 * np.pi * (zvals % Lz) / Lz
    C = np.mean(np.cos(theta)); S = np.mean(np.sin(theta))
    ang = np.arctan2(S, C) % (2.0 * np.pi)
    return float((ang * Lz) / (2.0 * np.pi))

def arc_width(zmin, zmax, L):
    return (zmax - zmin) % L

def in_range_periodic(zvals, start, end, L):
    start = start % L; end = end % L; z = zvals % L
    if start <= end:
        return (z >= start) & (z <= end)
    else:
        return (z >= start) | (z <= end)

def save_edges_to_csv(edges, path):
    np.savetxt(path, edges, delimiter=",", header="edges", comments="")

def save_map_longform(map2d, x_edges, y_edges, path_csv, header_name):
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    with open(path_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_center_A", "y_center_A", header_name])
        for i, xc in enumerate(x_centers):
            for j, yc in enumerate(y_centers):
                w.writerow([f"{xc:.8f}", f"{yc:.8f}", f"{map2d[i, j]:.12e}"])

def average_map(sum_map, total_frames, frames_with_presence, mode):
    if mode == "per_frame":
        denom = max(total_frames, 1)
    elif mode == "conditional_on_presence":
        denom = max(frames_with_presence, 1)
    else:
        raise ValueError("averaging_mode must be 'per_frame' or 'conditional_on_presence'")
    return sum_map / float(denom)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & STATIC SURFACES
# ─────────────────────────────────────────────────────────────────────────────

u = mda.Universe(
    data_file,
    trajectory_file,
    topology_format="DATA",
    format="LAMMPSDUMP",
    dt=frame_dt_ps
)

solid_ag = u.select_atoms(selection_str_from_types(solid_types))
species_ags = {name: u.select_atoms(selection_str_from_types(tlist))
               for name, tlist in species_types.items()}

u.trajectory[0]; u.atoms.wrap()
Lx0, Ly0, Lz0 = u.dimensions[:3]
x_edges = np.linspace(0.0, Lx0, n_bins_x + 1)
y_edges = np.linspace(0.0, Ly0, n_bins_y + 1)
dx = x_edges[1] - x_edges[0]; dy = y_edges[1] - y_edges[0]
bin_area = dx * dy

save_edges_to_csv(x_edges, out_dir / "x_edges.csv")
save_edges_to_csv(y_edges, out_dir / "y_edges.csv")

# Static surfaces (frame 0)
z_solid0 = solid_ag.positions[:, 2]
if surface_method == "naive_minmax":
    zmin_solid = float(np.min(z_solid0 % Lz0))
    zmax_solid = float(np.max(z_solid0 % Lz0))
else:
    zmin_solid, zmax_solid = minimal_arc_z_range(z_solid0, Lz0)
zcom_solid   = circular_mean_z(z_solid0, Lz0)
width_static = arc_width(zmin_solid, zmax_solid, Lz0)

with open(out_dir / "solid_surfaces_static.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["Lz", "zmin_solid", "zmax_solid", "zCOM_solid", "arc_width",
                "tolerance_mode", "surface_tolerance_A"])
    w.writerow([Lz0, zmin_solid, zmax_solid, zcom_solid, width_static,
                tolerance_mode, surface_tolerance])

print(f"[static] Lz={Lz0:.3f}  zmin={zmin_solid:.3f}  zmax={zmax_solid:.3f}  zCOM={zcom_solid:.3f}  width={width_static:.3f} Å")
print(f"[tolerance] mode={tolerance_mode}  tol={surface_tolerance} Å")

# ─────────────────────────────────────────────────────────────────────────────
# ACCUMULATORS & LOG
# ─────────────────────────────────────────────────────────────────────────────

accum = {name: {"bottom": np.zeros((n_bins_x, n_bins_y), dtype=np.float64),
                "top":    np.zeros((n_bins_x, n_bins_y), dtype=np.float64)}
         for name in species_types}

frames_total = 0
frames_with = {name: {"bottom": 0, "top": 0} for name in species_types}

per_frame_csv = out_dir / "solid_surfaces_per_frame.csv"
with open(per_frame_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow([
        "frame","time_ps","Lz",
        "tolerance_mode","surface_tolerance_A",
        "zmin_solid","zmax_solid","zCOM_solid",
        "zmin_eff","zmax_eff",
        "bottom_from_raw","bottom_to_raw","top_from_raw","top_to_raw",
        "bottom_from_mod","bottom_to_mod","top_from_mod","top_to_mod",
        "solid_arc_width_now"
    ])

    for f, ts in enumerate(u.trajectory):
        u.atoms.wrap()
        Lx, Ly, Lz = ts.dimensions[:3]

        z_solid = solid_ag.positions[:, 2]
        if z_solid.size == 0:
            continue

        frames_total += 1

        tol = float(surface_tolerance) if tolerance_mode != "none" else 0.0

        if tolerance_mode == "into_solid":
            zmin_eff = (zmin_solid + tol) % Lz
            zmax_eff = (zmax_solid - tol) % Lz
            b_from_raw = zmin_eff - z1_bottom; b_to_raw = zmin_eff
            t_from_raw = zmax_eff;              t_to_raw = zmax_eff + z2_top

        elif tolerance_mode == "polymer_gap":
            zmin_eff = zmin_solid; zmax_eff = zmax_solid
            b_from_raw = (zmin_solid - z1_bottom - tol); b_to_raw = (zmin_solid - tol)
            t_from_raw = (zmax_solid + tol);             t_to_raw = (zmax_solid + z2_top + tol)

        elif tolerance_mode == "none":
            zmin_eff = zmin_solid; zmax_eff = zmax_solid
            b_from_raw = zmin_eff - z1_bottom; b_to_raw = zmin_eff
            t_from_raw = zmax_eff;              t_to_raw = zmax_eff + z2_top

        else:
            raise ValueError("tolerance_mode must be 'into_solid', 'polymer_gap', or 'none'.")

        # Optional guardrail: check apparent thickness
        if solid_arc_width_tolerance_A is not None:
            zmin_now, zmax_now = minimal_arc_z_range(z_solid, Lz)
            width_now = arc_width(zmin_now, zmax_now, Lz)
            if abs(width_now - width_static) > solid_arc_width_tolerance_A:
                print(f"[warn frame {f}] solid arc width changed from {width_static:.3f} to {width_now:.3f} Å")

        # Log exact slab limits used (raw and modulo-Lz)
        b_from_mod = b_from_raw % Lz; b_to_mod = b_to_raw % Lz
        t_from_mod = t_from_raw % Lz; t_to_mod = t_to_raw % Lz
        w.writerow([
            f, f*frame_dt_ps, Lz,
            tolerance_mode, tol,
            zmin_solid, zmax_solid, zcom_solid,
            zmin_eff, zmax_eff,
            b_from_raw, b_to_raw, t_from_raw, t_to_raw,
            b_from_mod, b_to_mod, t_from_mod, t_to_mod,
            width_static
        ])

        # Accumulate per-species hist2d
        for name, ag in species_ags.items():
            if ag.n_atoms == 0:
                continue
            pos = ag.positions
            if pos.size == 0:
                continue

            x = pos[:, 0] % Lx
            y = pos[:, 1] % Ly
            z = pos[:, 2] % Lz

            mask_bottom = in_range_periodic(z, b_from_raw, b_to_raw, Lz)
            mask_top    = in_range_periodic(z, t_from_raw, t_to_raw, Lz)

            if np.any(mask_bottom):
                H_b, _, _ = np.histogram2d(x[mask_bottom], y[mask_bottom], bins=[x_edges, y_edges])
                accum[name]["bottom"] += H_b
                frames_with[name]["bottom"] += 1

            if np.any(mask_top):
                H_t, _, _ = np.histogram2d(x[mask_top], y[mask_top], bins=[x_edges, y_edges])
                accum[name]["top"] += H_t
                frames_with[name]["top"] += 1

# ─────────────────────────────────────────────────────────────────────────────
# AVERAGING, DENSITIES, CSVs, PER-SPECIES FIGURES (CENTERED COLORBAR)
# ─────────────────────────────────────────────────────────────────────────────

def density_maps(sum_bottom, sum_top, frames_total, frames_with_bottom, frames_with_top):
    avg_b = average_map(sum_bottom, frames_total, frames_with_bottom, averaging_mode)
    avg_t = average_map(sum_top,    frames_total, frames_with_top,    averaging_mode)
    sigma_b = avg_b / bin_area                           # atoms / Å^2
    sigma_t = avg_t / bin_area
    rho_b   = avg_b / (bin_area * z1_bottom) if z1_bottom > 0 else np.zeros_like(avg_b)  # atoms / Å^3
    rho_t   = avg_t / (bin_area * z2_top)    if z2_top    > 0 else np.zeros_like(avg_t)
    return avg_b, avg_t, sigma_b, sigma_t, rho_b, rho_t

# collect areal maps for overlays
sigma_all = { "Li": {"bottom": None, "top": None},
              "TFSI": {"bottom": None, "top": None},
              "PCL": {"bottom": None, "top": None} }

for name in ["Li", "TFSI", "PCL"]:
    sum_b = accum[name]["bottom"]
    sum_t = accum[name]["top"]
    fw_b  = frames_with[name]["bottom"]
    fw_t  = frames_with[name]["top"]

    avg_b, avg_t, sigma_b, sigma_t, rho_b, rho_t = density_maps(sum_b, sum_t, frames_total, fw_b, fw_t)

    # Save CSVs (long-form)
    save_map_longform(sigma_b, x_edges, y_edges, out_dir / f"areal_{name}_bottom_atoms_per_A2.csv", "sigma_atoms_per_A2")
    save_map_longform(sigma_t, x_edges, y_edges, out_dir / f"areal_{name}_top_atoms_per_A2.csv",    "sigma_atoms_per_A2")
    save_map_longform(rho_b,   x_edges, y_edges, out_dir / f"volumetric_{name}_bottom_atoms_per_A3.csv", "rho_atoms_per_A3")
    save_map_longform(rho_t,   x_edges, y_edges, out_dir / f"volumetric_{name}_top_atoms_per_A3.csv",    "rho_atoms_per_A3")

    # Per-species areal density figure with a centered colorbar between panels
    # Use a 1x3 GridSpec: left map | slim cbar axis | right map
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.055, 1.0])

    axL = fig.add_subplot(gs[0, 0])
    axC = fig.add_subplot(gs[0, 1])   # dedicated colorbar axis (center column)
    axR = fig.add_subplot(gs[0, 2], sharex=axL, sharey=axL)

    # Shared color scale (within this species)
    vmin = min(np.min(sigma_b), np.min(sigma_t))
    vmax = max(np.max(sigma_b), np.max(sigma_t))

    m1 = axL.pcolormesh(x_edges, y_edges, sigma_b.T, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
    axL.set_title(f"{name} — Bottom\nz ∈ [zmin_ref − {z1_bottom:.2f}, zmin_ref]")
    axL.set_xlabel("x (Å)"); axL.set_ylabel("y (Å)")

    m2 = axR.pcolormesh(x_edges, y_edges, sigma_t.T, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
    axR.set_title(f"{name} — Top\nz ∈ [zmax_ref, zmax_ref + {z2_top:.2f}]")
    axR.set_xlabel("x (Å)"); axR.set_ylabel("y (Å)")

    # Centered colorbar spanning both maps (uses the dedicated middle axis)
    cbar = fig.colorbar(m2, cax=axC)
    cbar.set_label(f"Areal density σ (atoms Å$^{{-2}}$), mode={averaging_mode}")

    fig.savefig(out_dir / f"{name}_areal_density.png", dpi=300)
    plt.close(fig)

    # store for overlays
    sigma_all[name]["bottom"] = sigma_b
    sigma_all[name]["top"]    = sigma_t

# ─────────────────────────────────────────────────────────────────────────────
# OVERLAY FIGURES (Li filled; TFSI & PCL as contours) + RGB COMPOSITES
# ─────────────────────────────────────────────────────────────────────────────

def make_overlay(side: str):
    assert side in ("bottom", "top")
    base = sigma_all["Li"][side]
    tfs  = sigma_all["TFSI"][side]
    pcl  = sigma_all["PCL"][side]

    # shared color scale for the filled Li map
    vmin = np.nanmin(base); vmax = np.nanmax(base)

    x_centers = 0.5*(x_edges[:-1] + x_edges[1:])
    y_centers = 0.5*(y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="xy")

    def quant_levels(M, qs=(0.60, 0.85)):
        flat_pos = M.ravel()[M.ravel() > 0]
        if flat_pos.size == 0: return None
        return np.quantile(flat_pos, qs)

    tfs_lvls = quant_levels(tfs)
    pcl_lvls = quant_levels(pcl)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.8))
    m = ax.pcolormesh(x_edges, y_edges, base.T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    cb = fig.colorbar(m, ax=ax)
    cb.set_label("Li areal density σ (atoms Å$^{-2}$)")

    proxies, labels = [], []
    if tfs_lvls is not None:
        ax.contour(Xc, Yc, tfs.T, levels=tfs_lvls, colors="red", linewidths=(1.2, 1.8))
        proxies.append(mlines.Line2D([], [], color="red", lw=2)); labels.append("TFSI contours")
    if pcl_lvls is not None:
        ax.contour(Xc, Yc, pcl.T, levels=pcl_lvls, colors="blue", linewidths=(1.2, 1.8))
        proxies.append(mlines.Line2D([], [], color="blue", lw=2)); labels.append("PCL contours")
    if proxies:
        ax.legend(proxies, labels, loc="upper right", frameon=True)

    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
    ax.set_title(f"Overlay ({side} slab): Li filled, TFSI/PCL contours")
    fig.savefig(out_dir / f"overlay_{side}.png", dpi=300)
    plt.close(fig)

    # RGB composite (R=TFSI, G=Li, B=PCL) — draw with imshow on a decent-sized figure
    def norm01(A):
        a = A.astype(float)
        mn, mx = np.nanmin(a), np.nanmax(a)
        return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a)

    rgb = np.stack([norm01(tfs.T), norm01(base.T), norm01(pcl.T)], axis=2)

    fig2, ax2 = plt.subplots(1, 1, figsize=(7.0, 5.8))
    ax2.imshow(rgb, origin="lower",
               extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
               interpolation="nearest", aspect="equal")
    ax2.set_xlabel("x (Å)"); ax2.set_ylabel("y (Å)")
    ax2.set_title(f"RGB ({side}): R=TFSI, G=Li, B=PCL")
    fig2.savefig(out_dir / f"rgb_{side}.png", dpi=300)
    plt.close(fig2)

make_overlay("bottom")
make_overlay("top")

# ─────────────────────────────────────────────────────────────────────────────
# README
# ─────────────────────────────────────────────────────────────────────────────

with open(out_dir / "README.txt", "w") as fh:
    fh.write(
        "Multi-species surface-projection maps\n"
        f"- Species: {list(species_types.keys())}\n"
        f"- Slabs: Bottom [zmin_ref - {z1_bottom} Å, zmin_ref], "
        f"Top [zmax_ref, zmax_ref + {z2_top} Å]\n"
        f"- Tolerance: mode={tolerance_mode}, tol={surface_tolerance} Å\n"
        f"- Averaging: {averaging_mode}\n"
        "- CSVs (long-form): areal_*_atoms_per_A2.csv and volumetric_*_atoms_per_A3.csv per species & side\n"
        "- Figures: one areal-density PNG per species with centered shared colorbar\n"
        "- Overlays: overlay_bottom.png, overlay_top.png (Li filled; TFSI/PCL contours)\n"
        "- RGB composites: rgb_bottom.png, rgb_top.png (R=TFSI, G=Li, B=PCL)\n"
    )
