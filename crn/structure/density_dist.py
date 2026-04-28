import os
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis import transformations as trans
import numpy as np
import pandas as pd
import datetime

#############################################
# Helper Functions (func_bins) ff_density2.py
#############################################

def get_values_for_bin(universe):
    # Print box dimensions and statistics for LLZO and PCL groups
    print("dimensions", universe.dimensions)
    llzo_group = universe.select_atoms("type 18 or type 19 or type 20", updating=True)
    position_values_1 = np.zeros(len(llzo_group))
    for i, atom in enumerate(llzo_group):
        position_values_1[i] = atom.position[2]
    print("min abs LLZO", min(abs(position_values_1)))
    print("max abs LLZO", max(abs(position_values_1)))
    print("min LLZO", min(position_values_1))
    print("max LLZO", max(position_values_1))
    
    # For PCL, select types 1-10
    pcl_group = universe.select_atoms("type 1 or type 2 or type 3 or type 4 or type 5 or type 6 or type 7 or type 8 or type 9 or type 10", updating=True)
    position_values_2 = np.zeros(len(pcl_group))
    for i, atom in enumerate(pcl_group):
        position_values_2[i] = atom.position[2]
    print("min abs PCL", min(abs(position_values_2)))
    print("max abs PCL", max(abs(position_values_2)))
    print("min PCL", min(position_values_2))
    print("max PCL", max(position_values_2))
    
def make_bin_sizes_based_on_nr_symmetric(nr_bins, universe):
    z_length = universe.dimensions[2]
    bin_width = z_length / nr_bins
    min_max_values = np.arange(-0.5 * nr_bins * bin_width, 0.5 * nr_bins * bin_width + bin_width, bin_width)
    list_of_bin_sizes = np.zeros((nr_bins, 2))
    for i in range(nr_bins):
        list_of_bin_sizes[i, 0] = min_max_values[i]
        list_of_bin_sizes[i, 1] = min_max_values[i+1]
    return list_of_bin_sizes

def get_bin_volumes(list_of_bin_sizes, universe):
    nr_bins = len(list_of_bin_sizes)
    x_max = universe.dimensions[0]
    y_max = universe.dimensions[1]
    bin_volumes = np.zeros(nr_bins)
    for bin_nr in range(nr_bins):
        z_min = list_of_bin_sizes[bin_nr, 0]
        z_max = list_of_bin_sizes[bin_nr, 1]
        bin_volumes[bin_nr] = x_max * y_max * (z_max - z_min)
    return bin_volumes

#############################################
# Main Analysis Script with Updated Selections
#############################################

# Load the Universe – update file paths as needed.
u = mda.Universe("combine_system.dat", "position.lammpstrj",
                 topology_format="DATA", format="LAMMPSDUMP", dt=20)
print("Total timesteps in trajectory:", len(u.trajectory))

# Define trajectory segment parameters
first_step = len(u.trajectory) - 10000
last_step  = len(u.trajectory)
timestep_spacing = 1

# Create results directory if it does not exist.
results_path = "results2"
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Center the system based on LLZO atoms
llzo_group = u.select_atoms("type 18 or type 19 or type 20", updating=True)
shift_factor = -u.dimensions[2] * 0.5
workflow = [trans.center_in_box(llzo_group, center='geometry'),
            trans.wrap(u.atoms),
            trans.translate([0, 0, shift_factor])]
u.trajectory.add_transformations(*workflow)

# Print diagnostic info for box and PCL
get_values_for_bin(u)

# Define bins and calculate bin volumes
nr_bins = 520
bin_sizes = make_bin_sizes_based_on_nr_symmetric(nr_bins, u)
bin_volumes = get_bin_volumes(bin_sizes, u)

# Updated atom selections:
all_litfsi_lithiums = u.select_atoms("type 16", updating=True)
all_llzo_lithiums   = u.select_atoms("type 17", updating=True)
all_OH              = u.select_atoms("type 1", updating=True)
all_COH             = u.select_atoms("type 2", updating=True)
all_ODB             = u.select_atoms("type 3", updating=True)
all_OE              = u.select_atoms("type 6", updating=True)
all_tfsi            = u.select_atoms("type 11 or type 12 or type 13 or type 14 or type 15", updating=True)
all_la              = u.select_atoms("type 18", updating=True)
all_zr              = u.select_atoms("type 19", updating=True)
all_llzo_o          = u.select_atoms("type 20", updating=True)
all_CH3             = u.select_atoms("type 7", updating=True)
all_CDB             = u.select_atoms("type 5", updating=True)
all_CH2             = u.select_atoms("type 4", updating=True)
all_C_TFSI          = u.select_atoms("type 11", updating=True)
all_F_TFSI          = u.select_atoms("type 12", updating=True)
all_S_TFSI          = u.select_atoms("type 13", updating=True)
all_N_TFSI          = u.select_atoms("type 14", updating=True)
all_O_TFSI          = u.select_atoms("type 15", updating=True)

# Define the timesteps and initialize array for counts
timesteps = np.arange(first_step, last_step, timestep_spacing)
nr_timesteps = len(timesteps)
nr_of_selections = 18  # total selections from group 0 to 17
N_in_slice = np.zeros((nr_bins, nr_timesteps, nr_of_selections))

# Loop over selected timesteps and bins; count atoms in each bin for each selection.
for cntr, ts in enumerate(u.trajectory[first_step:last_step:timestep_spacing], start=0):
    for bin_nr in range(nr_bins):
        z_min = bin_sizes[bin_nr, 0]
        z_max = bin_sizes[bin_nr, 1]
        slice_litfsi_lithiums = all_litfsi_lithiums.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        slice_llzo_lithiums   = all_llzo_lithiums.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        slice_OH              = all_OH.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        slice_COH             = all_COH.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        slice_ODB             = all_ODB.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        slice_OE              = all_OE.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        slice_tfsi            = all_tfsi.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        slice_la              = all_la.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        slice_zr              = all_zr.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        slice_llzo_o          = all_llzo_o.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        slice_CH3             = all_CH3.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        slice_CDB             = all_CDB.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        slice_CH2             = all_CH2.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        slice_C_TFSI          = all_C_TFSI.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        slice_F_TFSI          = all_F_TFSI.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        slice_S_TFSI          = all_S_TFSI.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        slice_N_TFSI          = all_N_TFSI.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        slice_O_TFSI          = all_O_TFSI.select_atoms("prop z > {} and prop z <= {}".format(z_min, z_max), updating=True)
        
        N_in_slice[bin_nr, cntr, 0]  = len(slice_litfsi_lithiums)
        N_in_slice[bin_nr, cntr, 1]  = len(slice_llzo_lithiums)
        N_in_slice[bin_nr, cntr, 2]  = len(slice_OH)
        N_in_slice[bin_nr, cntr, 3]  = len(slice_COH)
        N_in_slice[bin_nr, cntr, 4]  = len(slice_ODB)
        N_in_slice[bin_nr, cntr, 5]  = len(slice_OE)
        N_in_slice[bin_nr, cntr, 6]  = len(slice_tfsi)
        N_in_slice[bin_nr, cntr, 7]  = len(slice_la)
        N_in_slice[bin_nr, cntr, 8]  = len(slice_zr)
        N_in_slice[bin_nr, cntr, 9]  = len(slice_llzo_o)
        N_in_slice[bin_nr, cntr, 10] = len(slice_CH3)
        N_in_slice[bin_nr, cntr, 11] = len(slice_CDB)
        N_in_slice[bin_nr, cntr, 12] = len(slice_CH2)
        N_in_slice[bin_nr, cntr, 13] = len(slice_C_TFSI)
        N_in_slice[bin_nr, cntr, 14] = len(slice_F_TFSI)
        N_in_slice[bin_nr, cntr, 15] = len(slice_S_TFSI)
        N_in_slice[bin_nr, cntr, 16] = len(slice_N_TFSI)
        N_in_slice[bin_nr, cntr, 17] = len(slice_O_TFSI)

# Calculate mean number and density per bin
mean_number = np.zeros((nr_bins, nr_of_selections))
mean_dens   = np.zeros((nr_bins, nr_of_selections))
for bin_nr in range(nr_bins):
    for sel in range(nr_of_selections):
        mean_number[bin_nr, sel] = np.mean(N_in_slice[bin_nr, :, sel])
for col in range(nr_of_selections):
    mean_dens[:, col] = mean_number[:, col] / bin_volumes

# Create a DataFrame with updated column names.
df_mean_dens = pd.DataFrame(mean_dens, columns=[
    "LiTFSI Li", "LLZO Li", "OH", "COH", "ODB", "OE", "TFSI",
    "La", "Zr", "LLZO O", "CH3", "CDB", "CH2", "C_TFSI", "F_TFSI",
    "S_TFSI", "N_TFSI", "O_TFSI"
])
bin_widths = bin_sizes[:, 1] - bin_sizes[:, 0]
z_values = bin_sizes[:, 0] + bin_widths / 2
df_mean_dens.insert(0, "z_value", z_values)

# Save the density data as CSV
csv_filename = "{}/mean_density_{}bins_{}to{}_in{}.csv".format(results_path, nr_bins, first_step, last_step, timestep_spacing)
df_mean_dens.to_csv(csv_filename, index=False)
print("Mean density data saved to:", csv_filename)

# Append information to a README file.
today = datetime.datetime.now()
with open("{}/README.txt".format(results_path), "a") as f:
    f.write("{} \n".format(str(today)))
    f.write("Mean density calculated using the updated selection scheme.\n")
    f.write("FILE: {}\n".format(csv_filename))
    f.write("CHOSEN VALUES\n")
    f.write("nr_bins = {}\n".format(nr_bins))
    f.write("bin_volume (first bin) = {}\n".format(bin_volumes[0]))
    f.write("timestep_spacing = {}\n".format(timestep_spacing))
    f.write("Based on steps {} to {}\n\n\n".format(first_step, last_step))

# Plot the density profiles
plt.figure(figsize=(10, 6))
for col in df_mean_dens.columns[1:]:
    plt.plot(df_mean_dens["z_value"], df_mean_dens[col], label=col)
plt.xlabel("z (Angstrom)")
plt.ylabel("Mean Density (atoms per volume)")
plt.title("Mean Density Profile along z-axis")
plt.legend(loc='best', fontsize='small')
plt.tight_layout()
plot_filename = "{}/density_profile.png".format(results_path)
plt.savefig(plot_filename)
print("Density profile plot saved to:", plot_filename)
#plt.show()
