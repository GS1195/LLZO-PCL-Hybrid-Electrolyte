import os
import numpy as np
import pandas as pd
import scipy.constants
import datetime

# Set simulation and temperature parameters.
simulation = "700K/80LiTFSI"
T = 700  # Temperature in Kelvin

# Number of bins is inferred from the CSV file name ("520bins")
# Set the density file name (using relative paths)
density_filename = "/mnt/d/puresystem/paper2/density/5isto1/700/results2/mean_density.csv".format(simulation)

# Read in the density data.
df_atom_dens = pd.read_csv(density_filename)
print("Density data loaded from:", density_filename)

# Create a new column "Li_all" as the sum of "LiTFSI Li" and "LLZO Li"
Li_all_dens = df_atom_dens["LiTFSI Li"] + df_atom_dens["LLZO Li"]
df_atom_dens.insert(4, "Li_all", Li_all_dens, allow_duplicates=True)

# Separate the z_value column from the density columns.
z_vals = df_atom_dens["z_value"]
density_columns = df_atom_dens.columns.drop("z_value")

# Calculate free energy using the relation:
#   F = - k_B * T * ln(rho)
# Only apply the logarithm to the density columns.
df_free_energy = -scipy.constants.k * T * np.log(df_atom_dens[density_columns])
# Reinsert the z_value column.
df_free_energy.insert(0, "z_value", z_vals)


# Define the output directory (relative path) and create it if it doesn't exist.
results_path = "/mnt/d/puresystem/paper2/density/5isto1/700/results2/free_energy".format(simulation)
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Save the free energy data as CSV.
free_energy_filename = "{}/free_energy.csv".format(results_path)
df_free_energy.to_csv(free_energy_filename, index=False)
print("Free energy data saved to:", free_energy_filename)

# Append calculation details to a README file in the free_energy folder.
today = datetime.datetime.now()
with open("{}/README.txt".format(results_path), "a") as f:
    f.write("{}\n".format(str(today)))
    f.write("Free energy profiles calculated using free_energy.ipynb\n")
    f.write("Input density file: {}\n".format(density_filename))
    f.write("CHOSEN VALUES:\n")
    f.write("Based on density file: mean_density_520bins_1to11_in1.csv\n")
    f.write("Simulation: {}\n".format(simulation))
    f.write("Temperature: {} K\n\n\n".format(T))
