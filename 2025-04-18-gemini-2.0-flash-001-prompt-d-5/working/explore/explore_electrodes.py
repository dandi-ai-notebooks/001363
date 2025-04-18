# Explore the electrode data and plot electrode locations
import pynwb
import h5py
import remfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the electrode data
electrodes = nwb.electrodes.to_dataframe()

# Print some info
print(electrodes.head())
print(electrodes.describe())

# Plot the electrode locations
plt.figure(figsize=(8, 6))
plt.scatter(electrodes['x'], electrodes['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Electrode Locations')
plt.savefig('explore/electrode_locations.png')  # Save the plot to a file
plt.close()