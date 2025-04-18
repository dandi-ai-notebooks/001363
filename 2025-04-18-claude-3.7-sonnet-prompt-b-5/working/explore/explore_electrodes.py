# This script explores the electrodes information from the dataset

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get information about the subject and session
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject Age: {nwb.subject.age}")
print(f"Subject Species: {nwb.subject.species}")
print(f"Session Description: {nwb.session_description}")
print(f"Session Identifier: {nwb.identifier}")

# Get information about electrodes
electrodes_df = nwb.electrodes.to_dataframe()
print("\nElectrode Information:")
print(f"Number of electrodes: {len(electrodes_df)}")
print(electrodes_df.head())

# Display electrode location information
print("\nElectrode Locations:")
print(electrodes_df['location'].value_counts())

# Display electrode group information
print("\nElectrode Group Information:")
for group_name, group in nwb.electrode_groups.items():
    print(f"Group: {group_name}")
    print(f"  Description: {group.description}")
    print(f"  Location: {group.location}")
    print(f"  Device: {group.device.description}")

# Plot electrode positions if x, y coordinates are available
plt.figure(figsize=(10, 8))
plt.scatter(electrodes_df['x'], electrodes_df['y'], c=range(len(electrodes_df)), cmap='viridis', s=100)
plt.colorbar(label='Electrode index')
plt.title('Electrode Positions')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.grid(True)
plt.savefig('explore/electrode_positions.png')