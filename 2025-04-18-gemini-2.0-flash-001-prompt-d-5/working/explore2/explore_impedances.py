# Explore the electrode impedances
import pynwb
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e31801824/download/"  # Original URL

# Open the file using h5py directly
with h5py.File(url, 'r') as h5_file:
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
    nwb = io.read()

    # Get the electrode data
    electrodes = nwb.electrodes.to_dataframe()

    # Plot the electrode impedances
    plt.figure(figsize=(10, 6))
    plt.hist(electrodes['imp'], bins=20)
    plt.xlabel('Electrode Impedance (Ohms)')
    plt.ylabel('Number of Electrodes')
    plt.title('Distribution of Electrode Impedances')
    plt.savefig('explore2/electrode_impedances.png')
    plt.close()