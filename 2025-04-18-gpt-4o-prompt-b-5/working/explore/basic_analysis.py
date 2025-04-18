"""
This script performs basic exploratory analysis on the NWB file by loading metadata and a subset of 'ElectricalSeries' data.
It generates a plot to visualize the voltage signal from an electrode over a short time duration.

# Load the NWB file and familiarize with its metadata.
# Generate a plot for a subsample of 'ElectricalSeries' data from all electrodes to examine signal properties.
"""

import pynwb
import h5py
import numpy as np
import matplotlib.pyplot as plt
import remfile

# URL for NWB file
url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the ElectricalSeries
electrical_series = nwb.acquisition['ElectricalSeries']

# Load a subset of data: first 2000 samples from the first electrode
data_subset = electrical_series.data[:2000, 0]

# Create a plot of the data
plt.figure(figsize=(10, 4))
plt.plot(data_subset)
plt.title('Voltage Signal of First Electrode: First 2000 Samples')
plt.xlabel('Sample Index')
plt.ylabel('Voltage (V)')
plt.savefig('explore/electrode_voltage_plot.png')

io.close()
h5_file.close()
remote_file.close()