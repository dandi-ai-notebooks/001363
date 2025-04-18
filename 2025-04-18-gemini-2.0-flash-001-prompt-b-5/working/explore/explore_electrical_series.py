# explore/explore_electrical_series.py
# This script explores the ElectricalSeries data in the NWB file.
# It loads the NWB file, access the ElectricalSeries data,
# plots a small subset of the data to a PNG file, and prints some basic information.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Access the ElectricalSeries data
electrical_series = nwb.acquisition["ElectricalSeries"]
data = electrical_series.data
rate = electrical_series.rate

# Print some basic information about the data
print(f"Data shape: {data.shape}")
print(f"Data dtype: {data.dtype}")
print(f"Sampling rate: {rate}")

# Plot a small subset of the data (first 1000 samples from the first channel)
num_samples = 1000
channel_index = 0
time = np.arange(num_samples) / rate
subset_data = data[:num_samples, channel_index]

plt.figure(figsize=(10, 5))
plt.plot(time, subset_data)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title(f"ElectricalSeries data (first {num_samples} samples, channel {channel_index})")
plt.savefig("explore/electrical_series_subset.png")

print("Plot saved to explore/electrical_series_subset.png")