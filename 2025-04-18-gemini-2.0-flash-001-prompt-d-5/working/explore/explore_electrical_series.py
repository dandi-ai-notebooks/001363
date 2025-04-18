# Explore the ElectricalSeries data and plot a segment of the data
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/3d3eafca-bd41-4d52-8938-2c1f5a459a3e/download/"  # New URL
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the ElectricalSeries data
electrical_series = nwb.acquisition["ElectricalSeries"]
data = electrical_series.data
rate = electrical_series.rate

# Select a small amount of data from the first few channels
num_channels = 4
num_timepoints = 1000
start_time = 100  # seconds - changed from 10 to 100 - changed from 0 to 10

start_index = int(start_time * rate)
end_index = start_index + num_timepoints

# Load the data
data_subset = data[start_index:end_index, :num_channels]

# Create a time axis
time = np.arange(num_timepoints) / rate

# Plot the data
plt.figure(figsize=(12, 6))
for i in range(num_channels):
    plt.plot(time, data_subset[:, i] + i * 100, label=f'Channel {i}')  # Add offset for each channel
plt.xlabel('Time (s)')
plt.ylabel('Voltage (uV) + offset')
plt.title('Raw Electrophysiology Data')
plt.legend()
plt.savefig('explore2/electrical_series.png')  # Save the plot to a file - updated path
plt.close()