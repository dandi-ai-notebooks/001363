"""
This script explores the NWB file located at the given URL. It loads specific components of the dataset to generate informative plots and text output for further analysis. The focus is on creating plots that display aspects of the ElectricalSeries data and trial information without overwhelming the memory.

The script:
1. Loads the dataset using the provided URL.
2. Generates plots for the ElectricalSeries data, examining sections of the data to avoid memory overuse.
3. Visualizes the timing intervals of the trials.
"""

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import pandas as pd

# Use seaborn for better aesthetic plots
import seaborn as sns
sns.set_theme()

url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access ElectricalSeries data
electrical_series_data = nwb.acquisition["ElectricalSeries"].data
sampling_rate = nwb.acquisition["ElectricalSeries"].rate

# Plotting a small segment of the ElectricalSeries data
start_idx = 0
end_idx = 1000  # View first 1000 samples from channel 0
data_segment = electrical_series_data[start_idx:end_idx, 0]

plt.figure(figsize=(10, 4))
plt.plot(data_segment)
plt.title("Electrical Series Data Segment: Channel 0")
plt.xlabel("Sample Index")
plt.ylabel("Voltage (V)")
plt.savefig('explore/electrical_series_data_segment.png')
plt.close()

# Accessing trial information
trial_data = nwb.trials.to_dataframe()
start_times = trial_data['start_time']
end_times = trial_data['stop_time']

# Plot trial durations
durations = end_times - start_times
plt.figure(figsize=(10, 4))
plt.hist(durations, bins=30)
plt.title("Histogram of Trial Durations")
plt.xlabel("Duration (s)")
plt.ylabel("Frequency")
plt.savefig('explore/trial_duration_histogram.png')
plt.close()

io.close()
h5_file.close()