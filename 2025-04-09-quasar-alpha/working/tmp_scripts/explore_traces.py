# This script loads a small segment from an NWB file in Dandiset 001363
# It plots raw extracellular voltage traces for the first 2 seconds and all 32 channels
# The goal is to verify data access, visualize data appearance, and inspect metadata

import matplotlib.pyplot as plt
import numpy as np
import pynwb
import h5py
import remfile

# Load NWB file (streaming over the web)
url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwb = io.read()

es = nwb.acquisition["ElectricalSeries"]

rate = es.rate  # Sampling rate ~24414 Hz
duration_seconds = 2
num_samples = int(duration_seconds * rate)
num_channels = es.data.shape[1]

print(f"Number of samples requested: {num_samples}")
print(f"Number of channels: {num_channels}")

# Make sure not to request beyond available data length
num_samples = min(num_samples, es.data.shape[0])

# Load a manageable segment: the first 2 seconds from all channels
data_segment = es.data[:num_samples, :]

print(f"Loaded data segment shape: {data_segment.shape}")

time = np.arange(num_samples) / rate

plt.figure(figsize=(15, 10))
offset = 0
for ch in range(num_channels):
    plt.plot(time, data_segment[:, ch] + offset, label=f'Ch {ch}')
    offset += 0.5  # vertical offset for visualization

plt.xlabel('Time (s)')
plt.ylabel('Voltage + offset')
plt.title('Raw extracellular voltage traces: first 2 seconds, all channels')
plt.tight_layout()
plt.savefig('tmp_scripts/raw_traces.png')  # Save plot instead of plt.show()
print("Saved plot to tmp_scripts/raw_traces.png")

io.close()