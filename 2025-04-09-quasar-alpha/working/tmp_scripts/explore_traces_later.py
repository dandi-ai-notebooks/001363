# This script loads a 2-second segment starting at 100 seconds into the NWB file
# It plots raw extracellular voltage traces for all channels
# The goal is to see if other parts of the recording contain more informative signals

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

rate = es.rate
start_time = 100  # seconds
duration_seconds = 2
start_sample = int(start_time * rate)
num_samples = int(duration_seconds * rate)
num_channels = es.data.shape[1]

# Clamp to max samples
if start_sample + num_samples > es.data.shape[0]:
    num_samples = es.data.shape[0] - start_sample

print(f"Starting sample: {start_sample}")
print(f"Number of samples: {num_samples}")
print(f"Number of channels: {num_channels}")

data_segment = es.data[start_sample:start_sample+num_samples, :]

print(f"Loaded data segment shape: {data_segment.shape}")

time = np.arange(num_samples) / rate

plt.figure(figsize=(15, 10))
offset = 0
for ch in range(num_channels):
    plt.plot(time, data_segment[:, ch] + offset, label=f'Ch {ch}')
    offset += 0.5  # vertical offset for visualization

plt.xlabel('Time (s)')
plt.ylabel('Voltage + offset')
plt.title('Raw extracellular voltage traces: 100s-102s, all channels')
plt.tight_layout()
plt.savefig('tmp_scripts/raw_traces_later.png')  # Save plot
print("Saved plot to tmp_scripts/raw_traces_later.png")

io.close()