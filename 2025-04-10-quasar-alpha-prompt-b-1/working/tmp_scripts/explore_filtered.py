# This script streams the NWB file.
# It extracts a 1-second segment from channel 0, applies a 300-3000 Hz bandpass filter,
# and plots the filtered trace. It also includes a zoom-in view.
# The plots are saved as png files in tmp_scripts/.

import remfile
import h5py
import pynwb
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

nwb_url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"
file = remfile.File(nwb_url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwb = io.read()

data = nwb.acquisition["ElectricalSeries"].data
rate = nwb.acquisition["ElectricalSeries"].rate
duration_seconds = 1
num_samples = int(rate * duration_seconds)

snippet = data[:num_samples, 0][:]
times = np.arange(num_samples) / rate

# Apply bandpass filter (300-3000 Hz)
filtered = bandpass_filter(snippet, 300, 3000, rate)

# Plot full 1-second filtered snippet
plt.figure(figsize=(10, 4))
plt.plot(times, filtered)
plt.title("Filtered snippet (300-3000 Hz, 1 sec, channel 0)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (filtered, V)")
plt.tight_layout()
plt.savefig("tmp_scripts/filtered_snippet.png")
plt.close()
print("Saved filtered_snippet.png")

# Zoom-in on 50 ms segment
start_sec = 0.4  # center zoom around the middle near 0.4 sec
zoom_duration = 0.05  # 50 ms
start_idx = int(start_sec * rate)
end_idx = int((start_sec + zoom_duration) * rate)

plt.figure(figsize=(10, 4))
plt.plot(times[start_idx:end_idx], filtered[start_idx:end_idx])
plt.title("Filtered snippet zoom-in (50 ms window, channel 0)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (filtered, V)")
plt.tight_layout()
plt.savefig("tmp_scripts/filtered_zoom.png")
plt.close()
print("Saved filtered_zoom.png")