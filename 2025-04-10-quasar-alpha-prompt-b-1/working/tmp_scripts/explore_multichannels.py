# This script extracts a 50 ms data window and plots filtered snippets from multiple channels
# to assess spiking activity variability across electrodes in the NWB file.
# It saves a multi-panel figure in tmp_scripts/multichannel_snippets.png

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

start_sec = 0.4  # e.g., middle of the recording
win_sec = 0.05  # 50 ms
start_idx = int(start_sec * rate)
end_idx = int((start_sec + win_sec) * rate)

channels_to_plot = 8  # 0-7

fig, axs = plt.subplots(channels_to_plot // 2, 2, figsize=(12, 10), sharex=True, sharey=True)
axs = axs.flatten()

times = np.arange(start_idx, end_idx) / rate

for ch in range(channels_to_plot):
    snippet = data[start_idx:end_idx, ch][:]
    filtered = bandpass_filter(snippet, 300, 3000, rate)
    axs[ch].plot(times, filtered)
    axs[ch].set_title(f"Channel {ch}")
    axs[ch].set_xlabel("Time (s)")
    axs[ch].set_ylabel("Voltage (filtered, V)")

plt.suptitle("Filtered 50 ms snippets from multiple channels")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("tmp_scripts/multichannel_snippets.png")
plt.close()
print("Saved multichannel_snippets.png")