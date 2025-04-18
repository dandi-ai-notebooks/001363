# This script explores the electrical series data from the dataset

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get basic information about the electrical series
electrical_series = nwb.acquisition["ElectricalSeries"]
print(f"Sampling rate: {electrical_series.rate} Hz")
print(f"Number of electrodes: {electrical_series.data.shape[1]}")
print(f"Total time points: {electrical_series.data.shape[0]}")
print(f"Total duration: {electrical_series.data.shape[0] / electrical_series.rate:.2f} seconds")
print(f"Unit: {electrical_series.unit}")

# Get trial information
trials_df = nwb.trials.to_dataframe()

# Extract a subset of data from the first trial
first_trial = trials_df.iloc[0]
start_time = first_trial['start_time']
stop_time = first_trial['stop_time']
print(f"\nAnalyzing first trial: Start time = {start_time:.2f}s, Stop time = {stop_time:.2f}s")

# Convert time to indices
start_idx = int(start_time * electrical_series.rate)
stop_idx = int(stop_time * electrical_series.rate)
print(f"Start index: {start_idx}, Stop index: {stop_idx}")

# Extract data for the first trial (and add some padding before)
padding = int(0.5 * electrical_series.rate)  # 0.5 seconds padding
start_with_padding = max(0, start_idx - padding)
trial_data = electrical_series.data[start_with_padding:stop_idx, :]
trial_time = np.arange(trial_data.shape[0]) / electrical_series.rate - (padding / electrical_series.rate)
print(f"Extracted data shape: {trial_data.shape}")

# Plot raw voltage traces for a few channels
plt.figure(figsize=(14, 8))
channels_to_plot = [0, 5, 10, 15, 20, 25, 30]  # Select a few channels to plot
for i, channel in enumerate(channels_to_plot):
    # Offset each channel for clarity
    offset = i * 0.0002
    plt.plot(trial_time, trial_data[:, channel] + offset, label=f"Channel {channel}")

plt.axvline(x=0, color='r', linestyle='--', label='Stimulation Start')
plt.axvline(x=1.5, color='r', linestyle='--', label='Stimulation End')
plt.xlabel('Time (s) relative to stimulation onset')
plt.ylabel('Voltage (V) - Channels stacked')
plt.title('Raw Voltage Traces from First Trial')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('explore/first_trial_raw_traces.png')

# Compute and plot spectrogram for one channel
channel = 0  # Select the first channel for spectral analysis
print(f"\nComputing spectrogram for channel {channel}...")

# For the same trial data
fs = electrical_series.rate
nperseg = int(fs * 0.2)  # 200 ms window
noverlap = nperseg // 2

# Compute spectrogram
f, t, Sxx = signal.spectrogram(trial_data[:, channel], fs=fs, nperseg=nperseg, noverlap=noverlap)

# Plot spectrogram
plt.figure(figsize=(12, 8))
plt.pcolormesh(t - (padding / electrical_series.rate), f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
plt.axvline(x=0, color='r', linestyle='--', label='Stimulation Start')
plt.axvline(x=1.5, color='r', linestyle='--', label='Stimulation End')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s) relative to stimulation onset')
plt.title(f'Spectrogram for Channel {channel}')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.ylim(0, 5000)  # Focus on the frequency range of interest
plt.savefig('explore/first_trial_spectrogram.png')

# Extract data for multiple trials and compute average response
print("\nComputing average response across 10 trials...")
num_trials_to_average = 10
all_trials_data = []

for i in range(min(num_trials_to_average, len(trials_df))):
    trial = trials_df.iloc[i]
    start_idx = int(trial['start_time'] * electrical_series.rate)
    stop_idx = int(trial['stop_time'] * electrical_series.rate)
    
    # Use a fixed window size based on the first trial
    window_size = stop_idx - start_idx
    trial_data = electrical_series.data[start_idx:start_idx+window_size, :]
    all_trials_data.append(trial_data)

# Stack trials
all_trials_stacked = np.stack(all_trials_data)
print(f"Stacked trials shape: {all_trials_stacked.shape}")

# Compute mean across trials
avg_response = np.mean(all_trials_stacked, axis=0)
print(f"Average response shape: {avg_response.shape}")

# Create time vector for the average response
avg_time = np.arange(avg_response.shape[0]) / electrical_series.rate

# Plot average response for a few channels
plt.figure(figsize=(14, 8))
for i, channel in enumerate(channels_to_plot):
    offset = i * 0.0001
    plt.plot(avg_time, avg_response[:, channel] + offset, label=f"Channel {channel}")

plt.xlabel('Time (s) from stimulation onset')
plt.ylabel('Voltage (V) - Channels stacked')
plt.title(f'Average Response Across {num_trials_to_average} Trials')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('explore/average_response.png')