# This script compares data from two different subjects in the Dandiset
# to see if the responses to ultrasound stimulation are consistent

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# File 1 - Subject BH589 (from our previous analysis)
url1 = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"

# File 2 - Subject BH625 (choosing a different subject)
url2 = "https://api.dandiarchive.org/api/assets/63982aca-c92f-4d87-86df-e44ace913043/download/"

# Function to load NWB files
def load_nwb(url):
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    return io.read(), io, h5_file, remote_file

# Load both files
print("Loading NWB file 1...")
nwb1, io1, h5_file1, remote_file1 = load_nwb(url1)

print("Loading NWB file 2...")
nwb2, io2, h5_file2, remote_file2 = load_nwb(url2)

# Print basic information about both files
print("\n=== File 1 Information ===")
print(f"Subject: {nwb1.subject.subject_id}")
print(f"Session ID: {nwb1.identifier}")
print(f"Number of Electrodes: {len(nwb1.electrodes.id[:])}")
print(f"Number of Trials: {len(nwb1.trials.id[:])}")

print("\n=== File 2 Information ===")
print(f"Subject: {nwb2.subject.subject_id}")
print(f"Session ID: {nwb2.identifier}")
print(f"Number of Electrodes: {len(nwb2.electrodes.id[:])}")
print(f"Number of Trials: {len(nwb2.trials.id[:])}")

# Compare trial structure
trials_df1 = nwb1.trials.to_dataframe()
trials_df2 = nwb2.trials.to_dataframe()

print("\n=== Trial Duration Statistics ===")
print("File 1:")
duration1 = trials_df1['stop_time'] - trials_df1['start_time']
print(f"Mean: {duration1.mean():.4f} seconds")
print(f"Std Dev: {duration1.std():.4f} seconds")

print("File 2:")
duration2 = trials_df2['stop_time'] - trials_df2['start_time']
print(f"Mean: {duration2.mean():.4f} seconds")
print(f"Std Dev: {duration2.std():.4f} seconds")

# Compare trial intervals
print("\n=== Trial Interval Statistics ===")
print("File 1:")
intervals1 = trials_df1['start_time'].shift(-1) - trials_df1['start_time']
print(f"Mean: {intervals1.mean():.4f} seconds")
print(f"Std Dev: {intervals1.std():.4f} seconds")

print("File 2:")
intervals2 = trials_df2['start_time'].shift(-1) - trials_df2['start_time']
print(f"Mean: {intervals2.mean():.4f} seconds")
print(f"Std Dev: {intervals2.std():.4f} seconds")

# Compare electrode information
electrodes_df1 = nwb1.electrodes.to_dataframe()
electrodes_df2 = nwb2.electrodes.to_dataframe()

print("\n=== Electrode Information ===")
print("File 1:")
print(electrodes_df1.head())

print("\nFile 2:")
print(electrodes_df2.head())

# Function to analyze spectral content around a stimulus
def analyze_trial_spectral(nwb, trial_num=5, pre_window=1.0, post_window=1.0, electrode_idx=16):
    trials_df = nwb.trials.to_dataframe()
    trial_start = trials_df.iloc[trial_num-1]['start_time']
    sampling_rate = nwb.acquisition["ElectricalSeries"].rate
    
    # Calculate indices
    pre_start_idx = max(0, int((trial_start - pre_window) * sampling_rate))
    trial_start_idx = int(trial_start * sampling_rate)
    post_end_idx = min(int((trial_start + post_window) * sampling_rate), 
                       nwb.acquisition["ElectricalSeries"].data.shape[0])
    
    # Get data
    pre_data = nwb.acquisition["ElectricalSeries"].data[pre_start_idx:trial_start_idx, electrode_idx]
    post_data = nwb.acquisition["ElectricalSeries"].data[trial_start_idx:post_end_idx, electrode_idx]
    
    # Calculate PSD
    nperseg = min(1024, len(pre_data), len(post_data))
    f_pre, Pxx_pre = signal.welch(pre_data, fs=sampling_rate, nperseg=nperseg)
    f_post, Pxx_post = signal.welch(post_data, fs=sampling_rate, nperseg=nperseg)
    
    return f_pre, Pxx_pre, f_post, Pxx_post

# Compare spectral content for both files
print("\n=== Comparing Spectral Content ===")
f_pre1, Pxx_pre1, f_post1, Pxx_post1 = analyze_trial_spectral(nwb1)
f_pre2, Pxx_pre2, f_post2, Pxx_post2 = analyze_trial_spectral(nwb2)

# Plot spectral comparison
plt.figure(figsize=(12, 8))

# Limit frequency axis to focus on relevant neural frequencies
freq_mask1 = f_pre1 < 500
freq_mask2 = f_pre2 < 500

plt.subplot(2, 1, 1)
plt.semilogy(f_pre1[freq_mask1], Pxx_pre1[freq_mask1], 'b--', label='Pre-stimulus')
plt.semilogy(f_post1[freq_mask1], Pxx_post1[freq_mask1], 'b-', label='Post-stimulus')
plt.title(f'Subject {nwb1.subject.subject_id} - Power Spectral Density')
plt.ylabel('Power/Frequency (V^2/Hz)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
plt.semilogy(f_pre2[freq_mask2], Pxx_pre2[freq_mask2], 'r--', label='Pre-stimulus')
plt.semilogy(f_post2[freq_mask2], Pxx_post2[freq_mask2], 'r-', label='Post-stimulus')
plt.title(f'Subject {nwb2.subject.subject_id} - Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (V^2/Hz)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('subject_comparison.png')
plt.close()

# Calculate percent change in band power for both subjects
bands = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 100),
    'High': (100, 300)
}

# Function to get average power in a frequency band
def get_band_power(frequencies, power, low_freq, high_freq):
    idx = np.logical_and(frequencies >= low_freq, frequencies <= high_freq)
    if np.any(idx):
        return np.mean(power[idx])
    else:
        return np.nan

# Calculate band power changes
band_names = list(bands.keys())
changes1 = np.zeros(len(band_names))
changes2 = np.zeros(len(band_names))

for i, (band_name, (low_freq, high_freq)) in enumerate(bands.items()):
    pre_power1 = get_band_power(f_pre1, Pxx_pre1, low_freq, high_freq)
    post_power1 = get_band_power(f_post1, Pxx_post1, low_freq, high_freq)
    
    pre_power2 = get_band_power(f_pre2, Pxx_pre2, low_freq, high_freq)
    post_power2 = get_band_power(f_post2, Pxx_post2, low_freq, high_freq)
    
    if not np.isnan(pre_power1) and pre_power1 > 0:
        changes1[i] = 100 * (post_power1 - pre_power1) / pre_power1
    
    if not np.isnan(pre_power2) and pre_power2 > 0:
        changes2[i] = 100 * (post_power2 - pre_power2) / pre_power2

# Create band power comparison plot
plt.figure(figsize=(12, 6))
x = np.arange(len(band_names))
width = 0.35

plt.bar(x - width/2, changes1, width, label=f'Subject {nwb1.subject.subject_id}', color='b', alpha=0.7)
plt.bar(x + width/2, changes2, width, label=f'Subject {nwb2.subject.subject_id}', color='r', alpha=0.7)

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('Frequency Band')
plt.ylabel('Percent Change in Power (%)')
plt.title('Percent Change in Band Power After Stimulation - Subject Comparison')
plt.xticks(x, band_names)
plt.legend()
plt.tight_layout()
plt.savefig('subject_band_comparison.png')
plt.close()

# Clean up
io1.close()
io2.close()

print("\nCompleted subject comparison analysis. Generated plots:")
print("- subject_comparison.png: Compares power spectral density between subjects")
print("- subject_band_comparison.png: Compares percent change in band power between subjects")