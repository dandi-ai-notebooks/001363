# This script explores the spectral properties of the neural activity in the NWB file
# We'll examine changes in frequency content before and after stimulation

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

# Get trials data
trials_df = nwb.trials.to_dataframe()
print(f"Number of Trials: {len(trials_df)}")

# Get sampling rate
sampling_rate = nwb.acquisition["ElectricalSeries"].rate
print(f"Sampling Rate: {sampling_rate} Hz")

# Select a single trial to analyze
trial_number = 5
trial_start = trials_df.loc[trial_number, 'start_time']
trial_end = trials_df.loc[trial_number, 'stop_time']
trial_duration = trial_end - trial_start
print(f"\nAnalyzing Trial {trial_number}:")
print(f"Start time: {trial_start:.4f} seconds")
print(f"End time: {trial_end:.4f} seconds")
print(f"Duration: {trial_duration:.4f} seconds")

# Define time windows for analysis (1 second before and 1 second after stimulation)
pre_stim_start = trial_start - 1.0
post_stim_start = trial_start

# Calculate sample indices
pre_stim_start_idx = int(pre_stim_start * sampling_rate)
pre_stim_end_idx = int(trial_start * sampling_rate)
post_stim_start_idx = int(post_stim_start * sampling_rate)
post_stim_end_idx = int((post_stim_start + 1.0) * sampling_rate)

print(f"Pre-stim window: {pre_stim_start:.4f} to {trial_start:.4f} seconds")
print(f"Post-stim window: {post_stim_start:.4f} to {post_stim_start + 1.0:.4f} seconds")

# Select some electrodes to analyze
selected_electrodes = [0, 8, 16, 24, 31]

# Extract data for pre and post stimulation
pre_stim_data = nwb.acquisition["ElectricalSeries"].data[pre_stim_start_idx:pre_stim_end_idx, selected_electrodes]
post_stim_data = nwb.acquisition["ElectricalSeries"].data[post_stim_start_idx:post_stim_end_idx, selected_electrodes]

# Calculate spectrogram for one electrode
electrode_idx = 2  # Use electrode 16 (index 2 in selected_electrodes)
electrode_number = selected_electrodes[electrode_idx]

# Parameters for spectrogram
nperseg = 1024  # Length of each segment
noverlap = nperseg // 2  # Overlap between segments

# Calculate spectrograms
f_pre, t_pre, Sxx_pre = signal.spectrogram(
    pre_stim_data[:, electrode_idx], 
    fs=sampling_rate, 
    nperseg=nperseg, 
    noverlap=noverlap
)

f_post, t_post, Sxx_post = signal.spectrogram(
    post_stim_data[:, electrode_idx], 
    fs=sampling_rate, 
    nperseg=nperseg, 
    noverlap=noverlap
)

# Adjust time to reflect absolute timing
t_pre = t_pre + pre_stim_start
t_post = t_post + post_stim_start

# Plot spectrograms
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Limit frequency axis to focus on relevant neural frequencies (< 500 Hz)
freq_mask = f_pre < 500

# Plot pre-stimulus spectrogram
pcm1 = ax1.pcolormesh(t_pre, f_pre[freq_mask], 10 * np.log10(Sxx_pre[freq_mask, :]), 
                     shading='gouraud', cmap='viridis')
ax1.set_title(f'Pre-Stimulus Spectrogram - Electrode {electrode_number}')
ax1.set_ylabel('Frequency (Hz)')
ax1.set_xlabel('Time (s)')
plt.colorbar(pcm1, ax=ax1, label='Power/Frequency (dB/Hz)')

# Plot post-stimulus spectrogram
pcm2 = ax2.pcolormesh(t_post, f_post[freq_mask], 10 * np.log10(Sxx_post[freq_mask, :]), 
                     shading='gouraud', cmap='viridis')
ax2.set_title(f'Post-Stimulus Spectrogram - Electrode {electrode_number}')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_xlabel('Time (s)')
ax2.axvline(x=trial_start, color='r', linestyle='--', label='Stim Onset')
plt.colorbar(pcm2, ax=ax2, label='Power/Frequency (dB/Hz)')

plt.tight_layout()
plt.savefig('spectrograms.png')
plt.close()

# Calculate and plot power spectral density for pre and post stimulus periods
plt.figure(figsize=(12, 6))

# Use Welch's method to calculate PSD
f_pre_psd, Pxx_pre = signal.welch(pre_stim_data[:, electrode_idx], fs=sampling_rate, nperseg=nperseg)
f_post_psd, Pxx_post = signal.welch(post_stim_data[:, electrode_idx], fs=sampling_rate, nperseg=nperseg)

# Limit frequency axis to focus on relevant neural frequencies
freq_mask_psd = f_pre_psd < 500

plt.semilogy(f_pre_psd[freq_mask_psd], Pxx_pre[freq_mask_psd], 'b', label='Pre-stimulus')
plt.semilogy(f_post_psd[freq_mask_psd], Pxx_post[freq_mask_psd], 'r', label='Post-stimulus')
plt.title(f'Power Spectral Density - Electrode {electrode_number}')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (V^2/Hz)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('power_spectral_density.png')
plt.close()

# Calculate average power in different frequency bands before and after stimulation
# Frequency bands of interest: Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-100 Hz)
bands = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 100)
}

# Function to get average power in a frequency band
def get_band_power(frequencies, power, low_freq, high_freq):
    idx = np.logical_and(frequencies >= low_freq, frequencies <= high_freq)
    return np.mean(power[idx])

# Calculate band power for multiple electrodes
band_powers_pre = np.zeros((len(selected_electrodes), len(bands)))
band_powers_post = np.zeros((len(selected_electrodes), len(bands)))

for i, electrode_idx in enumerate(range(len(selected_electrodes))):
    # Calculate PSD for this electrode
    f_pre_psd, Pxx_pre = signal.welch(pre_stim_data[:, electrode_idx], fs=sampling_rate, nperseg=nperseg)
    f_post_psd, Pxx_post = signal.welch(post_stim_data[:, electrode_idx], fs=sampling_rate, nperseg=nperseg)
    
    # Calculate band power
    for j, (band_name, (low_freq, high_freq)) in enumerate(bands.items()):
        band_powers_pre[i, j] = get_band_power(f_pre_psd, Pxx_pre, low_freq, high_freq)
        band_powers_post[i, j] = get_band_power(f_post_psd, Pxx_post, low_freq, high_freq)

# Calculate percent change in band power
percent_change = 100 * (band_powers_post - band_powers_pre) / band_powers_pre

# Plot percent change in band power for each electrode
plt.figure(figsize=(12, 8))
band_names = list(bands.keys())
x = np.arange(len(band_names))
width = 0.15  # bar width
colors = ['blue', 'orange', 'green', 'red', 'purple']

for i, electrode_idx in enumerate(range(len(selected_electrodes))):
    plt.bar(x + i*width - 0.3, percent_change[i], width, label=f'Electrode {selected_electrodes[i]}', 
            color=colors[i], alpha=0.7)

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('Frequency Band')
plt.ylabel('Percent Change in Power (%)')
plt.title('Percent Change in Band Power After Stimulation')
plt.xticks(x, band_names)
plt.legend()
plt.tight_layout()
plt.savefig('band_power_change.png')
plt.close()

# Close the file
io.close()

print("\nCompleted spectral analysis. Generated plots:")
print("- spectrograms.png: Shows spectrograms before and after stimulation")
print("- power_spectral_density.png: Shows power spectral density before and after stimulation")
print("- band_power_change.png: Shows percent change in power in different frequency bands")