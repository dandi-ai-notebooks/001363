# This script explores the neural activity in the NWB file
# We'll examine data from a few electrodes during a trial

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get electrode information
electrodes_df = nwb.electrodes.to_dataframe()
print("=== Electrode Information ===")
print(f"Number of Electrodes: {len(electrodes_df)}")
print(electrodes_df[['x', 'y', 'z', 'location', 'group_name']].head())

# Get trials data
trials_df = nwb.trials.to_dataframe()
print("\n=== Trials Information ===")
print(f"Number of Trials: {len(trials_df)}")
print(trials_df.head())

# Select data from a single trial
trial_number = 5  # Let's look at the 5th trial
trial_start = trials_df.loc[trial_number, 'start_time']
trial_end = trials_df.loc[trial_number, 'stop_time']
print(f"\nExamining Trial {trial_number}:")
print(f"Start time: {trial_start:.4f} seconds")
print(f"End time: {trial_end:.4f} seconds")

# Calculate sample indices for the trial
sampling_rate = nwb.acquisition["ElectricalSeries"].rate
start_idx = int(trial_start * sampling_rate)
end_idx = int(trial_end * sampling_rate)
print(f"Start index: {start_idx}, End index: {end_idx}")

# Add some buffer before and after the trial for context
buffer = int(0.5 * sampling_rate)  # 0.5 second buffer
start_idx_with_buffer = max(0, start_idx - buffer)
end_idx_with_buffer = min(nwb.acquisition["ElectricalSeries"].data.shape[0], end_idx + buffer)

# Extract data for selected time window
time_slice = slice(start_idx_with_buffer, end_idx_with_buffer)
data = nwb.acquisition["ElectricalSeries"].data[time_slice, :]

# Calculate time points
time_points = np.arange(data.shape[0]) / sampling_rate + (start_idx_with_buffer / sampling_rate)

# Extract data for plotting
# Select 5 electrodes to plot
selected_electrodes = [0, 8, 16, 24, 31]  # A selection across the electrode array

# Plot the neural data
plt.figure(figsize=(12, 10))

for i, electrode_idx in enumerate(selected_electrodes):
    # Get electrode name
    electrode_name = electrodes_df.loc[electrode_idx, 'group_name']
    
    # Plot data with offset for visualization
    plt.subplot(len(selected_electrodes), 1, i+1)
    plt.plot(time_points, data[:, electrode_idx])
    plt.axvline(x=trial_start, color='r', linestyle='--', label='Trial Start')
    plt.axvline(x=trial_end, color='g', linestyle='--', label='Trial End')
    plt.title(f'Electrode {electrode_idx} ({electrode_name})')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.grid(True, alpha=0.3)
    
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.savefig('electrode_activity_trial.png')
plt.close()

# Plot a smaller time window focusing on just after stimulation onset
# for better visualization of immediate responses
post_stim_window = 0.2  # 200 ms after stimulus onset
time_slice_post_stim = slice(
    start_idx, 
    min(start_idx + int(post_stim_window * sampling_rate), nwb.acquisition["ElectricalSeries"].data.shape[0])
)

# Extract data for the post-stim time window
post_stim_data = nwb.acquisition["ElectricalSeries"].data[time_slice_post_stim, :]
post_stim_time = np.arange(post_stim_data.shape[0]) / sampling_rate + trial_start

# Plot the immediate post-stimulus activity
plt.figure(figsize=(12, 10))

for i, electrode_idx in enumerate(selected_electrodes):
    # Get electrode name
    electrode_name = electrodes_df.loc[electrode_idx, 'group_name']
    
    # Plot data with offset for visualization
    plt.subplot(len(selected_electrodes), 1, i+1)
    plt.plot(post_stim_time, post_stim_data[:, electrode_idx])
    plt.axvline(x=trial_start, color='r', linestyle='--', label='Stim Onset')
    plt.title(f'Electrode {electrode_idx} ({electrode_name}) - Immediate Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.grid(True, alpha=0.3)
    
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.savefig('electrode_immediate_response.png')
plt.close()

# Calculate and plot average activity across all trials for one electrode
# This will help identify consistent neural responses to stimulation
electrode_idx = 16  # Choose a representative electrode
window_size = int(0.5 * sampling_rate)  # 500 ms window

# Initialize array for trial-averaged data
all_trials_data = np.zeros((min(100, len(trials_df)), window_size))

# Loop through first 100 trials to keep computation manageable
for i, (trial_idx, trial) in enumerate(trials_df.iloc[:100].iterrows()):
    trial_start_idx = int(trial['start_time'] * sampling_rate)
    # Extract data for this trial, up to window_size samples
    end_idx = min(trial_start_idx + window_size, nwb.acquisition["ElectricalSeries"].data.shape[0])
    actual_len = end_idx - trial_start_idx
    
    if actual_len < window_size:
        # Skip trials near the end that don't have enough data
        continue
    
    trial_data = nwb.acquisition["ElectricalSeries"].data[trial_start_idx:end_idx, electrode_idx]
    all_trials_data[i, :actual_len] = trial_data

# Calculate mean and standard deviation across trials
mean_response = np.mean(all_trials_data, axis=0)
std_response = np.std(all_trials_data, axis=0)

# Time points for the window
window_time = np.arange(window_size) / sampling_rate

# Plot trial-averaged response
plt.figure(figsize=(12, 6))
plt.plot(window_time, mean_response, 'b', label=f'Mean Response (Electrode {electrode_idx})')
plt.fill_between(window_time, mean_response - std_response, mean_response + std_response, 
                 color='b', alpha=0.2, label='±1 Std Dev')
plt.axvline(x=0, color='r', linestyle='--', label='Stim Onset')
plt.title(f'Trial-Averaged Neural Response - Electrode {electrode_idx}')
plt.xlabel('Time from Stimulus Onset (s)')
plt.ylabel('Voltage (V)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('trial_averaged_response.png')
plt.close()

# Close the file
io.close()

print("\nCompleted neural activity analysis. Generated plots:")
print("- electrode_activity_trial.png: Shows activity across multiple electrodes for a single trial")
print("- electrode_immediate_response.png: Shows the immediate post-stimulus response")
print("- trial_averaged_response.png: Shows trial-averaged response for one electrode")