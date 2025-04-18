# This script explores the trial information from the dataset

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

# Extract trial information
trials_df = nwb.trials.to_dataframe()
print(f"Total number of trials: {len(trials_df)}")
print("\nFirst 5 trials:")
print(trials_df.head())

# Calculate trial durations and intervals
trials_df['duration'] = trials_df['stop_time'] - trials_df['start_time']
trials_df['interval'] = trials_df['start_time'].shift(-1) - trials_df['start_time']

print("\nTrial duration statistics (seconds):")
print(trials_df['duration'].describe())

print("\nTrial interval statistics (seconds):")
print(trials_df['interval'].describe())

# Plot trial durations
plt.figure(figsize=(12, 6))
plt.hist(trials_df['duration'], bins=30)
plt.title('Distribution of Trial Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('explore/trial_durations.png')

# Plot trial start times
plt.figure(figsize=(12, 6))
plt.plot(trials_df.index, trials_df['start_time'], 'o-')
plt.title('Trial Start Times')
plt.xlabel('Trial Number')
plt.ylabel('Start Time (seconds)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('explore/trial_start_times.png')

# Plot trial intervals
plt.figure(figsize=(12, 6))
plt.hist(trials_df['interval'][:-1], bins=30)  # Exclude last trial as it has no following interval
plt.title('Distribution of Inter-Trial Intervals')
plt.xlabel('Interval (seconds)')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('explore/trial_intervals.png')