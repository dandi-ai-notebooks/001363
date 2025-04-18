# This script explores the trial structure of the NWB file
# The goal is to understand the timing and spacing of trials

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

# Get trials data
trials_df = nwb.trials.to_dataframe()

# Calculate trial durations and intervals
trials_df['duration'] = trials_df['stop_time'] - trials_df['start_time']
trials_df['interval'] = trials_df['start_time'].shift(-1) - trials_df['start_time']

# Print summary statistics
print("=== Trial Duration Statistics ===")
print(f"Mean: {trials_df['duration'].mean():.4f} seconds")
print(f"Std Dev: {trials_df['duration'].std():.4f} seconds")
print(f"Min: {trials_df['duration'].min():.4f} seconds")
print(f"Max: {trials_df['duration'].max():.4f} seconds")

print("\n=== Trial Interval Statistics ===")
print(f"Mean: {trials_df['interval'].mean():.4f} seconds")
print(f"Std Dev: {trials_df['interval'].std():.4f} seconds")
print(f"Min: {trials_df['interval'].min():.4f} seconds")
print(f"Max: {trials_df['interval'].max():.4f} seconds")

# Plot histograms of trial durations and intervals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(trials_df['duration'], bins=20)
ax1.set_title('Trial Duration Distribution')
ax1.set_xlabel('Duration (seconds)')
ax1.set_ylabel('Count')

ax2.hist(trials_df['interval'], bins=20)
ax2.set_title('Trial Interval Distribution')
ax2.set_xlabel('Interval (seconds)')
ax2.set_ylabel('Count')

plt.tight_layout()
plt.savefig('trial_distribution.png')
plt.close()

# Plot trial start times over the recording
plt.figure(figsize=(12, 4))
plt.plot(trials_df.index, trials_df['start_time'], 'o-')
plt.title('Trial Start Times')
plt.xlabel('Trial Number')
plt.ylabel('Time (seconds)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('trial_timing.png')
plt.close()

# Extract information about the first few and last few trials
print("\n=== First 5 Trials ===")
print(trials_df.head())

print("\n=== Last 5 Trials ===")
print(trials_df.tail())

# Close the file
io.close()