# explore/explore_trials.py
# This script explores the trials data in the NWB file.
# It loads the NWB file, access the trials data,
# prints the first few rows of the trials dataframe, and plots the duration of each trial as a histogram.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Access the trials data
trials = nwb.intervals["trials"]
trials_df = trials.to_dataframe()

# Subtract session_start_time from start_time and stop_time
session_start_time = nwb.session_start_time.timestamp()
trials_df["start_time"] = trials_df["start_time"] - session_start_time
trials_df["stop_time"] = trials_df["stop_time"] - session_start_time


# Print the first few rows of the trials dataframe
print(trials_df.head())

# Calculate the duration of each trial
trials_df["duration"] = trials_df["stop_time"] - trials_df["start_time"]

# Plot the duration of each trial as a histogram
plt.figure(figsize=(10, 5))
plt.hist(trials_df["duration"], bins=20)
plt.xlabel("Trial duration (s)")
plt.ylabel("Number of trials")
plt.title("Distribution of trial durations")
plt.savefig("explore/trial_durations.png")

print("Plot saved to explore/trial_durations.png")