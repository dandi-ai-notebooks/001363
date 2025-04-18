# Explore the trials data and plot trial durations
import pynwb
import h5py
import remfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"  # Original URL
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the trials data
trials = nwb.intervals["trials"]
trials_df = trials.to_dataframe()

# Calculate trial durations
trials_df['duration'] = trials_df['stop_time'] - trials_df['start_time']

# Print the trial durations
print(trials_df['duration'])

# Plot the trial durations
plt.figure(figsize=(10, 6))
plt.hist(trials_df['duration'], bins=20)
plt.xlabel('Trial Duration (s)')
plt.ylabel('Number of Trials')
plt.title('Distribution of Trial Durations')
plt.savefig('explore2/trial_durations.png')
plt.close()