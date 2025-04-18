# Explore the trials data in the NWB file
import pynwb
import h5py
import remfile
import pandas as pd

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the trials data
trials = nwb.intervals["trials"]

# Convert to a Pandas DataFrame
trials_df = trials.to_dataframe()

# Print some info
print(trials_df.head())
print(trials_df.describe())