# This script explores the metadata of an NWB file from Dandiset 001363
# The goal is to understand what data is available and how it's structured

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

# Print basic metadata
print("=== NWB File Basic Metadata ===")
print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Institution: {nwb.institution}")

# Print subject information
print("\n=== Subject Information ===")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")
print(f"Description: {nwb.subject.description}")

# Examine trials information
print("\n=== Trials Information ===")
print(f"Number of Trials: {len(nwb.trials.id[:])}")
print(f"Trial Columns: {nwb.trials.colnames}")

# Print the first few trials
print("\n=== First 5 Trials ===")
trials_df = nwb.trials.to_dataframe()
print(trials_df.head())

# Examine electrode information
print("\n=== Electrode Information ===")
print(f"Number of Electrodes: {len(nwb.electrodes.id[:])}")
print(f"Electrode Columns: {nwb.electrodes.colnames}")

# Print the first few electrodes
print("\n=== First 5 Electrodes ===")
electrodes_df = nwb.electrodes.to_dataframe()
print(electrodes_df.head())

# Examine electrical series data
print("\n=== Electrical Series Data ===")
electrical_series = nwb.acquisition["ElectricalSeries"]
print(f"Data Shape: {electrical_series.data.shape}")
print(f"Sampling Rate: {electrical_series.rate} Hz")
print(f"Starting Time: {electrical_series.starting_time} {electrical_series.starting_time_unit}")
print(f"Unit: {electrical_series.unit}")

# Calculate recording duration
duration = electrical_series.data.shape[0] / electrical_series.rate
print(f"Recording Duration: {duration:.2f} seconds = {duration/60:.2f} minutes")

# Close the file
io.close()