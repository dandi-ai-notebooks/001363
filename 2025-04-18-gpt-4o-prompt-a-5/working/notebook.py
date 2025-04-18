# %% [markdown]
# # Exploring Dandiset 001363: Neural Spiking Data in the Rat Somatosensory Cortex
# 
# This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview
# In this study, we investigate the neuronal response to transcranial focused ultrasound stimulation (tFUS) on the somatosensory cortex using a 128-element array transducer and a chronically implanted ultraflexible nanoelectric thread electrode.
# 
# - Dandiset ID: 001363
# - Version: Draft
# - Access: Open
# - License: CC-BY-4.0
# - Researchers: Sandhya Ramachandran, Huan Gao, et al.
# - [Explore in Neurosift](https://neurosift.app/dandiset/001363)

# %% [markdown]
# ## What this Notebook Covers
# - Loading the Dandiset data using the DANDI API.
# - Exploring and visualizing NWB data files.
# - Analyzing specific data points and metadata.

# %% [markdown]
# ## Required Packages
# Ensure you have the following packages before proceeding:
# - dandi
# - pynwb
# - h5py
# - matplotlib
# - pandas

# %%
from dandi.dandiapi import DandiAPIClient
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import pandas as pd

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001363")
assets = list(dandiset.get_assets())

# Show assets
print(f"Found {len(assets)} assets in the dataset")
print("\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Load Specific NWB File and Display Metadata
# We will load the NWB file from `sub-BH589/sub-BH589_ses-20240827T160457_ecephys.nwb`.

# %%
# URL to the NWB file
url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"
# Load the NWB file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Display basic metadata
print("NWB Session Description:", nwb.session_description)
print("NWB Identifier:", nwb.identifier)
print("Session Start Time:", nwb.session_start_time)

# %% [markdown]
# ## Visualize Electrophysiology Data

# %%
# Extracting ElectricalSeries data
electrical_series = nwb.acquisition["ElectricalSeries"]

# Display basic info
print("Acquisition Comments:", electrical_series.comments)
print("Acquisition Description:", electrical_series.description)

# Visualize a subset of data
data = electrical_series.data[0:10, :10]  # First 10 samples, first 10 channels
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title("Sample Electrophysiology Data (Subset)")
plt.xlabel("Sample Index")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.show()

# %% [markdown]
# ## Explore Electrode Metadata

# %%
# Convert electrode table to DataFrame
electrodes_df = nwb.electrodes.to_dataframe()

# Display first rows
print(electrodes_df.head())

# %% [markdown]
# ## Findings and Future Directions
# This notebook has outlined the methods to load and visualize data from the NWB file. Future work could involve deeper analysis of spike activities and synchronization patterns, requiring dedicated computational tools.

io.close()  # Properly close the file to release resources