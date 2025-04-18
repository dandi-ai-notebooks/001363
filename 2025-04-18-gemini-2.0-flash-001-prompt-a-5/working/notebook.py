# %% [markdown]
# Exploring Dandiset 001363: Neural Spiking Data in the Rat Somatosensory Cortex

# %% [markdown]
# ⚠️ This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Use caution when interpreting the code or results.

# %% [markdown]
# ## Overview of the Dandiset
#
# This notebook explores the neural spiking data in the rat somatosensory cortex acquired using a flexible electrode in response to transcranial focused ultrasound. The dataset contains recordings from multiple sessions and animals, with varying levels of ultrasound pressure and duty cycles.
#
# You can find more information about this Dandiset on the DANDI Archive:
#
# [https://dandiarchive.org/dandiset/001363](https://dandiarchive.org/dandiset/001363)
#
# And visualize the data with neurosift:
#
# [https://neurosift.app/dandiset/001363](https://neurosift.app/dandiset/001363)

# %% [markdown]
# ## What this notebook will cover
#
# This notebook will guide you through the process of loading and visualizing data from the Dandiset. We will cover:
#
# 1.  Connecting to the DANDI archive and listing the available assets.
# 2.  Loading an NWB file and exploring its metadata.
# 3.  Loading and visualizing electrophysiology data.
# 4.  Exploring trial-related information.

# %% [markdown]
# ## Required Packages
#
# The following packages are required to run this notebook:
#
# *   `pynwb`
# *   `h5py`
# *   `remfile`
# *   `numpy`
# *   `matplotlib`
# *   `seaborn`
# *   `pandas`

# %%
# Connect to the DANDI archive
from dandi.dandiapi import DandiAPIClient

client = DandiAPIClient()
dandiset = client.get_dandiset("001363")
assets = list(dandiset.get_assets())

print(f"Found {len(assets)} assets in the dataset")
print("\\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Loading an NWB file and exploring its metadata
#
# We will now load one of the NWB files from the Dandiset and explore its metadata.
#
# We will load the file: `sub-BH589/sub-BH589_ses-20240827T160457_ecephys.nwb`

# %%
import pynwb
import h5py
import remfile

# Load
url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Display some metadata
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Institution: {nwb.institution}")
print(f"Subject ID: {nwb.subject.subject_id}")

# %% [markdown]
# ## Loading and visualizing electrophysiology data
#
# Now, let's load and visualize some electrophysiology data from the NWB file.

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load electrophysiology data
electrical_series = nwb.acquisition["ElectricalSeries"]
data = electrical_series.data
rate = electrical_series.rate
print(f"Shape of electrical series dataset: {data.shape}")
print(f"Sampling rate: {rate} Hz")

# Select a subset of channels and timepoints for visualization
num_channels = 32
num_timepoints = 10000

# Load a subset of data
data_subset = data[:num_timepoints, :num_channels]

# Create a time vector
time = np.arange(0, num_timepoints / rate, 1 / rate)

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(time, data_subset)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Electrophysiology Data")
plt.show()

# %% [markdown]
# ## Exploring trial-related information
#
# Let's explore the trial-related information in the NWB file.

# %%
# Access the trials table
trials = nwb.intervals["trials"]

# Print the column names
print(f"Trial table column names: {trials.colnames}")

# Convert trials table to a dataframe
import pandas as pd
trials_df = trials.to_dataframe()

# Print the first 5 rows of the trials table
print(trials_df.head())

# %% [markdown]
# ## Summarizing findings and future directions
#
# In this notebook, we have successfully loaded and visualized electrophysiology data from a DANDI Archive Dandiset. We have also explored the trial-related information in the NWB file.
#
# Possible future directions for analysis:
#
# *   Investigate the neural response to transcranial focused ultrasound stimulation (tFUS) on the somatosensory cortex.
# *   Analyze the effect of varying duty cycle on the neural response.
# *   Explore the effect of varying PRF on the neural response.