# %% [markdown]
# **NOTE:** This notebook was *AI-generated* using dandi-notebook-gen. Neither the code nor the outputs have been fully verified. Please exercise caution when interpreting the results; you may wish to validate the code before relying on insights.
#
# # Exploring DANDI Dataset 001363: Neural Spiking Data in the Rat Somatosensory Cortex Using a Flexible Electrode Responding to Transcranial Focused Ultrasound
#
# **Citation:** Ramachandran, Sandhya; Gao, Huan; Yu, Kai; Yeh, Kelly; He, Bin (2025) Neural Spiking Data in the Rat Somatosensory Cortex Using a Flexible Electrode Responding to Transcranial Focused Ultrasound (Version draft) [Data set]. DANDI Archive. https://dandiarchive.org/dandiset/001363/draft
#
# ## Dataset Description
# This dataset investigates neuronal response to transcranial focused ultrasound stimulation on rat somatosensory cortex. A flexible ultraflexible nanoelectric thread (NET) electrode enables exploration of higher ultrasound intensities, avoiding artifacts encountered with rigid electrodes.
#
# Experiments varied:
# - Ultrasound pressure (100, 400, 700, 1000, 1300 kPa)
# - Duty cycle (0.6%, 6%, 30%, 60%, 90%)
# - Pulse repetition frequency (30 Hz up to 4500 Hz)
#
# Each recording has 505 trials. Recordings include multi-electrode extracellular electrophysiology data.
#
# ## Contents
# - Retrieve dandiset info using DANDI API
# - Listing assets in the Dandiset
# - Loading NWB file remotely
# - Exploring electrophysiology data
# - Example visualizations (raw traces, heatmaps, trial timing)
#
# **Note:** This notebook assumes you have installed packages:\n
# `pynwb`, `remfile`, `h5py`, `matplotlib`, `seaborn`, `dandi`.\n
# If not, please install them before running.

# %%
# Import libraries
import pynwb
import remfile
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dandi.dandiapi import DandiAPIClient

sns.set_theme()

# %% [markdown]
# ## Accessing Dandiset 001363 and listing assets programmatically

# %%
client = DandiAPIClient()
dandiset = client.get_dandiset("001363")
assets = list(dandiset.get_assets())
print(f"Total number of assets in Dandiset: {len(assets)}")
for asset in assets[:5]:
    print(f"- Path: {asset.path}, size: {asset.size} bytes")

# %% [markdown]
# ## Loading an example NWB file remotely
# We'll load a single file as example:  
# `sub-BH589/sub-BH589_ses-20240827T160457_ecephys.nwb`  
# *Note:* The data shape is large (31M timepoints × 32 channels), so we will only stream small chunks.

# %%
nwb_url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"

file = remfile.File(nwb_url)
h5f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=h5f)
nwbfile = io.read()
print("Loaded NWB file.")

# %% [markdown]
# ## Exploring metadata

# %%
print("Session:", nwbfile.session_description)
print("Subject ID:", nwbfile.subject.subject_id)
print("Species:", nwbfile.subject.species)
print("Sex:", nwbfile.subject.sex)
print("Institution:", nwbfile.institution)
print("Device(s):")
for device in nwbfile.devices.values():
    print(f" - {device.name}: {device.description} (Manufactured by: {device.manufacturer})")

# %% [markdown]
# ### Electrode information

# %%
electrodes = nwbfile.electrodes
print("Electrode columns:", electrodes.colnames)
print("Number of electrodes:", len(electrodes))
print(electrodes.to_dataframe().head())

# %% [markdown]
# ### Trial intervals

# %%
trials = nwbfile.trials
print("Number of trials:", len(trials))
print("Trial columns:", trials.colnames)
try:
    print(trials.to_dataframe().head())
except:
    print("Could not convert trials to DataFrame.")

# %% [markdown]
# ## Exploring extracellular recording data

# %%
ephys = nwbfile.acquisition['ElectricalSeries']
print("ElectricalSeries info:")
print(f"Sampling rate: {ephys.rate} Hz")
print(f"Shape: {ephys.data.shape}")
print(f"Unit: {ephys.unit}")
print(f"Description: {ephys.description}")

# %% [markdown]
# ### Plot snippet of extracellular traces from a few channels and short time window

# %%
# Visualization for the first 100 ms on 4 example channels
num_channels = ephys.data.shape[1]
snippet_duration_sec = 0.1  # 100 ms
fs = ephys.rate
start_sample = 0
end_sample = int(snippet_duration_sec * fs)
data_snippet = ephys.data[start_sample:end_sample, :4]  # first 4 channels

time_vector = np.arange(start_sample, end_sample) / fs

plt.figure(figsize=(10, 6))
for ch in range(4):
    plt.plot(time_vector, data_snippet[:, ch] * 1e3 + ch*2, label=f'Ch {ch}')  # Convert volts to mV offset for display
plt.xlabel('Time (s)')
plt.ylabel('Voltage + offset (mV)')
plt.title('Example extracellular voltage traces (first 100 ms)')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Heatmap of activity over short window and subset of channels

# %%
snippet = ephys.data[start_sample:end_sample, :16]  # first 16 channels

plt.figure(figsize=(12, 4))
sns.heatmap(snippet.T, cmap='viridis', cbar_kws={'label': 'Voltage (V)'})
plt.xlabel('Timepoint')
plt.ylabel('Channel')
plt.title('Heatmap of extracellular signals (first 100 ms, 16 channels)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Histogram of voltage distribution (small subset)

# %%
snippet_hist = ephys.data[start_sample:end_sample, :8]  # 8 channels
plt.figure(figsize=(8, 4))
plt.hist(snippet_hist.flatten()*1e3, bins=100)
plt.xlabel('Voltage (mV)')
plt.ylabel('Count')
plt.title('Histogram of extracellular voltage values (snippet)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Trial structure visualization

# %%
trial_df = None
try:
    trial_df = trials.to_dataframe()
except:
    print("Could not convert trials to DataFrame.")
if trial_df is not None:
    plt.figure(figsize=(10, 4))
    for idx, row in trial_df.iterrows():
        plt.plot([row['start_time'], row['stop_time']], [idx, idx], color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Trial index')
    plt.title('Trial intervals')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Summary
# This notebook demonstrated how to:
# - Connect to DANDI and browse assets
# - Remote-stream a NWB file
# - Explore metadata: subject, devices, electrodes, trials
# - Visualize a small snippet of extracellular electrophysiology data
#
# For further analysis, please tailor the provided code to your scientific questions and validate all steps.