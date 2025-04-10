# %% [markdown]
# # Exploratory Analysis of DANDI Dandiset 001363
#
# **IMPORTANT DISCLAIMER:**  
# This notebook was autogenerated using `dandi-notebook-gen` tools powered by AI. The code has not been fully reviewed or validated. Please double-check the analyses and interpret the scientific results with caution.
#
# ---
#
# ## Citation
# Ramachandran, Sandhya; Gao, Huan; Yu, Kai; Yeh, Kelly; He, Bin (2025) Neural Spiking Data in the Rat Somatosensory Cortex Using a Flexible Electrode Responding to Transcranial Focused Ultrasound (Version draft) [Data set]. DANDI Archive. https://dandiarchive.org/dandiset/001363/draft
#
# ---
#
# ## About the dataset
# This dataset examines neuronal responses in the rat somatosensory cortex during transcranial focused ultrasound (tFUS) stimulation, recorded via an ultraflexible nanoelectrode array.
# Across experimental sessions, various ultrasound parameters were manipulated, including pressure amplitude, duty cycle, and pulse repetition frequency.
#
# Recordings contain ~21 minutes of continuous multi-channel extracellular signals per session, along with timestamped trial intervals associated with ultrasound pulses.
#
# ---
#
# ## Software Requirements
# This notebook requires the following Python packages (install them per your environment, e.g. via pip or conda):
# - pynwb
# - lindi
# - matplotlib
# - numpy
# - dandi
#
# ---
# %% [markdown]
# ## Listing Dandiset Assets via DANDI API

# %%
from dandi.dandiapi import DandiAPIClient
client = DandiAPIClient()
dandiset = client.get_dandiset("001363")
assets = list(dandiset.get_assets())
print(f"Number of assets in Dandiset 001363: {len(assets)}")
for idx, asset in enumerate(assets[:5]):
    print(f"{idx+1}. {asset.path} ({asset.size/1e9:.2f} GB)")

# %% [markdown]
# ## Selecting an example NWB file
# For this notebook, we focus on the following NWB recording session:
#
# **sub-BH589_ses-20240827T160457_ecephys.nwb**  
# URL: `https://lindi.neurosift.org/dandi/dandisets/001363/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/nwb.lindi.json`
#
# ---
# %% [markdown]
# ## Loading the NWB file with `lindi` and `pynwb`

# %%
import pynwb
import lindi

url = "https://lindi.neurosift.org/dandi/dandisets/001363/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start: {nwb.session_start_time}")

# %% [markdown]
# ## Subject metadata

# %%
subject = nwb.subject
print(f"Subject ID: {subject.subject_id}")
print(f"Species: {subject.species}")
print(f"Sex: {subject.sex}")
print(f"Age: {subject.age}")
print(f"Description: {subject.description}")

# %% [markdown]
# ## Overview of data contents

# %%
# List acquisitions and intervals tables
print(f"Acquisition keys: {list(nwb.acquisition.keys())}")
print(f"Intervals keys: {list(nwb.intervals.keys())}")

# Details on continuous signal:
es = nwb.acquisition.get("ElectricalSeries")
print(f"ElectricalSeries shape: {es.data.shape}")
print(f"Sampling rate: {es.rate} Hz")

# Trials info:
trials = nwb.intervals.get("trials")
print(f"Number of trials: {len(trials['id'])}")

# %% [markdown]
# ## Exploring electrode metadata

# %%
electrodes = nwb.electrodes
print("Columns:", electrodes.colnames)

for col in electrodes.colnames:
    vals = electrodes[col].data[:]
    print(f"{col}: {vals}")

# %% [markdown]
# ## Visualizing a snippet of data around a trial onset
#
# The following figure illustrates a ±100 ms snippet of continuous raw signals around a randomly selected trial onset across all channels.
# This brief window aids visualization without loading the full dataset.

# %%
import numpy as np
import matplotlib.pyplot as plt
import random

rate = es.rate

starts = trials['start_time'][:]
selected_trial = random.choice(starts)

window = 0.1  # seconds before and after
start_idx = int(max((selected_trial - window) * rate, 0))
end_idx = int(min((selected_trial + window) * rate, es.data.shape[0]))

snippet = es.data[start_idx:end_idx, :]
time = np.arange(snippet.shape[0]) / rate + (start_idx / rate - selected_trial)

plt.figure(figsize=(12, 8))
for ch in range(snippet.shape[1]):
    plt.plot(time * 1000, snippet[:, ch] * 1e6 + ch * 100, label=f'Ch {ch}')  # microvolt scale + offset
plt.xlabel('Time (ms)')
plt.ylabel('Microvolts + offset')
plt.title('Signal snippet around a random trial onset')
plt.tight_layout()
plt.show()

# %% [markdown]
# This plot shows relatively clean baseline neural signals across all electrodes during a typical trial window. Clear multi-unit spikes are not apparent in this brief window, but data quality appears sufficient for spike sorting or further analyses after appropriate processing.
#
# ---
#
# ## Suggestions for further analyses
#
# - Event-aligned averaging of neural signals across trials to identify ultrasound-evoked population responses.
# - Spike sorting to extract single and multi-unit activity.
# - Cross-channel synchronization or correlation analyses.
# - Relating neural features to ultrasound parameters (pressure, PRF, duty cycle).
# - Statistical analysis to validate observed effects.
#
# These analyses require additional processing pipelines beyond this notebook's scope.
#
# ---
#
# **Reminder:**  
# This notebook was automatically generated and should be validated and supplemented for your specific research needs.