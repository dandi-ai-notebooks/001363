# %% [markdown]
# # Exploring Dandiset 001363: Neural Spiking Data in the Rat Somatosensory Cortex
#
# **Note:** This notebook was AI-generated using `dandi-notebook-gen` and has not been fully verified. Please be cautious when interpreting the code or results.
#
# This notebook explores a Dandiset focused on neural spiking data using a flexible electrode with transcranial focused ultrasound. 
# For more detailed information, visit [NeuroSift Dandiset 001363](https://neurosift.app/dandiset/001363).
#
# ## Overview
#
# We will analyze data collected from a rat's somatosensory cortex in response to varying levels of ultrasound stimulation and duty cycles. 
# Key analysis steps include loading the Dandiset, exploring an NWB file, visualizing sections of recorded data, and summarizing potential 
# directions for future research.
#
# ## Required Packages
#
# - pynwb
# - h5py
# - remfile
# - matplotlib
# - pandas
# - seaborn

# %% [markdown]
# ## Connect to DANDI and Load Dandiset

# %%
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001363")
assets = list(dandiset.get_assets())

print(f"Found {len(assets)} assets in the dataset")
print("\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Load and Explore NWB File

# %%
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Display basic metadata
print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session Start Time: {nwb.session_start_time}")

# %% [markdown]
# ### Visualize Electrical Series Data

# %%
# Access ElectricalSeries data
electrical_series_data = nwb.acquisition["ElectricalSeries"].data

# Plot data segment
start_idx = 0
end_idx = 1000  # View first 1000 samples from channel 0
data_segment = electrical_series_data[start_idx:end_idx, 0]

plt.figure(figsize=(10, 4))
plt.plot(data_segment)
plt.title("Electrical Series Data Segment: Channel 0")
plt.xlabel("Sample Index")
plt.ylabel("Voltage (V)")
plt.show()

io.close()
h5_file.close()

# %% [markdown]
# ## Conclusion and Future Directions
#
# This notebook has demonstrated loading and visualizing a segment of neural spiking data from a rat's somatosensory cortex. 
# Future research could explore other stimulation parameters or analyze larger data sections to provide deeper insights into neuronal responses to ultrasound stimulation.