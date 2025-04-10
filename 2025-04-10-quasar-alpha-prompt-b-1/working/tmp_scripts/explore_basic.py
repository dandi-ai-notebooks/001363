# This script streams the NWB file remotely.
# It extracts metadata (subject info, number of trials, electrodes)
# and plots a 1-sec segment of raw data from the first electrode channel.
# The plot image is saved in tmp_scripts/raw_data_snippet.png.

import remfile
import h5py
import pynwb
import matplotlib.pyplot as plt

nwb_url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"

file = remfile.File(nwb_url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwb = io.read()

# --- Basic metadata ---
print("Subject:", nwb.subject)
print("Subject description:", getattr(nwb.subject, 'description', 'N/A'))
print("Subject species:", getattr(nwb.subject, 'species', 'N/A'))

trials = nwb.trials
print("Number of trials:", len(trials))

electrodes = nwb.electrodes
print("Number of electrodes:", len(electrodes.id))

# --- Extract data snippet ---
data = nwb.acquisition["ElectricalSeries"].data
rate = nwb.acquisition["ElectricalSeries"].rate
duration_seconds = 1
num_samples = int(rate * duration_seconds)

snippet = data[:num_samples, 0]  # first second from first channel
snippet = snippet[:]  # load into numpy array

times = [i / rate for i in range(num_samples)]

plt.figure(figsize=(10, 4))
plt.plot(times, snippet)
plt.title("Raw data snippet (1 sec, channel 0)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.tight_layout()
plt.savefig("tmp_scripts/raw_data_snippet.png")
plt.close()
print("Saved raw_data_snippet.png")