# This script plots trial start and stop times for Dandiset 001363 NWB file
# It verifies the presence of trials and visualizes their timing as a raster plot
# This can reveal task structure even if extracellular raw data appears flat

import matplotlib.pyplot as plt
import pynwb
import h5py
import remfile

url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwb = io.read()

trials = nwb.trials
start_times = trials['start_time'].data[:]
stop_times = trials['stop_time'].data[:]

num_trials = len(start_times)
duration = stop_times - start_times

plt.figure(figsize=(15, 6))
for i in range(num_trials):
    plt.plot([start_times[i], stop_times[i]], [i, i], color='black')

plt.xlabel('Time (s)')
plt.ylabel('Trial index')
plt.title('Trial start and stop times')
plt.tight_layout()
plt.savefig('tmp_scripts/trials_raster.png')
print("Saved plot to tmp_scripts/trials_raster.png")

io.close()