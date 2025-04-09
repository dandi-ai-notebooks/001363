# This script inspects metadata and variables in the NWB file for Dandiset 001363
# It prints information about subject, session, acquisition groups, intervals, units and other dataset variables
# This helps identify rich data elements useful for visualization and further exploration

import pynwb
import h5py
import remfile

url = "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwb = io.read()

print("Subject info:")
print(vars(nwb.subject))

print("\\nSession start:", nwb.session_start_time)

print("\\nDevices in file:")
for k, v in nwb.devices.items():
    print(f"Device {k}: {v.description} by {v.manufacturer}")

print("\\nAvailable acquisitions:")
for k in nwb.acquisition:
    print(f"- {k}")

print("\\nAvailable processing modules:")
for k in nwb.processing:
    print(f"- {k}")

print("\\nEpochs:", nwb.epochs)
print("Trials description:", getattr(nwb.trials, "description", "None"))
print("Trial columns:", getattr(nwb.trials, "colnames", "None"))
print("Number of trials:", len(nwb.trials))

if hasattr(nwb, 'units') and nwb.units is not None:
    print("\\nUnits info:")
    print("Columns:", nwb.units.colnames)
    print("Number of units:", len(nwb.units))
else:
    print("\\nNo units found in NWB file.")

print("\\nDone inspecting NWB file metadata.")

io.close()