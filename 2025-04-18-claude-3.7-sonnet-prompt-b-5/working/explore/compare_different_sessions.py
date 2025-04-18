# This script compares different recording sessions to examine the effects of different stimulation parameters

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# List of NWB files to compare (selected based on the dataset description)
# First session is our reference (100 kPa pressure)
# Second session is a higher pressure (1300 kPa)
nwb_files = [
    {
        "url": "https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/",
        "description": "First session (reference)",
        "color": "b"
    },
    {
        "url": "https://api.dandiarchive.org/api/assets/6b9aa3e6-2389-4f84-a2d0-a3201894ad3c/download/",
        "description": "Higher pressure session",
        "color": "r"
    }
]

# Function to load an NWB file and extract key information
def load_nwb_and_extract_info(file_info):
    print(f"Loading {file_info['description']}...")
    url = file_info['url']
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()
    
    # Get basic information
    info = {
        "description": file_info['description'],
        "color": file_info['color'],
        "session_id": nwb.identifier,
        "nwb": nwb
    }
    
    print(f"  Session ID: {info['session_id']}")
    return info

# Load all NWB files
session_info_list = []
for file_info in nwb_files:
    try:
        info = load_nwb_and_extract_info(file_info)
        session_info_list.append(info)
    except Exception as e:
        print(f"Error loading {file_info['description']}: {str(e)}")

# Check if we have at least one session loaded
if len(session_info_list) == 0:
    print("No sessions were successfully loaded. Exiting.")
    exit()

# Function to extract and analyze a subset of data from a trial
def analyze_trial_data(nwb, trial_index=0, channel=0):
    # Get electrical series data
    electrical_series = nwb.acquisition["ElectricalSeries"]
    
    # Get trial information
    trials_df = nwb.trials.to_dataframe()
    if trial_index >= len(trials_df):
        raise ValueError(f"Trial index {trial_index} is out of range (max: {len(trials_df)-1})")
        
    trial = trials_df.iloc[trial_index]
    # Convert pandas Series values to scalar float
    start_time = float(trial['start_time'])
    stop_time = float(trial['stop_time'])
    
    # Convert time to indices
    start_idx = int(start_time * electrical_series.rate)
    stop_idx = int(stop_time * electrical_series.rate)
    
    # Add padding
    padding = int(0.5 * electrical_series.rate)  # 500 ms padding
    start_with_padding = max(0, start_idx - padding)
    stop_with_padding = min(electrical_series.data.shape[0], stop_idx + padding)
    
    # Extract data
    trial_data = electrical_series.data[start_with_padding:stop_with_padding, channel]
    
    # Create time vector relative to stimulation onset
    trial_time = np.arange(len(trial_data)) / electrical_series.rate - (padding / electrical_series.rate)
    
    # Compute spectrogram
    fs = electrical_series.rate
    nperseg = int(fs * 0.1)  # 100 ms window
    noverlap = nperseg // 2
    f, t, Sxx = signal.spectrogram(trial_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    return {
        "trial_time": trial_time,
        "trial_data": trial_data,
        "spectrogram": {
            "f": f,
            "t": t - (padding / electrical_series.rate),  # Adjust time relative to stim onset
            "Sxx": Sxx
        },
        "start_time": start_time,
        "stop_time": stop_time,
        "duration": float(stop_time - start_time)
    }

# Compare first trial from each session
print("\nComparing first trial from each session...")
first_trial_data = []

for info in session_info_list:
    # Extract data from first trial, channel 0
    try:
        data = analyze_trial_data(info["nwb"], trial_index=0, channel=0)
        data["description"] = info["description"]
        data["color"] = info["color"]
        data["session_id"] = info["session_id"]
        first_trial_data.append(data)
    except Exception as e:
        print(f"Error analyzing trial data for {info['description']}: {str(e)}")

if len(first_trial_data) == 0:
    print("No trial data could be analyzed. Exiting.")
    exit()

# Plot raw voltage traces for first trial from each session
plt.figure(figsize=(14, 8))
for data in first_trial_data:
    plt.plot(data["trial_time"], data["trial_data"], color=data["color"], 
             label=f"{data['description']} ({data['session_id']})")

# Add vertical lines for stimulation period (using first session's duration)
plt.axvline(x=0, color='k', linestyle='--', label='Stimulation Start')
stim_duration = first_trial_data[0]["duration"] if first_trial_data else 1.5  # Default if no data
plt.axvline(x=stim_duration, color='k', linestyle='--', label='Stimulation End')

plt.xlabel('Time (s) relative to stimulation onset')
plt.ylabel('Voltage (V)')
plt.title('Comparison of Raw Voltage Traces from First Trial Across Sessions')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('explore/comparison_raw_traces.png')

# Plot spectrograms side by side
fig, axs = plt.subplots(1, len(first_trial_data), figsize=(18, 8), sharey=True)
if len(first_trial_data) == 1:
    axs = [axs]  # Make it a list for consistent indexing

for i, data in enumerate(first_trial_data):
    # Get spectrogram data
    spec = data["spectrogram"]
    
    pcm = axs[i].pcolormesh(spec["t"], spec["f"], 10 * np.log10(spec["Sxx"] + 1e-12), 
                        shading='gouraud', cmap='viridis')
    axs[i].set_title(f"{data['description']}\n({data['session_id']})")
    axs[i].set_xlabel('Time (s) relative to stimulation onset')
    
    # Add vertical lines for stimulation period
    axs[i].axvline(x=0, color='r', linestyle='--')
    axs[i].axvline(x=data["duration"], color='r', linestyle='--')
    
    # Focus on frequency range of interest
    axs[i].set_ylim(0, 5000)

# Add common labels
fig.text(0.5, 0.04, 'Time (s) relative to stimulation onset', ha='center')
fig.text(0.04, 0.5, 'Frequency (Hz)', va='center', rotation='vertical')
plt.suptitle('Comparison of Spectrograms from First Trial Across Sessions', fontsize=16)

# Add colorbar
cbar = fig.colorbar(pcm, ax=axs)
cbar.set_label('Power/Frequency (dB/Hz)')

plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
plt.savefig('explore/comparison_spectrograms.png')

# Calculate average power in different frequency bands for each session
print("\nCalculating average power in different frequency bands...")
frequency_bands = [
    (0, 100, "Delta/Theta (0-100 Hz)"),
    (100, 500, "Alpha/Beta (100-500 Hz)"),
    (500, 1000, "Low Gamma (500-1000 Hz)"),
    (1000, 5000, "High Gamma/Multi-unit (1000-5000 Hz)")
]

band_powers = {}
for data in first_trial_data:
    session_id = data["session_id"]
    spec = data["spectrogram"]
    band_powers[session_id] = {}
    
    for band_min, band_max, band_name in frequency_bands:
        # Find the indices corresponding to this frequency band
        band_indices = np.where((spec["f"] >= band_min) & (spec["f"] <= band_max))[0]
        
        # Calculate average power in the band during stimulation
        stim_indices = np.where((spec["t"] >= 0) & (spec["t"] <= data["duration"]))[0]
        
        if len(band_indices) > 0 and len(stim_indices) > 0:
            # Extract the portion of the spectrogram for this band during stimulation
            band_spec = spec["Sxx"][band_indices, :][:, stim_indices]
            
            # Calculate average power (in dB)
            avg_power = 10 * np.log10(np.mean(band_spec) + 1e-12)
            band_powers[session_id][band_name] = avg_power

# Plot comparison of power in different frequency bands
bands = [band[2] for band in frequency_bands]
plt.figure(figsize=(12, 8))
bar_width = 0.35
index = np.arange(len(bands))

for i, data in enumerate(first_trial_data):
    session_id = data["session_id"]
    if session_id in band_powers:
        values = [band_powers[session_id].get(band, 0) for band in bands]
        plt.bar(index + i*bar_width, values, bar_width, 
                label=f"{data['description']} ({session_id})", color=data["color"])

plt.xlabel('Frequency Band')
plt.ylabel('Average Power (dB)')
plt.title('Comparison of Power in Different Frequency Bands During Stimulation')
plt.xticks(index + bar_width/2, bands, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('explore/comparison_frequency_bands.png')

print("Analysis complete. Results saved to explore directory.")