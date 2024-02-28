import numpy as np
from pathlib import Path
import spikeinterface.full as si
from sklearn.metrics import auc

import matplotlib.pyplot as plt

from utils import compute_residuals, peak_detection_sweep

si.set_global_job_kwargs(n_jobs=0.8, progress_bar=True)

## LOAD DATA
output_folder = Path("YOUR_OUTPUT_PATH_HERE")
# LOAD YOUR RECORDING HERE
recording = None
# LOAD YOUR SORTING HERE
sorting = None 

print(recording)

### Preprocessing
recording_processed = si.phase_shift(recording)
recording_processed = si.highpass_filter(recording_processed)
bad_channel_ids, bad_channel_labels = si.detect_bad_channels(recording_processed)
# remove bad channels
recording_clean = recording_processed.remove_channels(bad_channel_ids)
recording_clean = si.common_reference(recording_clean)
print(recording_clean)

# Extract waveforms
we = si.extract_waveforms(recording_clean, sorting, folder=output_folder / f"waveforms",
                          overwrite=True, return_scaled=False)
residual, convolved = compute_residuals(we, with_scaling=True)

# Detection analysis
thresholds = np.arange(3, 11)[::-1]
thresholds
noise_levels = si.get_noise_levels(recording_clean)
peak_counts, peak_list = peak_detection_sweep(
    residual, thresholds, noise_levels=noise_levels, peak_sign="both"
)

# Plot detected peaks
fig, ax = plt.subplots()

ax.plot(peak_counts.keys(), peak_counts.values())
auc_val = auc(list(peak_counts.keys()), list(peak_counts.values()))
print(f"AUC: {auc_val}")
ax.legend()
