# %%
import matplotlib.pyplot as plt
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from ibldsp.voltage import destripe
from viewephys.gui import viewephys
import numpy as np
import pandas as pd
import ibldsp.waveforms as waveforms
from neuropixel import trace_header
import ibldsp

one = ONE(base_url='https://openalyx.internationalbrainlab.org')

benchmark_pids = ['1a276285-8b0e-4cc9-9f0a-a3a002978724',
                  '1e104bf4-7a24-4624-a5b2-c2c8289c0de7',
                  '5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e',
                  '5f7766ce-8e2e-410c-9195-6bf089fea4fd',
                  '6638cfb3-3831-4fc2-9327-194b76cf22e1',
                  '749cb2b7-e57e-4453-a794-f6230e4d0226',
                  'd7ec0892-0a6c-4f4f-9d8f-72083692af5c',
                  'da8dfec1-d265-44e8-84ce-6ae9c109b8bd',
                  'dab512bd-a02d-4c1f-8dbc-9155a163efc0',
                  'dc7e9403-19f7-409f-9240-05ee57cb7aea',
                  'e8f9fba4-d151-4b00-bee7-447f0f3e752c',
                  'eebcaf65-7fa4-4118-869d-a084e84530e2',
                  'fe380793-8035-414e-b000-09bfe5ece92a']
# %%
# Select a PID
pid = benchmark_pids[2]

# Load spike sorting
sl = SpikeSortingLoader(pid=pid, one=one)
spikes, clusters, channels = sl.load_spike_sorting(dataset_types=['clusters.amps', 'spikes.samples'])
# sl.merge_clusters()

# Get AP spikeglx.Reader objects
sr_ap = sl.raw_electrophysiology(band="ap", stream=True)

# Load raw data
window_secs_ap = np.array((0, 1))  # timepoint in recording to stream
first_sample, last_sample = int(window_secs_ap[0] * sr_ap.fs), int(window_secs_ap[1] * sr_ap.fs)
raw_ap = sr_ap[first_sample:last_sample, :-sr_ap.nsync].T

# Destripe
destriped = destripe(raw_ap, fs=sr_ap.fs)

##
# Get array of waveforms for all spikes

# Truncate spikes to the window
# we remove 100 ms to make sure there is enough raw data for the first/last spikes
spik_in = (spikes['times'] > window_secs_ap[0] + 0.1) & (spikes['times'] < window_secs_ap[1] - 0.100)
for k in spikes.keys():
    spikes[k] = spikes[k][spik_in]
spikes['samples_aligned'] = (spikes.samples - first_sample).astype('int')

arr = destriped.T

df = pd.DataFrame({"sample": spikes['samples_aligned'],
                   "peak_channel": clusters['channels'][spikes['clusters']]})
# generate channel neighbor matrix for NP1, default radius 200um
geom_dict = trace_header(version=1)
geom = np.c_[geom_dict["x"], geom_dict["y"]]
channel_neighbors = ibldsp.utils.make_channel_index(geom, radius=200.)
# radius = 200um, 38 chans
num_channels = 38
wfs, cind, trough_offset = waveforms.extract_wfs_array(arr, df, channel_neighbors, add_nan_trace=True)


##
# remove for all
arr_start = arr.copy()
nch = len(channels['x'])
# Do average for single cluster
for cluster_id in np.unique(spikes['clusters']):

    spikes_idx = spikes['clusters'] == cluster_id
    wfs_avg = np.nanmean(wfs[spikes_idx, :, :], axis=0)

    # Remove template from raw data
    spike_samples = spikes['samples'][spikes_idx]

    ch_cluster = cind[spikes_idx]
    spike_samples_start = spike_samples - trough_offset
    spikes_index_remove = np.array(range(0, wfs_avg.shape[1])) + np.array(spike_samples_start).reshape(-1, 1)

    arr_sub = arr_start.copy()
    for ispike in range(0, len(spike_samples_start)):
        # Substract average wav from raw data and replace in array
        # Edge case to remove ch indices outside
        idx_ch_notout = ch_cluster[ispike, :]
        idx_ch_notout = idx_ch_notout[np.where(idx_ch_notout < nch)[0]]
        indx = np.ix_(spikes_index_remove[ispike, :].astype('int'), idx_ch_notout.astype('int'))
        arr_sub[indx] = arr_start[indx] - wfs_avg.T[:, np.where(idx_ch_notout < nch)[0]]
        # TODO remove channels from avg that are outside
    print(cluster_id)
    arr_start = arr_sub.copy()

##
clu_ch = clusters['channels'][cluster_id]
viewers = {}
viewers['raw'] = viewephys(arr.T, fs=sr_ap.fs, title='raw')
viewers['remove'] = viewephys(arr_sub.T, fs=sr_ap.fs, title='remove')
for label in viewers:
    viewers[label].ctrl.add_scatter(spike_samples / sr_ap.fs * 1e3 - first_sample * 1e3, np.repeat(clu_ch, len(spike_samples)),
                           label='detects_ibl', rgb=(255, 0, 0))
