from pathlib import Path

import numpy as np

from one.api import ONE
from iblatlas.atlas import BrainRegions
from viewephys.gui import viewephys
from brainbox.io.one import SpikeSortingLoader


regions = BrainRegions()
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

PATH_CBIN = Path("/datadisk/Data/neuropixel/spike_sorting/benchmarks")
PATH_SPIKE_INTERFACE = Path("/datadisk/Data/neuropixel/spike_sorting/benchmarks/spikeinterface")
path_pictures = PATH_SPIKE_INTERFACE / "pictures"
path_pictures.mkdir(exist_ok=True, parents=True)

for pid in benchmark_pids:
    print(pid)
    if pid != '5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e':
        continue
    if PATH_SPIKE_INTERFACE.joinpath(f"{pid}_original.npy").exists():



        original = np.load(PATH_SPIKE_INTERFACE / f"{pid}_original.npy")
        residual = np.load(PATH_SPIKE_INTERFACE / f"{pid}_residual.npy")
        spikes_model = np.load(PATH_SPIKE_INTERFACE / f"{pid}_spikes_model.npy")


        from spikeinterface.extractors.cbin_ibl import CompressedBinaryIblExtractor
        from spikeinterface.extractors import read_alf_sorting


        cbin_file = PATH_CBIN.joinpath(f"{pid}.ap.cbin")
        recording = CompressedBinaryIblExtractor(cbin_file=cbin_file)
        path_sorting = PATH_CBIN.joinpath(pid, '1.5.0', 'alf')

        sorting = read_alf_sorting(path_sorting, sampling_frequency=recording.sampling_frequency)
        spike_samples = sorting.to_spike_vector()['sample_index']
        start_frame = int(200 * recording.sampling_frequency)
        end_frame = int(start_frame + (.1 * recording.sampling_frequency))
        sel_spikes = np.logical_and(spike_samples >= start_frame, spike_samples <= end_frame)
        spike_channels = sorting.alf_clusters['channels'][sorting.to_spike_vector()['unit_index'][sel_spikes]]
        spike_samples = spike_samples[sel_spikes] - start_frame


        eqcs = {}
        sl = SpikeSortingLoader(pid=pid, one=one)  # this is to get the channel geometry
        kwargs = {'br': regions, 'channels': sl.load_channels(), 'fs': recording.sampling_frequency}
        eqcs['original'] = viewephys(original, title='original', **kwargs)
        eqcs['residual'] = viewephys(residual, title='residual', **kwargs)
        eqcs['spikes_model'] = viewephys(spikes_model, title='signal model', **kwargs)

        for label, eqc in eqcs.items():
            eqc.viewBox_seismic.setYRange(0, recording.get_num_channels())
            eqc.viewBox_seismic.setXRange(25, 75)
            eqc.ctrl.set_gain(138)
            eqc.ctrl.add_scatter(spike_samples / recording.sampling_frequency * 1e3, spike_channels, rgb=(255, 0, 0), label='spikes')
            eqc.resize(1800, 900)
            eqc.grab().save(str(path_pictures.joinpath(f"data_{pid}_{label}.png")))

