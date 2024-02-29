# %%
from pathlib import Path
from one.api import ONE
import numpy as np
from brainbox.io.one import SpikeSortingLoader

from spikeinterface.extractors.cbin_ibl import CompressedBinaryIblExtractor

from spikeinterface.extractors import read_alf_sorting
import spikeinterface.full as si

from spikeinterface.core import BaseRecording
from spikeinterface.core.generate import InjectTemplatesRecording
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost


def compute_residuals(waveform_extractor, with_scaling=True, overwrite=False, sparsity=None,  **scaling_kwargs):
    """
    Compute residuals for a given recording and sorting.

    Parameters
    ----------
    waveform_extractor: si.WaveformExtractor
        The waveform extractor
    with_scaling: bool
        If True, use amplitude scaling

    Returns
    -------
    residuals: BaseRecording
        The recording after residual computation
    """
    if with_scaling:
        # compute amplitude scalings
        if waveform_extractor.has_extension("amplitude_scalings") and not overwrite:
            asc = waveform_extractor.load_extension("amplitude_scalings")
            amp_scalings= asc.get_data()
        else:
            amp_scalings = spost.compute_amplitude_scalings(waveform_extractor, **scaling_kwargs)
    else:
        amp_scalings = None

    templates = waveform_extractor.get_all_templates()
    if waveform_extractor.return_scaled:
        templates = templates / waveform_extractor.recording.get_channel_gains()[:, None]
    if sparsity:
        templates_full = np.zeros_like(templates)
        for i, (unit, sparsity_indices) in enumerate(sparsity.unit_id_to_channel_indices.items()):
            templates_full[i, :, sparsity_indices] = templates[i, :, sparsity_indices]
        templates = templates_full
    templates = templates.astype("float32")

    recording_float = spre.astype(waveform_extractor.recording, dtype="float32")

    residuals = InjectTemplatesRecording(
        sorting=waveform_extractor.sorting,
        parent_recording=recording_float,
        amplitude_factor=amp_scalings,
        templates=-templates
    )
    convolved = InjectTemplatesRecording(
        sorting=waveform_extractor.sorting,
        templates=templates,
        amplitude_factor=amp_scalings,
    )
    convolved = convolved.rename_channels(residuals.channel_ids)

    return residuals, convolved


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
PATH_SPIKE_INTERFACE.mkdir(exist_ok=True, parents=True)

for pid in benchmark_pids:
    if PATH_SPIKE_INTERFACE.joinpath(f"{pid}_original.npy").exists():
        continue
    cbin_file = PATH_CBIN.joinpath(f"{pid}.ap.cbin")
    recording = CompressedBinaryIblExtractor(cbin_file=cbin_file)
    sorting = read_alf_sorting(
        PATH_CBIN.joinpath(pid, '1.5.0', 'alf'), sampling_frequency=recording.sampling_frequency)
    print(sorting)

    print(f'{pid} pre-processing')
    file_preproc = PATH_SPIKE_INTERFACE / f"{pid}_preprocessed.json"
    if file_preproc.exists():
        recording_clean = si.load_extractor(file_preproc)
    else:
        recording_processed = si.phase_shift(recording)
        recording_processed = si.highpass_filter(recording_processed)
        bad_channel_ids, bad_channel_labels = si.detect_bad_channels(recording_processed)
        recording_clean = recording_processed.remove_channels(bad_channel_ids)
        recording_clean = si.common_reference(recording_clean)
        recording_clean.dump_to_json(PATH_SPIKE_INTERFACE / f"{pid}_preprocessed.json")

    print(recording_clean)
    folder_waveforms = PATH_SPIKE_INTERFACE.joinpath(f"waveforms_{pid}")
    # Extract waveforms
    if folder_waveforms.exists():
        we = si.load_waveforms(folder_waveforms)
    else:
        we = si.extract_waveforms(recording_clean, sorting, folder=folder_waveforms,
                                  overwrite=True, return_scaled=False, n_jobs=.7, sparse=False, progress_bar=True,
                                  max_spikes_per_unit=200)
    print(we)
    print('compute sparsity')
    sparsity = si.compute_sparsity(we, method='radius', radius_um=144)
    print('compute residuals')
    residual, convolved = compute_residuals(we, with_scaling=False, sparsity=sparsity)

    start_frame = int(200 * recording.sampling_frequency)
    end_frame = int(start_frame + (.1 * recording.sampling_frequency))

    vclean = recording_clean.get_traces(start_frame=start_frame, end_frame=end_frame).T
    vres = residual.get_traces(start_frame=start_frame, end_frame=end_frame).T
    vconvolved = convolved.get_traces(start_frame=start_frame, end_frame=end_frame).T
    np.save(PATH_SPIKE_INTERFACE / f"{pid}_original.npy", vclean)
    np.save(PATH_SPIKE_INTERFACE / f"{pid}_residual.npy", vres)
    np.save(PATH_SPIKE_INTERFACE / f"{pid}_spikes_model.npy", vconvolved)


