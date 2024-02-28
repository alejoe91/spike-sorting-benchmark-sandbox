import spikeinterface as si
from spikeinterface.core import BaseRecording
from spikeinterface.core.generate import InjectTemplatesRecording
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost


def compute_residuals(waveform_extractor, with_scaling=True, overwrite=False, **scaling_kwargs):
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
        # TODO: apply reverse scaling
        pass
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


def peak_detection_sweep(residuals, thresholds, noise_levels=None, peak_sign="both",
                         n_jobs=-1):
    """
    Perform a peak detection sweep on a given residual recording.

    Parameters
    ----------
    residuals: BaseRecording
        The residual recording
    thresholds: list
        The list of thresholds to use for peak detection
    noise_levels: np.array or None
        The noise levels to use for peak detection. If None, they are computed from the residual recording
    peak_sign: str, default: "both"
        The peak sign to use for peak detection

    Returns
    -------
    peak_counts: dict
        The number of detected peaks for each threshold normalized by the duration and number of channels
    peak_list: dict
        The list of detected peaks for each threshold
    """
    from spikeinterface.core.node_pipeline import run_node_pipeline
    from spikeinterface.sortingcomponents.peak_detection import DetectPeakLocallyExclusive

    assert isinstance(residuals, BaseRecording)
    nodes = []
    if noise_levels is None:
        noise_levels = si.get_noise_levels(residuals)
    for th in thresholds:
        node = DetectPeakLocallyExclusive(
            recording=residuals, peak_sign=peak_sign, detect_threshold=th, noise_levels=noise_levels
        )
        nodes.append(node)

    # run the pipeline
    outs = run_node_pipeline(
        residuals,
        nodes=nodes,
        job_name=f"detect peaks with {len(thresholds)} thresholds",
        job_kwargs=dict(n_jobs=n_jobs, progress_bar=True)
    )

    # peaks
    peak_counts = dict()
    peaks = dict()
    for th, out in zip(thresholds, outs):
        peak_counts[th] = len(out) / residuals.get_num_channels() / residuals.get_total_duration()
        peaks[th] = out

    return peak_counts, peaks
