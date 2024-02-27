import spikeinterface.full as si


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
            amp_scalings = si.compute_amplitude_scalings(waveform_extractor, **scaling_kwargs)
    else:
        amp_scalings = None

    templates = waveform_extractor.get_all_templates()

    recording_scaled = si.scale(
        waveform_extractor.recording,
        gain=waveform_extractor.recording.get_channel_gains(),
        dtype="float32"
    )

    residuals = si.InjectTemplatesRecording(
        sorting=waveform_extractor.sorting,
        parent_recording=recording_scaled,
        amplitude_factor=amp_scalings,
        templates=-templates
    )

    return residuals, amp_scalings
