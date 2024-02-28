



^C^C^CTraceback (most recent call last):
  File "/home/olivier/Documents/PYTHON/envs/iblenv/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3526, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-1-31dd49ef4879>", line 1, in <module>
    sparsity = si.compute_sparsity(we, method='radius', radius_um=144)
  File "/home/olivier/Documents/PYTHON/00_IBL/spikeinterface/src/spikeinterface/core/sparsity.py", line 428, in compute_sparsity
    sparsity = ChannelSparsity.from_radius(templates_or_waveform_extractor, radius_um, peak_sign=peak_sign)
  File "/home/olivier/Documents/PYTHON/00_IBL/spikeinterface/src/spikeinterface/core/sparsity.py", line 297, in from_radius
    best_chan = get_template_extremum_channel(templates_or_we, peak_sign=peak_sign, outputs="index")
  File "/home/olivier/Documents/PYTHON/00_IBL/spikeinterface/src/spikeinterface/core/template_tools.py", line 112, in get_template_extremum_channel
    peak_values = get_template_amplitudes(templates_or_waveform_extractor, peak_sign=peak_sign, mode=mode)
  File "/home/olivier/Documents/PYTHON/00_IBL/spikeinterface/src/spikeinterface/core/template_tools.py", line 52, in get_template_amplitudes
    templates_array = _get_dense_templates_array(templates_or_waveform_extractor)
  File "/home/olivier/Documents/PYTHON/00_IBL/spikeinterface/src/spikeinterface/core/template_tools.py", line 15, in _get_dense_templates_array
    templates_array = templates_or_waveform_extractor.get_all_templates(mode="average")
  File "/home/olivier/Documents/PYTHON/00_IBL/spikeinterface/src/spikeinterface/core/waveform_extractor.py", line 1270, in get_all_templates
    self.precompute_templates(modes=[mode], percentile=percentile)
  File "/home/olivier/Documents/PYTHON/00_IBL/spikeinterface/src/spikeinterface/core/waveform_extractor.py", line 1232, in precompute_templates
    arr = np.average(wfs, axis=0)
  File "/home/olivier/Documents/PYTHON/envs/iblenv/lib/python3.10/site-packages/numpy/lib/function_base.py", line 520, in average
    avg = a.mean(axis, **keepdims_kw)
  File "/home/olivier/Documents/PYTHON/envs/iblenv/lib/python3.10/site-packages/numpy/core/_methods.py", line 118, in _mean
    ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)
  File "/home/olivier/Documents/PYTHON/envs/iblenv/lib/python3.10/site-packages/numpy/core/memmap.py", line 319, in __array_wrap__
    def __array_wrap__(self, arr, context=None):
KeyboardInterrupt

