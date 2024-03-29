# %%
from pathlib import Path
import numpy as np

from one.api import ONE

from brainbox.io.one import SpikeSortingLoader
from ibldsp.voltage import destripe
from viewephys.gui import viewephys

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
# [600, 1200]

import spikeglx
for pid in benchmark_pids:
    cbin_file = PATH_CBIN.joinpath(f'{pid}.ap.cbin')
    ssl = SpikeSortingLoader(one=one, pid=pid)
    sr = spikeglx.Reader(cbin_file)
    break