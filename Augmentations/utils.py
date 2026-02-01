import numpy as np

def log_range_random(low, high, size=1):
    return 10**np.random.uniform(np.log10(low), np.log10(high), size=size)


def normalise_addwave(waves_orig, waves_temp, ampmin, ampmax):
    amp = log_range_random(ampmin, ampmax)
    local_std = np.std(waves_orig, axis=-1, keepdims=True)
    global_std = np.std(waves_orig)
    signal_ref = np.where(local_std < 1e-8, global_std, local_std)
    waves_temp = waves_temp / (np.std(waves_temp, axis=-1, keepdims=True) + 1e-8)

    waves_temp *= amp * signal_ref
    return waves_temp