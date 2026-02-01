import numpy as np
from scipy import signal

from registry import augmentation_registry
from Augmentations.base_augmentation import BaseAugmentation, AugmentationRequest
from Augmentations.utils import log_range_random, normalise_addwave

def generate_gaussian_noise(
    nsta, ndp, samplerate, lowfreq_min, lowfreq_max, highfreq_min, highfreq_max
):
    gaussian = np.random.randn(nsta, 3, ndp)

    for i in range(nsta):
        lowfreq = log_range_random(lowfreq_min, lowfreq_max)
        highfreq = log_range_random(highfreq_min, highfreq_max)
        b, a = signal.butter(
            2, [lowfreq/(samplerate/2), highfreq/(samplerate/2)], btype='band'
        )
        gaussian[i] = signal.filtfilt(b, a, gaussian[i], axis=-1)

    return gaussian


def generate_lowfreq_noise(
    base_noise, samplerate, lowfreq, highfreq
):      
    cutoff = log_range_random(lowfreq, highfreq)
    
    lowfreq_noise = signal.filtfilt(
        *signal.butter(2, cutoff/(samplerate/2), btype='low'),
        base_noise,
        axis=-1
    )
    return lowfreq_noise


def generate_harmonic_noise(
    nsta, ndp, samplerate, lowfreq, highfreq,
    am_lowfreq, am_highfreq, am_amp_min, am_amp_max, am_amp_jitter, 
    phase_jitter, nharm_max, harm_logmean, harm_logsigma,
    vibr_ar1_coeff_min, vibr_ar1_coeff_max, phase_noise_strength
):
    base_freq = log_range_random(lowfreq, highfreq)
    t = np.arange(ndp)[None, None, :] / samplerate

    shared_phase = np.random.uniform(0, 2*np.pi)
    phase_offset = shared_phase + phase_jitter*np.random.randn(nsta, 1, 1)

    base = 2*np.pi*base_freq / samplerate
    ar1_coeff = np.random.uniform(vibr_ar1_coeff_min, vibr_ar1_coeff_max)
    phase_noise = np.random.randn(nsta, 1, ndp)
    phase_noise = phase_noise_strength * signal.lfilter(
        [1], [1, ar1_coeff], phase_noise, axis=-1
    )

    phase = np.cumsum(base + phase_noise, axis=-1) + phase_offset

    
    # Amplitude modulation
    am_noise = generate_lowfreq_noise(
        np.random.randn(nsta, 1, ndp), samplerate, am_lowfreq, am_highfreq
    )
    am_amp_global = log_range_random(am_amp_min, am_amp_max, (nsta,1,1))
    am_amp = am_amp_global * (1 + am_amp_jitter*np.random.randn(nsta,1,1))
    amp = 1 + am_amp*am_noise

    wave = np.sin(phase)

    # Harmonics
    nharm = np.random.randint(1, nharm_max)
    for h in range(2, nharm+1):
        w = np.random.lognormal(mean=harm_logmean, sigma=harm_logsigma) / h
        wave += w * np.sin(h*phase + np.random.uniform(0, 2*np.pi))

    wave = amp * wave

    # Coherency
    coh_strength = np.random.uniform(0.3, 1.0)
    gaussian = np.random.randn(nsta, 3, ndp) * 0.05
    coh = np.repeat(wave, 3, axis=1)

    return coh_strength*coh + (1-coh_strength)*gaussian


def generate_spikes(
    nsta, ndp, samplerate, min_nspk, max_nspk, spike_chance,
    oscil_dur_min, oscil_dur_max, oscil_freq_min, oscil_freq_max,
    oscil_tau_min, oscil_tau_max, aperi_dur_min, aperi_dur_max,
    aperi_tau_min, aperi_tau_max, oscil_chance
):

    spikes = np.zeros((nsta, 3, ndp))
    
    spike_mask = np.random.rand(nsta, 3) < spike_chance
    spike_counts = np.random.randint(min_nspk, max_nspk, size=(nsta, 3))

    stnidxs, chnlidxs = np.where(spike_mask)

    if len(stnidxs) == 0:
        return spikes

    counts = spike_counts[stnidxs, chnlidxs]
    total_spikes = counts.sum()

    if total_spikes == 0:
        return spikes

    spike_pos = np.random.randint(0, ndp, size=total_spikes)

    is_oscil = np.random.rand(total_spikes) < oscil_chance

    durlen_max = int(max(oscil_dur_max, aperi_dur_max) * samplerate)
    t = np.arange(durlen_max)[None, :] / samplerate

    spike_templates = np.zeros((total_spikes, durlen_max))
    spike_polarity = np.random.choice([-1, 1], size=(total_spikes, 1))
    

    if np.any(is_oscil):
        spikeidx = np.where(is_oscil)[0]

        oscil_dur = log_range_random(oscil_dur_min, oscil_dur_max, (len(spikeidx),1))
        oscil_tau = log_range_random(oscil_tau_min, oscil_tau_max, (len(spikeidx),1))
        oscil_freq = log_range_random(oscil_freq_min, oscil_freq_max, (len(spikeidx),1))
        
        rise_tau = oscil_tau * np.random.uniform(0.1, 0.5, size=oscil_tau.shape)
        
        t_local = t[:, :int(oscil_dur_max*samplerate)]

        rise = 1 - np.exp(-t_local / rise_tau)
        decay = np.exp(-t_local / oscil_tau)

        envelop = rise * decay
        envelop_noise = 0.2 * np.random.randn(*envelop.shape)
        b, a = signal.butter(1, 0.3)
        envelop_noise = signal.filtfilt(b, a, envelop_noise, axis=-1)
        envelop = envelop * (1 + envelop_noise)

        oscil = envelop * np.sin(2*np.pi*oscil_freq*t_local)
        oscil += np.random.randn(len(spikeidx), t_local.shape[1])*decay * np.random.random()
        oscil *= (t_local < oscil_dur)

        spike_templates[spikeidx, :t_local.shape[1]] = oscil


    if np.any(~is_oscil):
        spikeidx = np.where(~is_oscil)[0]

        aperi_dur = log_range_random(aperi_dur_min, aperi_dur_max, (len(spikeidx),1))
        aperi_tau = log_range_random(aperi_tau_min, aperi_tau_max, (len(spikeidx),1))

        t_local = t[:, :int(aperi_dur_max * samplerate)]

        decay = np.exp(-t_local / aperi_tau) * (t_local < aperi_dur)
        decay[:, 0] *= np.random.uniform(1, 2)       

        spike_templates[spikeidx, :t_local.shape[1]] = decay

    spike_templates *= spike_polarity

    time_offsets = np.arange(durlen_max)
    timeidxs = spike_pos[:, None] + time_offsets[None, :]

    valid_mask = timeidxs < ndp

    timeidxs = timeidxs[valid_mask]
    spike_vals = spike_templates[valid_mask]

    stn_expand = np.repeat(np.repeat(stnidxs, counts), durlen_max)[valid_mask.ravel()]
    chnl_expand = np.repeat(np.repeat(chnlidxs, counts), durlen_max)[valid_mask.ravel()]

    np.add.at(spikes, (stn_expand, chnl_expand, timeidxs), spike_vals)

    return spikes

    
@augmentation_registry.register("random_syn_noise")
class RandomSyntheticNoise(BaseAugmentation):
    '''
    Generates downhole synthetic noise to be superimposed 
    onto waveforms.
    
    Parameters
    --------------------------------------------
    noise_amp_min: float
        Minimum scaling factor on standardized noise trace 
        relative to original signal standard deviation per
        station
        
    noise_amp_max: float
        Maximum scaling factor on standardized noise trace 
        relative to original signal standard deviation per
        station    
    '''
    required_params = ["noise_amp_min", "noise_amp_max"]
    optional_params = {
        "gau_lowfreq_min": 0.5,    # min low cutoff frequency (Hz) for gaussian noise bandpass filter
        "gau_lowfreq_max": 5,    # max low cutoff frequency (Hz) for gaussian noise bandpass filter
        "gau_highfreq_min": 500,    # min high cutoff frequency (Hz) for gaussian noise bandpass filter
        "gau_highfreq_max": 1000,    # max high cutoff frequency (Hz) for gaussian noise bandpass filter
        "vibr_lowfreq": 20,    # min base frequency (Hz) for harmonic vibration noise
        "vibr_highfreq": 150,    # max base frequency (Hz) for harmonic vibration noise
        "gain_lowfreq": 0.05,    # min cutoff frequency (Hz) for low-frequency gain drift modulation
        "gain_highfreq": 0.5,    # max cutoff frequency (Hz) for low-frequency gain drift modulation
        
        "min_nspk": 3,    # int: min number of spikes per channel
        "max_nspk": 15,    # int: max number of spikes per channel
        
        "spike_chance": 0.3,    # probability [0-1] per channel of containing spike events
        "flatline_chance": 0.3,    # probability [0-1] per station to contain no synthetic noise
        "humming_chance": 0.5,    # probability [0-1] for all to contain coherent vibration noise
        "gain_drift_chance": 0.3,    # probability [0-1] for all to contain multiplicative gain drift
        "oscil_chance": 0.8,    # probability [0-1] for a spike to be oscillatory vs simple decay
        
        "gaussian_amp_max": 0.5,    # max amplitude scaling for band-limited gaussian noise
        "vibration_amp_max": 0.5,    # max amplitude scaling for vibration noise
        "drift_amp_max": 0.5,    # max amplitude scaling for baseline drift
        "gain_amp_max": 2.0,    # max multiplicative gain drift scaling factor
        "spike_amp_max": 2.0,    # max amplitude scaling for spike noise
        
        "gaussian_amp_min": 0.05,    # min amplitude scaling for band-limited gaussian noise
        "vibration_amp_min": 0.05,    # min amplitude scaling for vibration noise
        "drift_amp_min": 0.01,    # min amplitude scaling for  baseline drift
        "gain_amp_min": 0.05,    # min multiplicative gain drift scaling factor
        "spike_amp_min": 0.05,    # min amplitude scaling for spike noise
        
        "drift_ar1_coeff_min": 0.95,    # min AR(1) coefficient [0-1] for drift correlation length
        "drift_ar1_coeff_max": 0.999,    # max AR(1) coefficient [0-1] for drift correlation length
        "vibr_ar1_coeff_min": 0.98,    # min AR(1) coefficient [0-1] for phase noise correlation
        "vibr_ar1_coeff_max": 0.999,    # max AR(1) coefficient [0-1] for phase noise correlation
        
        "am_lowfreq": 0.1,    # min amplitude modulation frequency (Hz) for vibration
        "am_highfreq": 80,    # max amplitude modulation frequency (Hz) for vibration
        "am_amp_min": 0.02,    # min global amplitude of amplitude modulation component
        "am_amp_max": 0.2,    # max global amplitude of amplitude modulation component
        "am_amp_jitter": 0.5,    # random jitter factor to AM amplitude per station
        "phase_jitter": 1.0,    # random phase perturbation (radians) per station for vibration
        "nharm_max": 4,    # int: max number of harmonics added to base vibration
        "harm_logmean": -1.0,    # log-normal mean for harmonic amplitude decay
        "harm_logsigma": 0.4,    # log-normal spread for harmonic variability
        "phase_noise_strength": 0.001,    # phase diffusion intensity
        
        "oscil_dur_min": 0.01,    # min oscillatory transient noise duration (seconds)
        "oscil_dur_max": 0.1,    # max oscillatory transient noise duration (seconds)
        "oscil_freq_min": 200,    # min oscillatory transient noise frequency (Hz)
        "oscil_freq_max": 600,    # max oscillatory transient noise frequency (Hz)
        "oscil_tau_min": 0.002,    # min oscillatory transient noise exponential decay constant (seconds) 
        "oscil_tau_max": 0.01,    # max oscillatory transient noise exponential decay constant (seconds) 
        "aperi_dur_min": 0.05,    # min aperiodic transient noise duration (seconds)
        "aperi_dur_max": 0.4,    # max aperiodic transient noise duration (seconds)
        "aperi_tau_min": 0.02,    # min aperiodic transient noise exponential decay constant (seconds) 
        "aperi_tau_max": 0.1    # max aperiodic transient noise exponential decay constant (seconds) 
    }
    scope = "raw"
    
    def __init__(self, param_dict):
        super().__init__(param_dict)

    def augment_raw(self, waves_all, eventdf, samplerate=None):
        waves_all_aug = waves_all.copy()
        
        if np.random.random() < self.augment_chance:
            nsta, _, ndp = waves_all.shape
            if samplerate is None:
                raw_samplerate = eventdf['trace_sampling_rate_hz'].iloc[0]
            else:
                raw_samplerate = samplerate
            
            gaussian = generate_gaussian_noise(
                nsta, ndp, raw_samplerate, self.gau_lowfreq_min, self.gau_lowfreq_max,
                self.gau_highfreq_min, self.gau_highfreq_max
            )
            gaussian_amp = log_range_random(self.gaussian_amp_min, self.gaussian_amp_max, (nsta,1,1))
            gaussian *= gaussian_amp

            vibration = generate_harmonic_noise(
                nsta, ndp, raw_samplerate, self.vibr_lowfreq, self.vibr_highfreq,
                self.am_lowfreq, self.am_highfreq, self.am_amp_min, self.am_amp_max,
                self.am_amp_jitter, self.phase_jitter, 
                self.nharm_max, self.harm_logmean, self.harm_logsigma,
                self.vibr_ar1_coeff_min, self.vibr_ar1_coeff_max, self.phase_noise_strength
            )
            vibration_amp = log_range_random(self.vibration_amp_min, self.vibration_amp_max, (nsta,1,1))
            vibration *= vibration_amp

            spikes = generate_spikes(
                nsta, ndp, raw_samplerate, self.min_nspk, self.max_nspk, self.spike_chance,
                self.oscil_dur_min, self.oscil_dur_max, self.oscil_freq_min, self.oscil_freq_max,
                self.oscil_tau_min, self.oscil_tau_max, self.aperi_dur_min, self.aperi_dur_max,
                self.aperi_tau_min, self.aperi_tau_max, self.oscil_chance
            )
            spike_amp = log_range_random(self.spike_amp_min, self.spike_amp_max, (nsta,1,1))
            spikes *= spike_amp
            
            self.logger.debug(
                "noise array shapes (gaussian, vibration, spikes): "
                f"{gaussian.shape, vibration.shape, spikes.shape}"
            )
            
            
            ### Combining noise
            noise = gaussian
   
            if np.random.random() < self.humming_chance:
                noise += vibration
            
            # Baseline drift
            ar1_coeff = np.random.uniform(self.drift_ar1_coeff_min, self.drift_ar1_coeff_max)
            white = np.random.randn(nsta, 3, ndp)
            drift = signal.lfilter([1], [1, -ar1_coeff], white, axis=-1)
            drift /= (np.std(drift, axis=-1, keepdims=True) + 1e-8)
            drift_amp = log_range_random(self.drift_amp_min, self.drift_amp_max, (nsta,1,1))
            noise += drift_amp*drift

            # Gain drift
            if np.random.random() < self.gain_drift_chance:
                base_noise = np.random.randn(1, 1, ndp)
                gain = generate_lowfreq_noise(
                    base_noise, raw_samplerate, self.gain_lowfreq, self.gain_highfreq
                )
                gain_amp = log_range_random(self.gain_amp_min, self.gain_amp_max)
                noise = noise * (1 + gain_amp*gain)
                
            noise += spikes
            
            flat_mask = np.random.rand(nsta) < self.flatline_chance
            noise[flat_mask] = 0
        
            noise = normalise_addwave(waves_all, noise, self.noise_amp_min, self.noise_amp_max)
            waves_all_aug += noise

        return waves_all_aug, samplerate
