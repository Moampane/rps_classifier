"""
Helper functions for live feature extraction.
"""

import numpy as np


# extract signals from data for live classification
def extract_signals(data):
    signal_volts = data

    NUM_CH = 4  # number of channels

    # list of lists of signals
    all_ch_signals = []

    for ch in range(NUM_CH):
        ch_signals = [voltage[ch] for voltage in signal_volts]
        all_ch_signals.append(ch_signals)

    return all_ch_signals


# feature extraction functions
# get root mean square from list of signal data
def get_rms(data):
    return np.sqrt(sum([x * x for x in data]) / len(data))


# get waveform length from list of signal data
def get_wl(data):
    return sum([abs(data[idx + 1] - data[idx]) for idx in range(len(data) - 1)]) / len(
        data
    )


# get variance from list of signal data
def get_var(data):
    return np.var(data)


# get integrated EMG from list of signal data
def get_iemg(data):
    return sum([abs(x) for x in data])


# get mean frequency from list of signal data
def get_mf(data):
    data = np.array(data)
    fft = [abs(x) for x in np.fft.fft(data).real]
    freqs = [abs(x) for x in np.fft.fftfreq(data.shape[-1])]

    return np.average(freqs, weights=fft)


# get peak frequency from list of signal data
def get_pf(data):
    data = np.array(data)
    fft = [abs(x) for x in np.fft.fft(data).real]
    freqs = np.fft.fftfreq(data.shape[-1])
    return abs(freqs[np.where(fft == max(fft[1:]))[0]][0])


# live classification feature extraction
def extract_feature(data, func: list):
    ex_signals = extract_signals(data)

    features = [[f(signal) for signal in ex_signals] for f in func]

    features = np.transpose(np.concatenate(features))

    return features
