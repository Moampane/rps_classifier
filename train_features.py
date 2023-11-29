import scipy
import numpy as np
import pandas as pd


# extract signals from data for to train classifier
def extract_signals(file_name):
    mat = scipy.io.loadmat(file_name)
    try:
        signal_volts = mat["dataChTimeTr"]
    except:
        signal_volts = mat[list(mat.keys())[-1]]

    # FS = 1000 # measuring frequency
    NUM_CH = 4  # number of channels

    # dictionary of lists of signals
    all_ch_signals = []

    for ch in range(NUM_CH):
        trials = [
            idx for idx in range(signal_volts.shape[-1])
        ]  # makes list of indices for voltages of a signal
        ch_signals = [
            [v[tr - 1] for v in signal_volts[ch]] for tr in trials
        ]  # makes list of signals
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


# training feature extraction
def extract_feature(file_name, func: list):
    signals = extract_signals(file_name)

    features = [[[f(trial) for trial in ch] for ch in signals] for f in func]

    features = np.transpose(np.concatenate(features))

    return features
