import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# extract signals from specific data channel
def extract_signal(data, lab_in, label_conv, ch_in):
    ch = ch_in
    all_cond_signals = (
        {}
    )  # dictionary of signals, key is condition, value is list of signals for that condition from the dataset
    labnam_in = np.asarray([[1], [2], [3]])

    for cond in labnam_in:
        trials = [
            idx - 1 for idx in range(len(lab_in)) if lab_in[idx - 1] == cond[0]
        ]  # makes list of indices of all trials of a specific condition
        cond_signals = [
            [v[tr - 1] for v in data[ch - 1]] for tr in trials
        ]  # makes list of signals from each trial of a specificed condition
        all_cond_signals[label_conv[cond[0] - 1]] = cond_signals

    return all_cond_signals


# feature extraction functions
# get root mean square from list of signal data
def get_rms(data, num_timesteps):
    return np.sqrt(sum([x * x for x in data]) / num_timesteps)


# get waveform length from list of signal data
def get_wl(data, num_timesteps):
    return (
        sum([abs(data[idx + 1] - data[idx]) for idx in range(len(data) - 1)])
        / num_timesteps
    )


# get variance from list of signal data
def get_var(data, num_timesteps):
    return np.var(data)


# get integrated EMG from list of signal data
def get_iemg(data, num_timesteps):
    return sum([abs(x) for x in data])


# get mean frequency from list of signal data
def get_mf(data, num_timesteps):
    data = np.array(data)
    fft = [abs(x) for x in np.fft.fft(data).real]
    freqs = [abs(x) for x in np.fft.fftfreq(data.shape[-1])]

    return np.average(freqs, weights=fft)


# get peak frequency from list of signal data
def get_pf(data, num_timesteps):
    data = np.array(data)
    fft = [abs(x) for x in np.fft.fft(data).real]
    freqs = np.fft.fftfreq(data.shape[-1])
    return abs(freqs[np.where(fft == max(fft[1:]))[0]][0])


# function for extracting features of each condition from a dictionary of (condition, list of signals) pairs
def extract_feature_condition(data, lab_in, label_conv, ch_in, num_timesteps, func):
    all_cond_signal = extract_signal(
        data, lab_in, label_conv, ch_in
    )  # dictionary of signals, key is condition, value is list of signals for that condition from the dataset
    cond_features = {
        cond: [func(signal, num_timesteps) for signal in all_cond_signal[cond]]
        for cond in all_cond_signal
    }  # dictionary of features, key is condition, value is list of features for that condition
    return cond_features


def prep_data_feature(data, lab_in, label_conv, num_timesteps, func):
    numCh = 4
    values = {
        f"{func.__name__[4:].upper()} ch {ch_num}": extract_feature_condition(
            data, lab_in, label_conv, ch_num, num_timesteps, func
        )
        for ch_num in range(numCh + 1)[1:]
    }
    columns = [
        [
            val
            for sublist in [values[ch][cond] for cond in values[ch]]
            for val in sublist
        ]
        for ch in values
    ]

    return values.keys(), columns


def make_feature_table(
    data_file, label_file, num_rocks, num_papers, num_scissors, file_name
):
    # load data from mat files
    dataset = scipy.io.loadmat(data_file)  # dataChTimeTr
    labels = scipy.io.loadmat(label_file)  # labels

    # extract values from data
    feature_data = dataset[list(dataset.keys())[-1]]
    feature_data_dim = dataset[list(dataset.keys())[-1]].shape
    feature_num_timesteps = feature_data_dim[1]

    # extract labels / what the data actually is from label data
    feature_labels = labels[list(labels.keys())[-1]]

    # make list to convert 0, 1, 2 to rock, paper, scissors
    conversion = ["rock", "paper", "scissors"]

    # extract features from data
    rms_headers, rms_columns = prep_data_feature(
        feature_data, feature_labels, conversion, feature_num_timesteps, get_rms
    )
    wl_headers, wl_columns = prep_data_feature(
        feature_data, feature_labels, conversion, feature_num_timesteps, get_wl
    )
    var_headers, var_columns = prep_data_feature(
        feature_data, feature_labels, conversion, feature_num_timesteps, get_var
    )
    iemg_headers, iemg_columns = prep_data_feature(
        feature_data, feature_labels, conversion, feature_num_timesteps, get_iemg
    )
    mf_headers, mf_columns = prep_data_feature(
        feature_data, feature_labels, conversion, feature_num_timesteps, get_mf
    )
    pf_headers, pf_columns = prep_data_feature(
        feature_data, feature_labels, conversion, feature_num_timesteps, get_pf
    )

    # make labels for corresponding data
    labels_column = [
        *list(np.zeros(num_rocks, dtype=int)),
        *list(np.ones(num_papers, dtype=int)),
        *list(2 * np.ones(num_scissors, dtype=int)),
    ]

    # make headers for data
    headers = [
        *list(rms_headers),
        *list(wl_headers),
        *list(var_headers),
        *list(iemg_headers),
        *list(mf_headers),
        *list(pf_headers),
        *["labels"],
    ]

    # put extracted features into a dataframe
    feature_data = pd.DataFrame(
        list(
            zip(
                rms_columns[0],
                rms_columns[1],
                rms_columns[2],
                rms_columns[3],
                wl_columns[0],
                wl_columns[1],
                wl_columns[2],
                wl_columns[3],
                var_columns[0],
                var_columns[1],
                var_columns[2],
                var_columns[3],
                iemg_columns[0],
                iemg_columns[1],
                iemg_columns[2],
                iemg_columns[3],
                mf_columns[0],
                mf_columns[1],
                mf_columns[2],
                mf_columns[3],
                pf_columns[0],
                pf_columns[1],
                pf_columns[2],
                pf_columns[3],
                labels_column,
            )
        ),
        columns=headers,
    )

    # name and export output file
    # labels_idx = label_file.index('labels') + len('labels')
    # end_idx = label_file.index('.')
    # name = label_file[labels_idx:end_idx]

    # feature_data.to_csv(f'{name}_features.csv', index=False)
    feature_data.to_csv(f"{file_name}.csv", index=False)
