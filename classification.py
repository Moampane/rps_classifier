"""
Helper functions for classification.
"""

import pickle
import numpy as np


def classify_labeled(model, test_data):
    """
    Classify a set of features as either rock, paper, or scissors.

    arg(s):
        model(string): The file path to a model trained by train_classifier.py.
        test_data(string): The file path to a set of features extracted from EMG signals.

    return(s):
        A list of values representing whether a player threw rock, paper, or scissors
        based on the features extracted from their EMG signals.

    """

    rps_labels = ["rock", "paper", "scissors"]

    model_dict = pickle.load(open(model, "rb"))
    model = model_dict["model"]

    test_labels = np.asarray(test_data.pop("labels"))

    # get best features
    # make mf features
    mf_headers = [f"MF ch {num}" for num in [x + 1 for x in range(4)]]

    # make pf features
    pf_headers = [f"PF ch {num}" for num in [x + 1 for x in range(4)]]

    # make features arrays
    test_feature = np.asarray(test_feature)

    # make predictions
    test_predict = model.predict(test_feature)

    # Print prediction and actual for labeled data
    [
        print(
            f"You threw {rps_labels[test_labels[idx]]}, we predicted {rps_labels[test_predict[idx]]}"
        )
        for idx in range(len(test_predict))
    ]

    return test_predict


def live_classify(model, features):
    """
    Classify a set of signal(s) as either rock, paper, or scissors.

    arg(s):
        model(string): The file path to a model trained by train_classifier.py.
        features(numpy array): Numpy array representing the extracted features of EMG data (shape is trials x features)

    return(s):
        A list of values representing whether a player threw rock, paper, or scissors
        based on the EMG signal(s).

    """

    rps_labels = ["rock", "paper", "scissors"]

    model_dict = pickle.load(open(model, "rb"))
    model = model_dict["model"]

    # make predictions
    test_predict = model.predict(features.reshape(1, -1))

    # Print prediction and actual for labeled data
    [print(f"We predicted {rps_labels[int(idx-1)]}") for idx in test_predict]

    return int(test_predict)
