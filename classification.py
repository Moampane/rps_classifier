import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


def classify(model, test_data):
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

    test_data = pd.read_csv(test_data)

    # Comment out for live classification?
    test_labels = np.asarray(test_data.pop("labels"))

    # get best features
    # make mf features
    mf_headers = [f"MF ch {num}" for num in [x + 1 for x in range(4)]]

    # make pf features
    pf_headers = [f"PF ch {num}" for num in [x + 1 for x in range(4)]]

    # make best test features
    test_feature = pd.concat(
        [test_data[mf_headers], test_data[pf_headers]], axis=1, join="inner"
    )

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

    # get accuracies
    # test_acc = accuracy_score(test_labels, test_predict)

    # print(f"{test_acc*100}% of samples were classified correctly!")


cur_model = "rfc_model.p"
cur_data = "KD1_features.csv"

print(classify(model=cur_model, test_data=cur_data))
