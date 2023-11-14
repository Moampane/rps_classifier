import pickle
from classification import live_classify
from features import extract_feature, get_mf, get_pf, get_wl
from sklearn.metrics import accuracy_score
import scipy
import numpy as np


class RunPythonModel:
    def __init__(self, modelPath):
        self.model = (
            "rfc_model.p"  # replace this with whatever code you need to load model
        )

    def get_rps(self, data):
        """
        Function to take in data and return the rps. You can
        place this function wherever you want, but add code here that
        takes in the data and returns rock, paper, or scissors
        after putting the data through your model.
        """

        use_data = [row[1:5] for row in data]

        live_features = extract_feature(data=use_data, func=[get_pf, get_mf])

        print(live_features)

        predictions = live_classify(model=self.model, features=live_features)

        return predictions
