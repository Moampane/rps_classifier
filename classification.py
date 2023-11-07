import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

model_dict = pickle.load(open("rfc_model.p", "rb"))
model = model_dict["model"]

test_file = "Mo1_features.csv"
test_data = pd.read_csv(test_file)

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

# get accuracies
test_acc = accuracy_score(test_labels, test_predict)

print(f"{test_acc*100}% of samples were classified correctly!")
