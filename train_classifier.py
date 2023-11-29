"""
Creates the rps classifier by training on data input as the file path in the variable train_file.
"""

import pickle
import numpy as np
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from train_features import (
    extract_feature,
    get_mf,
    get_pf,
)

train_file = "data/Unfiltered_Mos5.mat"

label_load = scipy.io.loadmat("data/Unfiltered_Mos5_Ges.mat")
labels = label_load[list(label_load.keys())[-1]]
labels = np.asarray([num for sublist in labels for num in sublist])

# post processing
two_idxs = [idx for idx in range(len(labels)) if labels[idx] == 2]
three_idxs = [idx for idx in range(len(labels)) if labels[idx] == 3]
labels[two_idxs] = 3
labels[three_idxs] = 2

train_feature = extract_feature(train_file, [get_pf, get_mf])

# split training data for training accuracy
x_train, x_test, y_train, y_test = train_test_split(
    train_feature, labels, test_size=0.2, shuffle=True, stratify=labels
)

model = RandomForestClassifier()

# fit model
model.fit(x_train, y_train)

# make predictions
train_predict = model.predict(x_test)

# get accuracies
train_acc = accuracy_score(y_test, train_predict)

print(f"{train_acc*100}% of samples were classified correctly!")

f = open("rfc_model.p", "wb")
pickle.dump({"model": model}, f)
f.close()
