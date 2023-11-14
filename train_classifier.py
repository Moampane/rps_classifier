import pickle
import pandas as pd
import numpy as np
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from features import extract_feature, get_mf, get_pf, get_iemg, get_wl, get_rms, get_var

# train_file = "data/exampleEMGdata180trial_train.mat"
# train_file = "data/aditi_ian.mat"
train_file = "data/Unfiltered_Mos5.mat"

temp_load = scipy.io.loadmat(train_file)
label_load = scipy.io.loadmat("data/Unfiltered_Mos5_Ges.mat")
# labels = temp_load["labels"]
# labels = temp_load["label_names"]
labels = label_load[list(label_load.keys())[-1]]
labels = np.asarray(
    [num for sublist in labels for num in sublist]
)  # flatten because numpy flatten doesnt want to work

train_feature = extract_feature(train_file, [get_pf, get_mf])

# split training data for training accuracy
x_train, x_test, y_train, y_test = train_test_split(
    train_feature, labels, test_size=0.2, shuffle=True, stratify=labels
)

model = RandomForestClassifier()
# model = KNeighborsClassifier(n_neighbors=10)
# model = MLPClassifier(
#     hidden_layer_sizes=(81, 27, 9, 3),
#     activation="relu",
#     random_state=1,
#     solver="adam",
#     max_iter=200,
# )

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
