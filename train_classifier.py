import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_file = "big_chunker.csv"
train_data = pd.read_csv(train_file)

labels = np.asarray(train_data.pop("labels"))

# get best features
# make mf features
mf_headers = [f"MF ch {num}" for num in [x + 1 for x in range(4)]]

# make pf features
pf_headers = [f"PF ch {num}" for num in [x + 1 for x in range(4)]]

# make best train features
train_feature = pd.concat(
    [train_data[mf_headers], train_data[pf_headers]], axis=1, join="inner"
)

# make features arrays
train_feature = np.asarray(train_feature)

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
