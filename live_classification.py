from classification import live_classify
from features import extract_feature, get_mf, get_pf, get_wl
from sklearn.metrics import accuracy_score
import scipy
import numpy as np

# live_file = "data/exampleEMGdata120trial_test.mat"
live_file = "data/dataMo1.mat"
model = "rfc_model.p"

temp_load = scipy.io.loadmat(live_file)
# labels = temp_load["labels"]  # remove for actual live classification
label_load = scipy.io.loadmat("data/labelsMo1.mat")
labels = label_load[list(label_load.keys())[-1]]
labels = np.asarray(
    [num for sublist in labels for num in sublist]
)  # flatten because numpy flatten doesnt want to work laso remove for actual live classification

live_features = extract_feature(file_name=live_file, func=[get_pf, get_mf])

predictions = live_classify(model=model, features=live_features)
print(predictions)

print(accuracy_score(labels, predictions))  # remove for actual live classification
