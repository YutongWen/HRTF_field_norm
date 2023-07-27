#%%
import sys
assert sys.version_info >= (3, 5)
from matplotlib import pyplot as plt
from pandas import DataFrame
import seaborn as sn

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.model_selection import train_test_split

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from HRTFdatasets import MergedHRTFDataset

# from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

all_names = ["ari", "hutubs", "3d3a", "bili", "listen", "crossmod", "sadie", "riec"]
dataset = MergedHRTFDataset(all_names, "all", "log", norm_way=6)

hidden_shape = [16]

X_all = []
y_all = []
for item in dataset:
    loc, hrtf, _, name = item
    index = np.where((loc[:, 1] == 0) & np.isin(loc[:, 0], [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]))[0] # , 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
    hrtf = hrtf[index,:]
    X_all.append(hrtf.float().flatten().numpy())
    y_all.append(all_names.index(name))
X_all = np.stack(X_all)
y_all = np.array(y_all)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

classifier = MLPClassifier(hidden_layer_sizes=hidden_shape, activation='tanh', max_iter=10000)

classifier.fit(X_train, y_train)

y_train_predict = classifier.predict(X_train)
print(confusion_matrix(y_train, y_train_predict))

y_test_predict = classifier.predict(X_test)
print(confusion_matrix(y_test, y_test_predict))
matrix = confusion_matrix(y_test, y_test_predict)

fig1, ax = plt.subplots()
fig1.suptitle(f'Confusion Matrix for Normalized HRTF at Different Locations Using MLP with hidden shape {hidden_shape}')
df_cm = DataFrame(matrix, index=all_names, columns=all_names)
sn.heatmap(df_cm, cmap='Oranges', annot=True, ax=ax)

fig2, axs = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(16, 12))
fig2.suptitle(f'Confusion Matrix for Normalized HRTF at Different Locations Using MLP with hidden shape {hidden_shape} ')

# train for twelve different locations
azimuth_id = 0
for azimuth, ax in zip([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], axs.flat):
    X_all = []
    y_all = []
    # azimuth = 30
    for indx in range(0, len(dataset), 2):
        item = dataset[indx]
        loc, hrtf, _, name = item # hrtf is 20 log10
        index = np.where((loc[:, 1] == 0) & np.isin(loc[:, 0], [azimuth]))[0]
        X_all.append(hrtf[index, :].float().flatten().numpy())
        y_all.append(all_names.index(name))
    X_all = np.stack(X_all)
    y_all = np.array(y_all)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    
    classifier = MLPClassifier(hidden_layer_sizes=hidden_shape, activation='tanh', max_iter=10000)
    # svm_clf.fit(X_train, y_train)
    classifier.fit(X_train, y_train)
    print(f'svm trained using loc ({azimuth}, 0)')
    y_train_predict = classifier.predict(X_train)
    print(confusion_matrix(y_train, y_train_predict))

    y_test_predict = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_test_predict))
    matrix_linear = confusion_matrix(y_test, y_test_predict)
    
    df_cm_linear = DataFrame(matrix_linear, index=all_names, columns=all_names)
    ax.set(title=f'location({azimuth}, 0)')
    sn.heatmap(df_cm_linear, cmap='Blues', annot=True, ax=ax)
    
    azimuth_id += 1