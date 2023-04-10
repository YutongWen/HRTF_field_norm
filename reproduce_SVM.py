import sys
assert sys.version_info >= (3, 5)

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
import torch
from tqdm import tqdm
import pickle as pkl
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from HRTFdatasets import MergedHRTFDataset
from SOFAdatasets import ARI, HUTUBS, Prin3D3A, Listen, RIEC, BiLi, Crossmod, SADIE

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix

### For reproducing https://arxiv.org/pdf/2212.04283.pdf

def intersection(lst1, lst2):
    if lst2 == []:
        return lst1
    return list(set(lst1) & set(lst2))

common_location = []
for dataset_func in [ARI, HUTUBS, Prin3D3A, RIEC, BiLi, Listen, Crossmod, SADIE]:  # ARI, HUTUBS, ITA, CIPIC, Prin3D3A, RIEC, BiLi, Listen,
    dataset = dataset_func()
    print(dataset.name)
    loc = dataset.locations
    common_location = intersection(loc, common_location)
    print(common_location)


all_names = ["ari", "hutubs", "3d3a", "bili", "listen", "crossmod", "sadie", "riec"]
dataset = MergedHRTFDataset(all_names, "all", "log", norm_way=2)

X_all = []
y_all = []
for item in dataset:
    loc, hrtf, name = item
    index = np.where((loc[:, 1] == 0) & np.isin(loc[:, 0], [60]))[0] # , 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
#     if hrtf[index, :].flatten().numpy().shape[0] != 1104:
#         print(name)
    X_all.append(hrtf[index, :].float().flatten().numpy())
    y_all.append(all_names.index(name))
X_all = np.stack(X_all)
y_all = np.array(y_all)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

svm_clf = LinearSVC(C=100, random_state=42) # SVC(kernel='rbf', gamma=0.5, C=1)
scaler = StandardScaler()
scaled_svm_clf = Pipeline([
        ("scaler", scaler),
        ("svc", svm_clf),
    ])
# svm_clf.fit(X_train, y_train)
scaled_svm_clf.fit(X_train, y_train)

y_train_predict = scaled_svm_clf.predict(X_train)
print(confusion_matrix(y_train, y_train_predict))

y_test_predict = scaled_svm_clf.predict(X_test)
print(confusion_matrix(y_test, y_test_predict))


'''

X_all = []
y_all = []

for dataset_func in [ARI, HUTUBS, Prin3D3A, RIEC, BiLi, Listen, Crossmod, SADIE]:  # ARI, HUTUBS, ITA, CIPIC, Prin3D3A, RIEC, BiLi, Listen,
    dataset = dataset_func()
    print(dataset.name)
    for i in tqdm(range(len(dataset))):
#         idx, ear, loc, irs = item
        with open(os.path.join("/data2/neil/HRTF/prepocessed_hrirs", "%s_%03d.pkl" % (dataset.name, i)), 'rb') as handle:
            loc, irs = pkl.load(handle)
        index = np.where((loc[:, 1] == 0) & np.isin(loc[:, 0], [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]))[0]
        hrtf = np.abs(np.fft.fft(irs[index, :], n=256))
        hrtf = hrtf[1:128]
        X_all.append(hrtf.flatten())
        y_all.append(all_names.index(dataset.name))
X_all = np.stack(X_all)
y_all = np.array(y_all)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

svm_clf = LinearSVC(C=100, random_state=42)# SVC(kernel='rbf', gamma=0.5, C=0.8)
scaler = StandardScaler()
scaled_svm_clf = Pipeline([
        ("scaler", scaler),
        ("svc", svm_clf),
    ])
# svm_clf.fit(X_train, y_train)
scaled_svm_clf.fit(X_train, y_train)

y_train_predict = scaled_svm_clf.predict(X_train)
print(confusion_matrix(y_train, y_train_predict))

y_test_predict = scaled_svm_clf.predict(X_test)
print(confusion_matrix(y_test, y_test_predict))

'''
