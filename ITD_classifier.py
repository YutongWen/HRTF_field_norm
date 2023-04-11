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

scale_choice = {'linear':0, 'log':1}
compute_choice = {'multi':0, 'add':1}
train_test_is_same = False
norm_to_zero_one = True
scale = 'linear'
compute = 'multi'


all_names = ["ari", "hutubs", "3d3a", "bili", "listen", "crossmod", "sadie", "riec"]
dataset = MergedHRTFDataset(all_names, "all", "log", norm_way=-1)


X_all = []
y_all = []
for item in dataset:
    loc, _, ITD_array, name = item
    index = np.where((loc[:, 1] == 0) & np.isin(loc[:, 0], [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]))[0] # , 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
#     if hrtf[index, :].flatten().numpy().shape[0] != 1104:
#         print(name)
    
    X_all.append(ITD_array[index])
    y_all.append(all_names.index(name))
X_all = np.stack(X_all)
y_all = np.array(y_all)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

svm_clf = LinearSVC(C=100, random_state=42) # SVC(kernel='rbf', gamma=0.5, C=1)
# svm_clf = SVC(kernel='rbf', gamma=0.5, C=1)
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
matrix = confusion_matrix(y_test, y_test_predict)
print(y_test)

fig1, ax = plt.subplots()
df_cm = DataFrame(matrix, index=all_names, columns=all_names)
sn.heatmap(df_cm, cmap='Oranges', annot=True, ax=ax)

fig2, axs = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(16, 12))
fig2.suptitle(f'Confusion Matrix for ITD at Different Locations Linear SVM')

fig3, axs_fig3 = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(16, 12))
fig3.suptitle(f'Confusion Matrix for ITD at Different Locations Kernel SVM')

azimuth_id = 0
for azimuth, ax, ax_fig3 in zip([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], axs.flat, axs_fig3.flat):
    X_all = []
    y_all = []
    # azimuth = 30
    for indx in range(0, len(dataset)):
        item = dataset[indx]
        loc, hrtf, ITD_array, name = item # hrtf is 20 log10
        index = np.where((loc[:, 1] == 0) & np.isin(loc[:, 0], [azimuth]))[0]
        # index = np.where((loc[:, 1] == 0) & np.isin(loc[:, 0], [30]))[0]
    #     if hrtf[index, :].flatten().numpy().shape[0] != 1104:
    #         print(name)
        '''
        if name == "ari":
            print(f'hrtf shape is {hrtf.shape}')
            print(f'loc shape is {loc.shape}')
            print(f'ITD shape is {ITD_array.shape}')
        '''
        '''
        if train_test_is_same:
            if scale_choice[scale] == 1:
                hrtf_norm = 20 * np.log10(np.array(dataset.all_datasets[name].hrtf_normalized_common_loc[azimuth_id]))
                # hrtf_norm = 20 * np.log10(np.array(dataset.all_datasets[name].hrtf_normalized_all_loc))
                
                hrtf_norm_factor = 20 * np.log10(dataset.all_datasets['crossmod'].hrtf_normalized_common_loc[azimuth_id])
                # hrtf_norm_factor = 20 * np.log10(dataset.all_datasets['hutubs'].hrtf_normalized_all_loc)
            else:
                hrtf_norm = np.array(dataset.all_datasets[name].hrtf_normalized_common_loc[azimuth_id])
                hrtf_norm_factor = dataset.all_datasets['crossmod'].hrtf_normalized_common_loc[azimuth_id]
                
            if compute_choice[compute] == 1: 
                hrtf = np.subtract(hrtf, hrtf_norm)
                hrtf = np.add(hrtf, hrtf_norm_factor)
            else:
                hrtf = np.divide(hrtf, hrtf_norm)
                hrtf = np.multiply(hrtf, hrtf_norm_factor)
        '''
        X_all.append(ITD_array[index])
        y_all.append(all_names.index(name))
    X_all = np.stack(X_all)
    y_all = np.array(y_all)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    '''
    if train_test_is_same == False:
        X_test_normalized = []
        for hrtf, label in zip(X_test, y_test):
            name = all_names[label]
            
            if scale_choice[scale] == 1:
                hrtf_norm = 20 * np.log10(np.array(dataset.all_datasets[name].hrtf_normalized_common_loc[azimuth_id]))
                # hrtf_norm = 20 * np.log10(np.array(dataset.all_datasets[name].hrtf_normalized_all_loc))
                
                hrtf_norm_factor = 20 * np.log10(dataset.all_datasets['crossmod'].hrtf_normalized_common_loc[azimuth_id])
                # hrtf_norm_factor = 20 * np.log10(dataset.all_datasets['hutubs'].hrtf_normalized_all_loc)
            else:
                hrtf_norm = np.array(dataset.all_datasets[name].hrtf_normalized_common_loc[azimuth_id])
                hrtf_norm_factor = dataset.all_datasets['crossmod'].hrtf_normalized_common_loc[azimuth_id]
                
            if compute_choice[compute] == 1: 
                hrtf = np.subtract(hrtf, hrtf_norm)
                hrtf = np.add(hrtf, hrtf_norm_factor)
            else:
                hrtf = np.divide(hrtf, hrtf_norm)
                hrtf = np.multiply(hrtf, hrtf_norm_factor)
        
            X_test_normalized.append(hrtf)
        X_test = np.array(X_test_normalized)
    '''
    svm_clf = LinearSVC(C=100, random_state=42)
    kernel_svm = SVC(kernel='rbf', gamma=0.5, C=1) # SVC(kernel='rbf', gamma=0.5, C=1)
    scaler = StandardScaler()
    scaled_svm_clf = Pipeline([
            ("scaler", scaler),
            ("svc", svm_clf),
        ])
    scaled_kernel_svm = Pipeline([
        ("scaler", scaler),
        ("svc", kernel_svm),
    ])
    
    # svm_clf.fit(X_train, y_train)
    scaled_svm_clf.fit(X_train, y_train)
    print(f'svm trained using loc ({azimuth}, 0)')
    y_train_predict = scaled_svm_clf.predict(X_train)
    print(confusion_matrix(y_train, y_train_predict))

    y_test_predict = scaled_svm_clf.predict(X_test)
    print(confusion_matrix(y_test, y_test_predict))
    matrix_linear = confusion_matrix(y_test, y_test_predict)
    
    df_cm_linear = DataFrame(matrix_linear, index=all_names, columns=all_names)
    ax.set(title=f'location({azimuth}, 0)')
    sn.heatmap(df_cm_linear, cmap='Blues', annot=True, ax=ax)
    
    # kernel svm
    scaled_kernel_svm.fit(X_train, y_train)
    print(f'kernel svm trained using loc ({azimuth}, 0)')
    y_train_predict = scaled_kernel_svm.predict(X_train)
    print(confusion_matrix(y_train, y_train_predict))

    y_test_predict = scaled_kernel_svm.predict(X_test)
    print(confusion_matrix(y_test, y_test_predict))
    matrix_kernel = confusion_matrix(y_test, y_test_predict)
    
    df_cm_kernel = DataFrame(matrix_kernel, index=all_names, columns=all_names)
    ax_fig3.set(title=f'location({azimuth}, 0)')
    sn.heatmap(df_cm_kernel, cmap='Blues', annot=True, ax=ax_fig3)
    
    azimuth_id += 1
# %%
