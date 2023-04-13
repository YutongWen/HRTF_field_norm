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


scale_choice = {'linear':0, 'log':1}
compute_choice = {'multi':0, 'add':1}
train_test_is_same = True
norm_to_zero_one = True
scale = 'log'
compute = 'add'


all_names = ["ari", "hutubs", "3d3a", "bili", "listen", "crossmod", "sadie", "riec"]
dataset = MergedHRTFDataset(all_names, "all", "log", norm_way=4)

X_all = []
y_all = []
for item in dataset:
    loc, hrtf, _, name = item
    index = np.where((loc[:, 1] == 0) & np.isin(loc[:, 0], [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]))[0] # , 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
    # index = np.where(loc[:, 0] == 0)[0]
#     if hrtf[index, :].flatten().numpy().shape[0] != 1104:
#         print(name)
    hrtf = hrtf[index,:]
    '''
    for azimuth_id in range(12):
        if scale_choice[scale] == 1:
            hrtf_norm = 20 * np.log10(np.array(dataset.all_datasets[name].hrtf_normalized_common_loc[azimuth_id]))
            # hrtf_norm = 20 * np.log10(np.array(dataset.all_datasets[name].hrtf_normalized_all_loc))
                
            hrtf_norm_factor = 20 * np.log10(dataset.all_datasets['hutubs'].hrtf_normalized_common_loc[azimuth_id])                # hrtf_norm_factor = 20 * np.log10(dataset.all_datasets['hutubs'].hrtf_normalized_all_loc)
        else:
            hrtf_norm = np.array(dataset.all_datasets[name].hrtf_normalized_common_loc[azimuth_id])
            hrtf_norm_factor = dataset.all_datasets['hutubs'].hrtf_normalized_common_loc[azimuth_id]
                    
        if compute_choice[compute] == 1: 
            hrtf[azimuth_id] = np.subtract(hrtf[azimuth_id], hrtf_norm)
            hrtf[azimuth_id] = np.add(hrtf[azimuth_id], hrtf_norm_factor)
        else:
            hrtf[azimuth_id] = np.divide(hrtf[azimuth_id], hrtf_norm)
            hrtf[azimuth_id] = np.multiply(hrtf[azimuth_id], hrtf_norm_factor)
    '''
    
    
    X_all.append(hrtf.float().flatten().numpy())
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

fig1, ax = plt.subplots()
df_cm = DataFrame(matrix, index=all_names, columns=all_names)
sn.heatmap(df_cm, cmap='Oranges', annot=True, ax=ax)



fig2, axs = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(16, 12))
fig2.suptitle(f'Confusion Matrix for Normalized HRTF at Different Locations Linear SVM\nchoice of {scale}, {compute}, train test the same is {train_test_is_same}')

fig3, axs_fig3 = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(16, 12))
fig3.suptitle(f'Confusion Matrix for Normalized HRTF at Different Locations Kernel SVM\nchoice of {scale}, {compute}, train test the same is {train_test_is_same}')

# train for twelve different locations
azimuth_id = 0
for azimuth, ax, ax_fig3 in zip([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], axs.flat, axs_fig3.flat):
    X_all = []
    y_all = []
    # azimuth = 30
    for indx in range(0, len(dataset), 2):
        item = dataset[indx]
        loc, hrtf, _, name = item # hrtf is 20 log10
        index = np.where((loc[:, 1] == 0) & np.isin(loc[:, 0], [azimuth]))[0]
        # index = np.where((loc[:, 1] == 0) & np.isin(loc[:, 0], [30]))[0]
    #     if hrtf[index, :].flatten().numpy().shape[0] != 1104:
    #         print(name)
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
        X_all.append(hrtf[index, :].float().flatten().numpy())
        y_all.append(all_names.index(name))
    X_all = np.stack(X_all)
    y_all = np.array(y_all)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

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
    count = 0
    for x, label in zip(X_train, y_train):
        if count > 10:
            break
        fig, ax = plt.subplots()
        ax.plot(x)
        ax.set(xticks=list(np.arange(0, 128 + 16, 16)),
               xticklabels=['{:,.2f}k'.format(x) for x in list(np.arange(0, 128 + 16, 16) / 256 * 44.1)],
               title=f"HRTF at position ({azimuth}, 0) in {all_names[label]}",
               ylabel='Log Magnitude',
               xlabel='Frequency (Hz)')
        count += 1
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

# %%
