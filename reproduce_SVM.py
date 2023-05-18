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
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, cross_validate
from scipy.stats import norm
from sklearn.utils import shuffle
import random
random.seed(42) 

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

n_sample = 36 # size of sadie, smallest size dataset

norms = [0, 5, 4]
titles = ['raw data', 'normalized by averaging shared positions', 'normalized by individual position']
fig1, axs = plt.subplots(1, 3, figsize=(28, 7.5))
for n, ax, title in zip(norms, axs.flat, titles):
    all_names = ["ari", "bili", "crossmod", "hutubs", "listen", "riec", "sadie", "3d3a"]
    all_names_capital = []
    for name in all_names:
        all_names_capital.append(name.upper())
    dataset = MergedHRTFDataset(all_names, "all", "log", norm_way=n)
    # sampling 36 samples from each dataset
    dataset_sampled = []
    for key, value in dataset.all_datasets.items():
        sample_idx = random.sample(range(len(value)), n_sample)
        for idx in sample_idx:
            loc, hrtf, ITD_array = value[idx]
            loc, hrtf, ITD_array = dataset.extend_locations(loc, hrtf, ITD_array)
            item = (loc, hrtf, ITD_array, key)
            dataset_sampled.append(item)

    X_all = []
    y_all = []
    for item in dataset_sampled:
        loc, hrtf, _, name = item
        index = np.where((loc[:, 1] == 0) & np.isin(loc[:, 0], [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]))[0] # , 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
        # index = np.where(loc[:, 0] == 0)[0]
    #     if hrtf[index, :].flatten().numpy().shape[0] != 1104:
    #         print(name)
        hrtf = hrtf[index,:]

        X_all.append(hrtf.float().flatten().numpy())
        y_all.append(all_names.index(name))
    X_all = np.stack(X_all)
    y_all = np.array(y_all)
    X_all, y_all = shuffle(X_all, y_all, random_state=40)
    # X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42) 
    # svm_clf = LinearSVC(C=100, random_state=42)
    svm_clf = SVC(kernel='rbf', gamma=0.004, C=0.005)
    scaler = StandardScaler()
    scaled_svm_clf = Pipeline([
            ("scaler", scaler),
            ("svc", svm_clf),
        ])
    # svm_clf.fit(X_train, y_train)
    '''
    scaled_svm_clf.fit(X_train, y_train)

    y_train_predict = scaled_svm_clf.predict(X_train)
    print(metrics.confusion_matrix(y_train, y_train_predict))

    y_test_predict = scaled_svm_clf.predict(X_test)
    print(metrics.confusion_matrix(y_test, y_test_predict))
    '''
    y_pred = cross_val_predict(svm_clf, X_all, y_all, cv=5)
    matrix = metrics.confusion_matrix(y_all, y_pred, normalize='true')
    matrix *= 100
    cv_results = cross_validate(svm_clf, X_all, y_all, cv=5, return_train_score=True)
    print(cv_results['train_score'])
    print(cv_results['test_score'])
    print(np.mean(cv_results['test_score']))

    # matrix = metrics.confusion_matrix(y_test, y_test_predict, normalize='true')
    # conf_mat = metrics.confusion_matrix(y_test, y_test_predict)
    conf_mat = metrics.confusion_matrix(y_all, y_pred)
    accuracy = np.trace(conf_mat) / np.sum(conf_mat)
    n = np.sum(conf_mat)
    ci = norm.interval(0.95, loc=accuracy, scale=np.sqrt(accuracy*(1-accuracy)/n))
    accuracy = accuracy * 100
    low = ci[0] * 100
    # up = ci[1] * 100
    acc_range = accuracy - low
    title = f'Classification Accuracy ({accuracy:.2f}% \xb1 {acc_range:.2f})'
    # fig1, ax = plt.subplots()
    ax.set_title(title, fontsize=14)
    df_cm = DataFrame(matrix, index=all_names_capital, columns=all_names_capital)
    map = sn.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={"size": 14}, ax=ax, vmin=0.0, vmax=75.0)
    map.set_xlabel('Predicted Label', fontsize=14)
    map.set_ylabel('True Label', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    for tick in ax.get_yticklabels():
        tick.set_rotation(45)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    # break
plt.savefig('svm_result.pdf', dpi=500, bbox_inches='tight')
    
    
    
    
    
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = [False, True])
# cm_display.plot()
'''

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
