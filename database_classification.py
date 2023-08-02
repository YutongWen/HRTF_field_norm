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
np.random.seed(42)
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
    loc = dataset.locations
    common_location = intersection(loc, common_location)


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
        index = np.where((loc[:, 1] == 0) & np.isin(loc[:, 0], [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]))[0]
        hrtf = hrtf[index,:]

        X_all.append(hrtf.float().flatten().numpy())
        y_all.append(all_names.index(name))
    X_all = np.stack(X_all)
    y_all = np.array(y_all)
    X_all, y_all = shuffle(X_all, y_all, random_state=40)
    svm_clf = SVC(kernel='rbf', gamma=0.004, C=0.005)
    scaler = StandardScaler()
    scaled_svm_clf = Pipeline([
            ("scaler", scaler),
            ("svc", svm_clf),
        ])
    
    y_pred = cross_val_predict(svm_clf, X_all, y_all, cv=5)
    matrix = metrics.confusion_matrix(y_all, y_pred, normalize='true')
    matrix *= 100
    cv_results = cross_validate(svm_clf, X_all, y_all, cv=5, return_train_score=True)
    print(cv_results['train_score'])
    print(cv_results['test_score'])
    print(np.mean(cv_results['test_score']))

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
    ax.set_title(title, fontsize=16)
    df_cm = DataFrame(matrix, index=all_names_capital, columns=all_names_capital)
    map = sn.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={"size": 15}, ax=ax, vmin=0.0, vmax=75.0)
    map.set_xlabel('Predicted Label', fontsize=14)
    map.set_ylabel('Ground-truth Label', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    for tick in ax.get_yticklabels():
        tick.set_rotation(45)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
plt.savefig('svm_result.pdf', dpi=500, bbox_inches='tight')
