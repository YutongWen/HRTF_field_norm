#%%
import SOFAdatasets
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
import pickle as pkl
import os
from matplotlib import pyplot as plt


class HRTFDataset(Dataset):
    def __init__(self, dataset="ari", freq=15, scale="linear", norm_way=0, hrtf_norm_factor=None):
        ## assert dataset is one of HRTFDataset
        dataset_dict = {"ari": "ARI", "hutubs": "HUTUBS", "ita": "ITA", "cipic": "CIPIC",
                        "3d3a": "Prin3D3A", "riec": "RIEC", "bili": "BiLi",
                        "listen": "Listen", "crossmod": "Crossmod", "sadie": "SADIE"}
        self.name = dataset
        self.dataset_obj = getattr(SOFAdatasets, dataset_dict[self.name])()
        # print(self.dataset_obj.subject_dict)
        self.freq = freq
        self.scale = scale
        self.norm_way = norm_way
        self.hrtf_norm_factor = hrtf_norm_factor
        self._create_ITD_dict()
        self._normalize_hrtf_common_loc()
        self._normalize_hrtf()
        # print(self.hrtf_normalized)
        self.hrtf_normalized_all_loc = None
        self._normalize_hrtf_all_loc()
        # self.max_mag = self._find_global_max_magnitude()

    def __len__(self):
        return self.dataset_obj.__len__()
    
    def _create_ITD_dict(self):
        # ITD for one ear (seen as left ear is left_arrival_time - right_arrival_time)
        sr = 44100.0
        time_increment = 1/sr # unit is seconds
        self.ITD_dict = {} # maps subject_idx to its ITD at all loc, len(subject_idx) = 2 * num of people
        for subject_ID in self.dataset_obj.subject_IDs:
            L_idx = self.dataset_obj._get_ear_ID(subject_ID, 0)
            R_idx = self.dataset_obj._get_ear_ID(subject_ID, 1)
            # print(f'L id is {L_idx}, R id is {R_idx}')
            L_irs = np.abs(self._get_hrir(L_idx))
            R_irs = np.abs(self._get_hrir(R_idx))
            L_peaks = np.argmax(L_irs, axis=1)
            R_peaks = np.argmax(R_irs, axis=1)
            try:
                L_ITDs = (L_peaks - R_peaks) * time_increment
                R_ITDs = (R_peaks - L_peaks) * time_increment
                self.ITD_dict[L_idx] = L_ITDs
                self.ITD_dict[R_idx] = R_ITDs
            except Exception as e:
                print(e, f'in {self.name}')
        '''
        index = 0
        for path in self.dataset_obj.all_sofa_files:
            # raw subject_ID, length is the num of people
            # hrir sample rate is 44100
            _, locations = self.dataset_obj._get_locations_from_one_sofa(path)
            ear_L_peak_idxs = np.array([0] * len(locations))
            ear_R_peak_idxs = np.array([0] * len(locations))
            for loc in locations:
                # each loc is a tuple
                ear_L_hrir_single_loc = np.abs(self.dataset_obj._get_HRIR(self.dataset_obj.subject_IDs[index], loc, 'left'))
                ear_R_hrir_single_loc = np.abs(self.dataset_obj._get_HRIR(self.dataset_obj.subject_IDs[index], loc, 'right'))
                loc_idx = self.dataset_obj.all_location_dict[len(locations)][loc]
                L_peak_idx = np.argmax(ear_L_hrir_single_loc)
                R_peak_idx = np.argmax(ear_R_hrir_single_loc)
                ear_L_peak_idxs[loc_idx] = L_peak_idx
                ear_R_peak_idxs[loc_idx] = R_peak_idx
            try:
                L_ITDs = (ear_L_peak_idxs - ear_R_peak_idxs) * time_increment
                R_ITDs = (ear_R_peak_idxs - ear_L_peak_idxs) * time_increment
                self.ITD_dict.append(L_ITDs)
                self.ITD_dict.append(R_ITDs)
            except Exception as e:
                print(e, f'in {self.name}')
            index += 1
        '''
                
            
    def _get_hrir(self, idx):
        with open(os.path.join("/data2/neil/HRTF/prepocessed_hrirs", "%s_%03d.pkl" % (self.name, idx)), 'rb') as handle:
            _, hrir = pkl.load(handle)
        return hrir
            

    def _get_hrtf(self, idx, freq, scale="linear", norm_way=0):
        # location, hrir = self.dataset_obj[idx]
        with open(os.path.join("/data2/neil/HRTF/prepocessed_hrirs", "%s_%03d.pkl" % (self.name, idx)), 'rb') as handle:
            location, hrir = pkl.load(handle)
        tf = np.abs(np.fft.fft(hrir, n=256))
        tf = tf[:, 1:93]  # first 128 freq bins, but up to 16k
        # tf = tf[:, 3:93]   # 500 Hz to 16kHz contribute to localization and are equalized
        ## how to normalize
        ## first way is to devide by max value

        if norm_way == 0:
            tf = tf / np.max(tf)
        ## second way is to devide by top 5% top value
        elif norm_way == 1:
            mag_flatten = tf.flatten()
            max_mag = np.mean(sorted(mag_flatten)[-int(mag_flatten.shape[0] / 20):])
            tf = tf / max_mag
        ## third way is to compute total energy of the equator
        elif norm_way == 2:
            equator_index = np.where(np.logical_and(location[:, 1] > -1, location[:, 1] <= 0))
            tf_equator = tf[equator_index]
            equator_azi = location[equator_index, 0][0]
            new_equator_index = np.argsort(equator_azi)
            new_equator_azi = equator_azi[new_equator_index]
            new_equator_tf = tf_equator[new_equator_index]

            total_energy = 0
            for x in range(len(new_equator_index)):
                if x == 0:
                    d_azi = 360 - new_equator_azi[-1]
                    # d_azi = new_equator_azi[1] - new_equator_azi[0]
                else:
                    d_azi = new_equator_azi[x] - new_equator_azi[x - 1]
                total_energy += np.square(new_equator_tf[x]).mean() * d_azi
            tf = tf / np.sqrt(total_energy / 360)
            # print(np.sqrt(total_energy / 360))
        ## fourth way is to normalize on common locations
        ## [(0.0, 0.0), (180.0, 0.0), (210.0, 0.0), (330.0, 0.0), (30.0, 0.0), (150.0, 0.0)]
        elif norm_way == 3:
            common_index = np.where(np.logical_and(np.logical_and(location[:, 1] > -1, location[:, 1] <= 0),
                                                   np.array(
                                                       [round(x) in [0, 180, 210, 330, 30, 150] for x in location[:, 0]])))
            tf_common = tf[common_index]
            mean_energy = np.sqrt(np.square(tf_common).mean())
            # print(mean_energy)
            tf = tf / mean_energy
        elif norm_way == 4:
            '''
            hrtf_norm = np.array(self.hrtf_normalized)
            tf = np.divide(tf, hrtf_norm)
            if self.hrtf_norm_factor is not None:
                tf = np.multiply(tf, self.hrtf_norm_factor)
            '''
            
            
            '''
            hrtf_norm = np.array(self.hrtf_normalized)
            hrtf_norm = 20 * np.log10(hrtf_norm)
            '''
            '''
            equator_index = np.where(np.logical_and(location[:, 1] > -1, location[:, 1] <= 0))
            tf_equator = tf[equator_index]
            equator_azi = location[equator_index, 0][0]
            new_equator_index = np.argsort(equator_azi)
            new_equator_azi = equator_azi[new_equator_index]
            new_equator_tf = tf_equator[new_equator_index]

            total_energy = 0
            for x in range(len(new_equator_index)):
                if x == 0:
                    d_azi = 360 - new_equator_azi[-1]
                    # d_azi = new_equator_azi[1] - new_equator_azi[0]
                else:
                    d_azi = new_equator_azi[x] - new_equator_azi[x - 1]
            
                
                equator = new_equator_tf[x]
                equator_norm = np.subtract(new_equator_tf[x], hrtf_norm)
                if self.hrtf_norm_factor is not None:
                    equator_norm = np.add(equator_norm, self.hrtf_norm_factor) 
                equator = equator_norm - np.min(equator_norm) + np.min(equator)
                # equator_norm - np.min(equator_norm) + np.min(equator)
                # equator /= np.max(equator)
                
                
                
             
                total_energy += np.square(equator).mean() * d_azi
            tf = tf / np.sqrt(total_energy / 360) 
            '''
            loc_key = tf.shape[0]
            
            tf = 20 * np.log10(tf)
            tf_norm = 20 * np.log10(self.hrtf_normalized_all_loc[loc_key])
            # hrtf_norm = np.array(self.hrtf_normalized)
            try:
                tf -= tf_norm
                if self.hrtf_norm_factor is not None:
                    hrtf_norm_factor = 20 * np.log10(self.hrtf_norm_factor)
                    tf = np.add(tf, hrtf_norm_factor)
            except ValueError:
                print('ValueError, location points doesn"t match')
            # tf_norm = np.divide(tf, hrtf_norm)
            
                # tf_norm = np.multiply(tf_norm, hrtf_norm_factor)
            # tf = tf_norm - np.min(tf_norm) + np.min(tf) # shift to greater than one
            # tf /= np.max(tf)
            tf = 10**(tf/20)
            # tf = tf_norm
            '''
            norm_factor = norm_factor - np.min(norm_factor)
            test_hrtf = np.subtract(test_hrtf, norm_factor)
            test_hrtf -= np.min(test_hrtf)
            test_hrtf /= np.max(test_hrtf)
            '''
            
            
            
            # tf /= np.max(tf) # normalize between zero
        else:
            pass

        if scale == "linear":
            tf = tf
        elif scale == "log":
            tf = 20 * np.log10(tf)
        if freq == "all":
            return location, tf
        return location, tf[:, freq][:, np.newaxis]

    def _find_global_max_magnitude(self):
        max_mag = 0
        for i in range(self.__len__()):
            _, tf_mag = self._get_hrtf(i, "all", "linear")
            cur_max_mag = np.max(tf_mag)
            if cur_max_mag > max_mag:
                max_mag = cur_max_mag
        return max_mag

    def __getitem__(self, idx):
        location, hrtf = self._get_hrtf(idx, self.freq, self.scale, self.norm_way)
        # return location, hrtf / self.max_mag
        return location, hrtf

    def _plot_frontal_data(self, idx, ax):
        loc_idx = self.dataset_obj._get_frontal_locidx()
        _, hrtf = self._get_hrtf(idx, "all", "linear")
        # hrtf = hrtf / self.max_mag # normalization
        ax.plot(20 * np.log10(hrtf[loc_idx]), label=self.dataset_obj.name.upper()+" Subject%s" % self.dataset_obj._get_subject_ID(idx))
        ax.set(xticks=list(np.arange(0, 128 + 16, 16)),
               xticklabels=['{:,.2f}k'.format(x) for x in list(np.arange(0, 128 + 16, 16) / 256 * 44.1)],
               title="Frontal HRTF",
               ylabel='Log Magnitude',
               xlabel='Frequency (Hz)')
        
    def _plot_data_at_loc(self, idx, loc, ax):
        loc_idx = self.dataset_obj._get_locidx_from_location(loc)
        _, hrtf = self._get_hrtf(idx, "all", "linear")
        # hrtf = hrtf / self.max_mag # normalization
        ax.plot(20 * np.log10(hrtf[loc_idx]), label=self.dataset_obj.name.upper()+" Subject%s" % self.dataset_obj._get_subject_ID(idx))
        ax.set(xticks=list(np.arange(0, 128 + 16, 16)),
               xticklabels=['{:,.2f}k'.format(x) for x in list(np.arange(0, 128 + 16, 16) / 256 * 44.1)],
               title="Frontal HRTF",
               ylabel='Log Magnitude',
               xlabel='Frequency (Hz)')
        
    def _plot_normalized_hrtf(self, ax):
        ax.plot(20 * np.log10(self.hrtf_normalized), label='normalized hrtf')
        ax.set(title=f"{self.name}")
        
    def _plot_normalized_hrtf_at_loc(self, loc_id, ax):
        ax.plot(20 * np.log10(self.hrtf_normalized_common_loc[loc_id]), label='normalized hrtf')
        ax.set(title=f"{self.name}")
        
    def _set_ax(self, ax):
        ax.set(xticks=list(np.arange(0, 128 + 16, 16)),
               xticklabels=['{:,.2f}k'.format(x) for x in list(np.arange(0, 128 + 16, 16) / 256 * 44.1)],
               ylabel='Log Magnitude',
               xlabel='Frequency (Hz)')
        
    def _normalize_hrtf(self):
        _, hrtf = self._get_hrtf(0, "all", "linear", -1)
        self.hrtf_normalized = [0.0] * len(hrtf[0])
        
        for hrtf in self.hrtf_normalized_common_loc:
            self.hrtf_normalized = [sum(x) for x in zip(self.hrtf_normalized, hrtf)]
        self.hrtf_normalized = [x / 12 for x in self.hrtf_normalized]
    
    def _normalize_hrtf_common_loc(self):
        _, hrtf = self._get_hrtf(0, "all", "linear")
        # hrtf_normalized_single_loc = [0.0] * len(hrtf[0])
        self.hrtf_normalized_common_loc = []
        for i in range(12):
            normalization_factor = 0
            azimuth = 30 * i
            hrtf_normalized_single_loc = [0.0] * len(hrtf[0])
            for idx in range(self.__len__()):
                _, hrtf = self._get_hrtf(idx, "all", "linear", -1)
                loc_key = hrtf.shape[0]
                normalization_factor += 1
                try:
                    loc_idx = self.dataset_obj.all_location_dict[loc_key][(azimuth, 0)]
                    hrtf_normalized_single_loc = [sum(x) for x in zip(hrtf_normalized_single_loc, hrtf[loc_idx])]
                except KeyError:
                    # print(f'KeyError in dataset {self.name}')
                    try:
                        loc_idx = self.dataset_obj.all_location_dict[loc_key][(azimuth, -0.72)]
                    except KeyError:
                        print(f'KeyError in dataset {self.name}, no loc ({azimuth}, 0)')
                        normalization_factor -= 1 
                except Exception as e:
                    print(e)
                    print(f'error in {self.name}')
            if normalization_factor != 0:
                hrtf_normalized_single_loc = [x / normalization_factor for x in hrtf_normalized_single_loc]
            self.hrtf_normalized_common_loc.append(hrtf_normalized_single_loc)
    
    def _normalize_hrtf_all_loc(self):
        '''
        _, hrtf = self._get_hrtf(0, "all", "linear")
        self.hrtf_normalized_all_loc = [0.0] * len(hrtf[0])
        normalization_factor = 0
        for idx in self.dataset_obj.subject_dict.values():
            _, hrtf = self._get_hrtf(idx, "all", "linear", -1)
            for loc_idx in self.dataset_obj.location_dict.values():
                normalization_factor += 1
                try:   
                    self.hrtf_normalized_all_loc = [sum(x) for x in zip(self.hrtf_normalized_all_loc, hrtf[loc_idx])]
                except IndexError:
                    print(f'IndexError, mismatch of hrtf size within {self.name}')
                    normalization_factor -= 1    
        self.hrtf_normalized_all_loc = [x / normalization_factor for x in self.hrtf_normalized_all_loc]
        '''
        
        if len(self.dataset_obj.all_location_dict) == 1:
            _, hrtf = self._get_hrtf(0, "all", "linear")
            self.hrtf_normalized_all_loc = {} # dict of np.array
            normalization_factor = 0 
            for key, value in self.dataset_obj.all_location_dict.items():
                self.hrtf_normalized_all_loc[key] = np.array([[0.0] * len(hrtf[0])] * len(value))
            # self.hrtf_normalized_all_loc = np.array([[0.0] * len(hrtf[0])] * len(self.dataset_obj.location_dict))
            for idx in range(self.__len__()):
                _, hrtf = self._get_hrtf(idx, "all", "linear", -1)
                loc_key = hrtf.shape[0]
                normalization_factor += 1
                try:
                    self.hrtf_normalized_all_loc[loc_key] += hrtf
                except Exception as e:
                    print(e)
                    normalization_factor -= 1
                    print(f'Error in {self.name}')
            for key in self.hrtf_normalized_all_loc.keys():
                self.hrtf_normalized_all_loc[key] /= normalization_factor
        else:
            # if len(self.dataset_obj.all_location_dict) > 1
            
            _, hrtf = self._get_hrtf(0, "all", "linear")
            normalization_factor = {} # dict of int for different location tuples
            loc_norm_dict = {} # dict contains all possible loc in the dataset
            self.hrtf_normalized_all_loc = {} # dict of np.array
            for key, value in self.dataset_obj.all_location_dict.items():
                self.hrtf_normalized_all_loc[key] = np.array([[0.0] * len(hrtf[0])] * len(value))
            for value in self.dataset_obj.all_location_dict.values():
                # value is a dict maps from loc tuple to loc_id
                for key in value.keys():
                    if key not in loc_norm_dict:
                        loc_norm_dict[key] = np.array([[0.0] * len(hrtf[0])])
                        normalization_factor[key] = 0
            for idx in range(self.__len__()):
                _, hrtf = self._get_hrtf(idx, "all", "linear", -1)
                loc_key = hrtf.shape[0]
                for key in self.dataset_obj.all_location_dict[loc_key].keys():
                    # each key is a tuple of location
                    loc_idx = self.dataset_obj.all_location_dict[loc_key][key]
                    loc_norm_dict[key] += hrtf[loc_idx]
                    normalization_factor[key] += 1
            for key in loc_norm_dict.keys():
                loc_norm_dict[key] /= normalization_factor[key] 
            for key, value in loc_norm_dict.items():
                # key is tuple of location, value is np array of normalized hrtf
                for key_j, value_j in self.dataset_obj.all_location_dict.items():
                    # key_j is location dim, value_j is dict maps from loc tuple to loc_id
                    if key in value_j:
                        self.hrtf_normalized_all_loc[key_j][value_j[key]] = value
            
    

class MergedHRTFDataset(Dataset):
    def __init__(self, all_dataset_names, freq, scale="linear", norm_way=2):
        self.all_dataset_names = all_dataset_names
        # ["ari", "hutubs", "cipic", "3d3a", "riec", "bili", "listen", "crossmod", "sadie", "ita"]
        self.all_datasets = {}
        self.length_array = []
        self.all_data = []
        dataset_norm = HRTFDataset('3d3a', freq, scale, norm_way)
        self.norm_factor = np.array(dataset_norm.hrtf_normalized)
        for dataset_name in self.all_dataset_names:
            self.all_datasets[dataset_name] = HRTFDataset(dataset_name, freq, scale, norm_way, self.norm_factor)
        for dataset in self.all_datasets.values():
            for item_idx in range(len(dataset)):
                locs, hrtfs = dataset[item_idx]
                self.all_data.append((locs, hrtfs, dataset.name, 
                                      np.array(dataset.hrtf_normalized),
                                      np.array(dataset.hrtf_normalized_all_loc),
                                      dataset.hrtf_normalized_common_loc))
            self.length_array.append(len(dataset))
        # self.length_sum = np.insert(np.cumsum(self.length_array), 0, 0)
        
        '''
        self.common_loc = []
        for i in range(12):
            loc_all = [0] * len(self.all_datasets['ari'].hrtf_normalized)
            loc_all = np.array(loc_all)
            for dataset in self.all_datasets.values():
                loc_all = np.add(loc_all, np.array(dataset.hrtf_normalized_common_loc[i]))
            loc_all = loc_all/len(self.all_datasets)
            self.common_loc.append(loc_all)
        '''
        
        

    def __len__(self):
        return np.sum(self.length_array)

    def extend_locations(self, locs, hrtfs):
        ## Extend locations from -30 to 0 and from 360 to 390
        index1 = np.where(locs[:, 0] > 330)
        new_locs1 = locs.copy()[index1]
        new_locs1[:, 0] -= 360
        index2 = np.where(locs[:, 0] < 30)
        new_locs2 = locs.copy()[index2]
        new_locs2[:, 0] += 360
        num_loc = new_locs1.shape[0] + locs.shape[0] + new_locs2.shape[0]
        # assign values for locs
        new_locs = torch.zeros(num_loc, 2)
        new_locs[new_locs1.shape[0]:-new_locs2.shape[0]] = torch.from_numpy(locs)
        new_locs[:new_locs1.shape[0]] = torch.from_numpy(new_locs1)
        new_locs[-new_locs2.shape[0]:] = torch.from_numpy(new_locs2)
        # assign values for hrtfs
        new_hrtfs = torch.zeros(num_loc, hrtfs.shape[1])
        new_hrtfs[new_locs1.shape[0]:-new_locs2.shape[0]] = torch.from_numpy(hrtfs)
        new_hrtfs[:new_locs1.shape[0]] = torch.from_numpy(hrtfs.copy()[index1])
        new_hrtfs[-new_locs2.shape[0]:] = torch.from_numpy(hrtfs.copy()[index2])
        return new_locs, new_hrtfs

    def __getitem__(self, idx):
        locs, hrtfs, names, normalized_hrtf, normalized_hrtf_all_loc, normalized_hrtf_common_loc = self.all_data[idx]
        locs, hrtfs = self.extend_locations(locs, hrtfs)
        return locs, hrtfs, names
        # return locs, hrtfs, names, normalized_hrtf, normalized_hrtf_all_loc, normalized_hrtf_common_loc

    def collate_fn(self, samples):
        B = len(samples)
        len_sorted, _ = torch.sort(torch.Tensor([sample[0].shape[0] for sample in samples]), descending=True)
        max_num_loc = int(len_sorted[0].item())
        n_freq = samples[0][1].shape[1]
        locs, hrtfs, masks, names = [], [], [], []
        locs = -torch.ones((B, max_num_loc, 2))
        hrtfs = -torch.ones((B, max_num_loc, n_freq))
        masks = torch.zeros((B, max_num_loc, n_freq))
        for idx, sample in enumerate(samples):
            num_loc = sample[0].shape[0]
            loc, hrtf, name = sample
            locs[idx, :num_loc, :] = loc
            hrtfs[idx, :num_loc, :] = hrtf
            masks[idx, :num_loc, :] = 1
            names.append(name)
        return locs, hrtfs, masks, default_collate(names)
    
    def plot_normalized_hrtfs(self):
        fig, axs = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(10, 13))
        fig.suptitle('Normalized HRTF')
        index = 0
        for i in range(4):
            for j in range(2):
                self.all_datasets[self.all_dataset_names[index]]._plot_normalized_hrtf(axs[i, j])
                index += 1
        for ax in axs.flat:
            ax.set(xticks=list(np.arange(0, 128 + 16, 16)),
               xticklabels=['{:,.2f}k'.format(x) for x in list(np.arange(0, 128 + 16, 16) / 256 * 44.1)],
               ylabel='Log Magnitude',
               xlabel='Frequency (Hz)')
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        for ax in axs.flat:
            ax.label_outer()
            
    def plot_normalized_hrtfs_at_loc(self, loc_id=0):
        fig, axs = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(10, 13))
        azimuth = loc_id * 30
        fig.suptitle(f'Normalized HRTF at location ({azimuth}, 0)')
        index = 0
        for i in range(4):
            for j in range(2):
                self.all_datasets[self.all_dataset_names[index]]._plot_normalized_hrtf_at_loc(loc_id, axs[i, j])
                index += 1
        for ax in axs.flat:
            ax.set(xticks=list(np.arange(0, 128 + 16, 16)),
               xticklabels=['{:,.2f}k'.format(x) for x in list(np.arange(0, 128 + 16, 16) / 256 * 44.1)],
               ylabel='Log Magnitude',
               xlabel='Frequency (Hz)')
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        for ax in axs.flat:
            ax.label_outer()
    
    def get_dataset_normalized_hrtf(self, name='riec'):
        return np.array(self.all_datasets[name].hrtf_normalized)


class PartialHRTFDataset(MergedHRTFDataset):
    def __init__(self, dataset_name="riec", freq=15, scale="linear", norm_way=2):
        super().__init__(dataset_name, freq, scale, norm_way)

    def __len__(self):
        return 210

    def __getitem__(self, idx):
        locs, hrtfs, names = self.all_data[idx]
        indices = np.array([  0,   2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,
        26,  28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,
        52,  54,  56,  58,  60,  62,  64,  66,  68,  70,  73,  75,  77,
        79,  81,  83,  85,  87,  89,  91,  93,  95,  97,  99, 101, 103,
       105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129,
       131, 133, 135, 137, 139, 141, 143, 144, 146, 148, 150, 152, 154,
       156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180,
       182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206,
       208, 210, 212, 214, 217, 219, 221, 223, 225, 227, 229, 231, 233,
       235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 255, 257, 259,
       261, 263, 265, 267, 269, 271, 273, 275, 277, 279, 281, 283, 285,
       287, 288, 290, 292, 294, 296, 298, 300, 302, 304, 306, 308, 310,
       312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336,
       338, 340, 342, 344, 346, 348, 350, 352, 354, 356, 358, 361, 363,
       365, 367, 369, 371, 373, 375, 377, 379, 381, 383, 385, 387, 389,
       391, 393, 395, 397, 399, 401, 403, 405, 407, 409, 411, 413, 415,
       417, 419, 421, 423, 425, 427, 429, 431, 432, 434, 436, 438, 440,
       442, 444, 446, 448, 450, 452, 454, 456, 458, 460, 462, 464, 466,
       468, 470, 472, 474, 476, 478, 480, 482, 484, 486, 488, 490, 492,
       494, 496, 498, 500, 502, 505, 507, 509, 511, 513, 515, 517, 519,
       521, 523, 525, 527, 529, 531, 533, 535, 537, 539, 541, 543, 545,
       547, 549, 551, 553, 555, 557, 559, 561, 563, 565, 567, 569, 571,
       573, 575, 576, 578, 580, 582, 584, 586, 588, 590, 592, 594, 596,
       598, 600, 602, 604, 606, 608, 610, 612, 614, 616, 618, 620, 622,
       624, 626, 628, 630, 632, 634, 636, 638, 640, 642, 644, 646, 649,
       651, 653, 655, 657, 659, 661, 663, 665, 667, 669, 671, 673, 675,
       677, 679, 681, 683, 685, 687, 689, 691, 693, 695, 697, 699, 701,
       703, 705, 707, 709, 711, 713, 715, 717, 719, 720, 722, 724, 726,
       728, 730, 732, 734, 736, 738, 740, 742, 744, 746, 748, 750, 752,
       754, 756, 758, 760, 762, 764, 766, 768, 770, 772, 774, 776, 778,
       780, 782, 784, 786, 788, 790, 793, 795, 797, 799, 801, 803, 805,
       807, 809, 811, 813, 815, 817, 819, 821, 823, 825, 827, 829, 831,
       833, 835, 837, 839, 841, 843, 845, 847, 849, 851, 853, 855, 857,
       859, 861, 863, 864])
        locs = locs[indices]
        hrtfs = hrtfs[indices]
        locs, hrtfs = self.extend_locations(locs, hrtfs)
        return locs, hrtfs, names


class HRTFFitting(Dataset):
    def __init__(self, location, hrtf, part="full"):
        super(HRTFFitting, self).__init__()
        assert part in ["full", "half", "random_half"]
        num_locations = location.shape[0]
        assert hrtf.shape[0] == num_locations

        self.hrtf = hrtf
        self.coords = location

        if part == "full":
            self.indices = np.arange(num_locations)
        elif part == "half":
            self.indices = np.arange(0, num_locations, 2)
        elif part == "random_half":
            self.indices = np.random.choice(num_locations, num_locations // 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError
        return self.coords[self.indices], self.hrtf[self.indices]


def fitting_dataset_wrapper(idx, dataset="crossmod", freq=1, part="full"):
    dataset = HRTFDataset(dataset, freq)
    loc, hrtf = dataset[idx]
    return HRTFFitting(loc, hrtf, part)

#%%


if __name__ == "__main__":
    res = HRTFDataset(dataset='crossmod')

    # print(res.ITD_dict)
    # fig, ax = plt.subplots()
    # res._plot_frontal_data(1, ax)
    # res._plot_normalized_hrtf(ax)
    # res._set_ax(ax)
    
    # no ita, sadie, ari
    datasets = MergedHRTFDataset(["3d3a", "ita", "sadie", "ari",
                                  "riec", "bili", "hutubs", "listen",
                                  "crossmod", "cipic"],
                                 freq=15)
    # datasets.plot_normalized_hrtfs()
    '''
    for i in range(12):
        datasets.plot_normalized_hrtfs_at_loc(loc_id=i)
    '''
    '''
    for item in datasets:
        loc, hrtf, name, hrtf_normalize_factor, _, _ = item
        print(loc)
        print(loc.shape)
        break
    '''

