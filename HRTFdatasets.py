#%%
import SOFAdatasets
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
import pickle as pkl
import os
from matplotlib import pyplot as plt
import matplotlib
from scipy.signal import minimum_phase
from scipy.signal import hilbert



class HRTFDataset(Dataset):
    def __init__(self, dataset="ari", freq=15, scale="linear", norm_way=0, hrtf_norm_factor=None):
        ## assert dataset is one of HRTFDataset
        dataset_dict = {"ari": "ARI", "hutubs": "HUTUBS", "ita": "ITA", "cipic": "CIPIC",
                        "3d3a": "Prin3D3A", "riec": "RIEC", "bili": "BiLi",
                        "listen": "Listen", "crossmod": "Crossmod", "sadie": "SADIE"}
        self.name = dataset
        self.dataset_obj = getattr(SOFAdatasets, dataset_dict[self.name])()
        self.freq = freq
        self.scale = scale
        self.norm_way = norm_way
        self.hrtf_norm_factor = hrtf_norm_factor
        self._create_ITD_dict()
        self._normalize_hrtf_common_loc()
        self._normalize_hrtf()
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
                
            
    def _get_hrir(self, idx):
        with open(os.path.join("/data2/neil/HRTF/prepocessed_hrirs", "%s_%03d.pkl" % (self.name, idx)), 'rb') as handle:
            _, hrir = pkl.load(handle)
        return hrir
           
    def _get_mag_phase(self, idx, freq, scale='linear'):
        with open(os.path.join("/data2/neil/HRTF/prepocessed_hrirs", "%s_%03d.pkl" % (self.name, idx)), 'rb') as handle:
            location, hrir = pkl.load(handle)
        
        ir = hrir[0]
        fig, ax = plt.subplots()
        ax.plot(ir, label=self.dataset_obj.name.upper()+" Subject%s" % self.dataset_obj._get_subject_ID(idx))
        
        
        for i in range(len(hrir)):
            hrir[i] = hilbert(hrir[i])
            amp = np.abs(hrir[i])
            phase = np.unwrap(np.angle(hrir[i]))
            hrir[i] = np.real(np.fft.ifft(amp * np.exp(1j * phase)))
        
        ir = hrir[0]
        fig, ax = plt.subplots()
        ax.plot(ir, label=self.dataset_obj.name.upper()+" Subject%s" % self.dataset_obj._get_subject_ID(idx))
        
        tf_mag = np.abs(np.fft.fft(hrir, n=256))
        tf_phase = np.angle(np.fft.fft(hrir, n=256))
        tf_mag = tf_mag[:, 1:93]  # first 128 freq bins, but up to 16k
        tf_phase = tf_phase[:, 1:93]
        
        if scale == "linear":
            pass
        elif scale == "log":
            tf_mag = 20 * np.log10(tf_mag)
        if freq == "all":
            return location, tf_mag, tf_phase
        return location, tf_mag[:, freq][:, np.newaxis], tf_phase[:, freq][:, np.newaxis]

    def _get_hrtf(self, idx, freq, scale="linear", norm_way=0):
        with open(os.path.join("/data2/neil/HRTF/prepocessed_hrirs", "%s_%03d.pkl" % (self.name, idx)), 'rb') as handle:
            location, hrir = pkl.load(handle)
        
        tf = np.abs(np.fft.fft(hrir, n=256))
        tf = tf[:, 1:105] # up to 18k
        # tf = tf[:, 1:93]  # first 128 freq bins, but up to 16k
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
        ## fifth way is to normalize on ear and position bases avg person hrtf
        elif norm_way == 4:
            loc_key = tf.shape[0]
            tf = 20 * np.log10(tf)
            if (idx % 2) == 0:
                tf_norm = 20 * np.log10(self.hrtf_normalized_all_loc_L[loc_key])
            else:
                tf_norm = 20 * np.log10(self.hrtf_normalized_all_loc_R[loc_key])
            try:
                tf -= tf_norm
                if self.hrtf_norm_factor is not None:
                    hrtf_norm_factor = 20 * np.log10(self.hrtf_norm_factor)
                    tf = np.add(tf, hrtf_norm_factor)
            except ValueError:
                print('ValueError, location points doesn"t match')
            tf = 10**(tf/20)
        ## sixth way is to normalize on avg person hrtf
        elif norm_way == 5:
            loc_key = tf.shape[0]
            tf = 20 * np.log10(tf)
            tf_norm = 20 * np.log10(self.hrtf_normalized_all_loc_mix[loc_key])
            # hrtf_norm = np.array(self.hrtf_normalized)
            try:
                tf -= tf_norm
                if self.hrtf_norm_factor is not None:
                    hrtf_norm_factor = 20 * np.log10(self.hrtf_norm_factor)
                    tf = np.add(tf, hrtf_norm_factor)
            except ValueError:
                print('ValueError, location points doesn"t match')
            tf = 10**(tf/20)  
        ## seventh way is to normalize on position base person hrtf
        elif norm_way == 6:
            loc_key = tf.shape[0]
            tf = 20 * np.log10(tf)
            tf_norm = 20 * np.log10(self.hrtf_normalized_all_loc[loc_key])
            try:
                tf -= tf_norm
                if self.hrtf_norm_factor is not None:
                    hrtf_norm_factor = 20 * np.log10(self.hrtf_norm_factor)
                    tf = np.add(tf, hrtf_norm_factor)
            except ValueError:
                print('ValueError, location points doesn"t match')
            tf = 10**(tf/20)
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
        return location, hrtf, self.ITD_dict[idx]

    def _plot_frontal_data(self, idx, ax):
        loc_idx = self.dataset_obj._get_frontal_locidx()
        _, hrtf = self._get_hrtf(idx, "all", "linear")
        ax.plot(20 * np.log10(hrtf[loc_idx]), label=self.dataset_obj.name.upper()+" Subject%s" % self.dataset_obj._get_subject_ID(idx))
        ax.set(xticks=list(np.arange(0, 128 + 16, 16)),
               xticklabels=['{:,.2f}k'.format(x) for x in list(np.arange(0, 128 + 16, 16) / 256 * 44.1)],
               title="Frontal HRTF",
               ylabel='Log Magnitude',
               xlabel='Frequency (Hz)')
        
    def _plot_data_at_loc(self, idx, loc, ax):
        loc_idx = self.dataset_obj._get_locidx_from_location(loc)
        _, hrtf = self._get_hrtf(idx, "all", "linear", 4)
        hrtf_at_loc = hrtf[loc_idx]
        ax.plot(20 * np.log10(hrtf_at_loc), label=self.dataset_obj.name.upper()+" Subject%s" % self.dataset_obj._get_subject_ID(idx))
        ax.set(xticks=list(np.arange(0, 128 + 16, 16)),
               xticklabels=['{:,.2f}k'.format(x) for x in list(np.arange(0, 128 + 16, 16) / 256 * 44.1)],
               title=f"HRTF at loc {loc} in {self.name}",
               ylabel='Log Magnitude',
               xlabel='Frequency (Hz)')
        
    def plot_hrtf_phase(self, idx, ax):
        # _, hrtf = self._get_hrtf(idx, "all", "linear", -1)
        loc, _, phase = self._get_mag_phase(idx, "all", "linear")
        index = np.where((loc[:, 1] == 0) & np.isin(loc[:, 0], [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]))[0]
        x, y = np.meshgrid(list(range(phase.shape[1])), list(range(len(index))))
        phase = phase[index]
        ax.plot_surface(x, y, phase)
        
    def plot_hrtf_phase_at_loc(self, idx, loc, ax):
        # _, hrtf = self._get_hrtf(idx, "all", "linear", -1)
        loc_idx = self.dataset_obj._get_locidx_from_location(loc)
        _, _, phase = self._get_mag_phase(idx, "all", "linear")
        phase_at_loc = phase[loc_idx]
        # hrtf_at_loc = hrtf_at_loc/np.max(hrtf_at_loc) 
        # hrtf = hrtf / self.max_mag # normalization
        ax.plot(phase_at_loc, label=self.dataset_obj.name.upper()+" Subject%s" % self.dataset_obj._get_subject_ID(idx))
        ax.set(xticks=list(np.arange(0, 128 + 16, 16)),
               xticklabels=['{:,.2f}k'.format(x) for x in list(np.arange(0, 128 + 16, 16) / 256 * 44.1)],
               title=f"phase at loc {loc} in {self.name}",
               ylabel='Log Magnitude',
               xlabel='Frequency (Hz)')
        
    def _plot_normalized_hrtf(self, ax):
        ax.plot(20 * np.log10(self.hrtf_normalized), label='normalized hrtf')
        ax.set(title=f"{self.name}")
        
    def _plot_normalized_hrtf_common_loc(self, ax):
        self._set_ax(ax)
        for i in range(12):
            self._plot_normalized_hrtf_at_loc(i, ax)
        
    def _plot_normalized_hrtf_at_loc(self, loc_id, ax):
        x = np.arange(0, 128)
        x = x / 256 * 44100
        x = x[1:105]
        ax.plot(x, 20 * np.log10(self.hrtf_normalized_common_loc[loc_id]), label=f'{self.name.upper()}')
        
    def _plot_normalized_hrtf_all_loc(self, ax):
        _, hrtf = self._get_hrtf(0, "all", "linear")
        norm_all_loc = np.array([0.0] * len(hrtf[0]))
        print(norm_all_loc.shape)
        for tfs in self.hrtf_normalized_all_loc.values():
            for tf in tfs:
                norm_all_loc += tf
            length = len(tfs)
            break
        norm_all_loc /= length
        x = np.arange(0, 128)
        x = x / 256 * 44100
        x = x[1:105]
        ax.plot(x, 20 * np.log10(self.hrtf_normalized), label=f'{self.name.upper()}')
        
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
        if len(self.dataset_obj.all_location_dict) == 1:
            _, hrtf = self._get_hrtf(0, "all", "linear")
            self.hrtf_normalized_all_loc_L = {} 
            self.hrtf_normalized_all_loc_R = {}
            self.hrtf_normalized_all_loc = {}
            normalization_factor = 0 
            for key, value in self.dataset_obj.all_location_dict.items():
                self.hrtf_normalized_all_loc_L[key] = np.array([[0.0] * len(hrtf[0])] * len(value))
                self.hrtf_normalized_all_loc_R[key] = np.array([[0.0] * len(hrtf[0])] * len(value))
                self.hrtf_normalized_all_loc[key] = np.array([[0.0] * len(hrtf[0])] * len(value))
            for idx in range(self.__len__()):
                _, hrtf = self._get_hrtf(idx, "all", "linear", -1)
                loc_key = hrtf.shape[0]
                try:
                    self.hrtf_normalized_all_loc[loc_key] += hrtf
                    if (idx % 2) == 0: # left ear
                        normalization_factor += 1
                        self.hrtf_normalized_all_loc_L[loc_key] += hrtf
                    else:
                        self.hrtf_normalized_all_loc_R[loc_key] += hrtf
                except Exception as e:
                    print(e)
                    normalization_factor -= 1
                    print(f'Error in {self.name}')
            for key in self.hrtf_normalized_all_loc_L.keys():
                self.hrtf_normalized_all_loc_L[key] /= normalization_factor
                self.hrtf_normalized_all_loc_R[key] /= normalization_factor
                self.hrtf_normalized_all_loc[key] /= (normalization_factor * 2)
        else:
            _, hrtf = self._get_hrtf(0, "all", "linear")
            normalization_factor = {} 
            loc_norm_dict_L = {} 
            loc_norm_dict_R = {}
            loc_norm_dict = {}
            self.hrtf_normalized_all_loc_L = {}
            self.hrtf_normalized_all_loc_R = {}
            self.hrtf_normalized_all_loc = {}
            for key, value in self.dataset_obj.all_location_dict.items():
                self.hrtf_normalized_all_loc_L[key] = np.array([[0.0] * len(hrtf[0])] * len(value))
                self.hrtf_normalized_all_loc_R[key] = np.array([[0.0] * len(hrtf[0])] * len(value))
                self.hrtf_normalized_all_loc[key] = np.array([[0.0] * len(hrtf[0])] * len(value))
            for value in self.dataset_obj.all_location_dict.values():
                # value is a dict maps from loc tuple to loc_id
                for key in value.keys():
                    if key not in loc_norm_dict_L:
                        loc_norm_dict_L[key] = np.array([[0.0] * len(hrtf[0])])
                        loc_norm_dict_R[key] = np.array([[0.0] * len(hrtf[0])])
                        loc_norm_dict[key] = np.array([[0.0] * len(hrtf[0])])
                        normalization_factor[key] = 0
            for idx in range(self.__len__()):
                _, hrtf = self._get_hrtf(idx, "all", "linear", -1)
                loc_key = hrtf.shape[0]
                for key in self.dataset_obj.all_location_dict[loc_key].keys():
                    loc_idx = self.dataset_obj.all_location_dict[loc_key][key]
                    loc_norm_dict[key] += hrtf[loc_idx]
                    if (idx % 2) == 0: # left ear
                        loc_norm_dict_L[key] += hrtf[loc_idx]
                        normalization_factor[key] += 1
                    else:
                        loc_norm_dict_R[key] += hrtf[loc_idx]
            for key in loc_norm_dict_L.keys():
                loc_norm_dict_L[key] /= normalization_factor[key] 
                loc_norm_dict_R[key] /= normalization_factor[key] 
                loc_norm_dict[key] /= (normalization_factor[key] * 2)
            for key, value in loc_norm_dict_L.items():
                for key_j, value_j in self.dataset_obj.all_location_dict.items():
                    if key in value_j:
                        self.hrtf_normalized_all_loc_L[key_j][value_j[key]] = value
            for key, value in loc_norm_dict_R.items():
                for key_j, value_j in self.dataset_obj.all_location_dict.items():
                    if key in value_j:
                        self.hrtf_normalized_all_loc_R[key_j][value_j[key]] = value
            for key, value in loc_norm_dict.items():
                for key_j, value_j in self.dataset_obj.all_location_dict.items():
                    if key in value_j:
                        self.hrtf_normalized_all_loc[key_j][value_j[key]] = value
        self.hrtf_normalized_all_loc_mix = {}
        for key, value in self.hrtf_normalized_all_loc.items():
            self.hrtf_normalized_all_loc_mix[key] = np.mean(value, axis=0)
            
    

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
                locs, hrtfs, ITD_array = dataset[item_idx]
                self.all_data.append((locs, hrtfs, dataset.name, 
                                      np.array(dataset.hrtf_normalized),
                                      np.array(dataset.hrtf_normalized_all_loc_L),
                                      dataset.hrtf_normalized_common_loc,
                                      ITD_array))
            self.length_array.append(len(dataset))        

    def __len__(self):
        return np.sum(self.length_array)

    def extend_locations(self, locs, hrtfs, ITD_array):
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
        # assign values for ITDs
        new_ITDs = torch.zeros(num_loc)
        new_ITDs[new_locs1.shape[0]:-new_locs2.shape[0]] = torch.from_numpy(ITD_array)
        new_ITDs[:new_locs1.shape[0]] = torch.from_numpy(ITD_array.copy()[index1])
        new_ITDs[-new_locs2.shape[0]:] = torch.from_numpy(ITD_array.copy()[index2])
        return new_locs, new_hrtfs, new_ITDs

    def __getitem__(self, idx):
        locs, hrtfs, names, _, _, _, ITD_array = self.all_data[idx]
        locs, hrtfs, ITD_array = self.extend_locations(locs, hrtfs, ITD_array)
        return locs, hrtfs, ITD_array, names
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
            loc, hrtf, _, name = sample
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
        
    def plot_normalized_hrtf_all_loc(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        plt.xscale("log")
        font = {'size': 14}  
        matplotlib.rc('font', **font)
        # fig.suptitle('System Frequency Response')
        for dataset in self.all_datasets.values():
            dataset._plot_normalized_hrtf_all_loc(ax)
        # xticks=list(np.arange(0, 128 + 16, 16)
        
        ax.set(xticks= [250, 500, 1000, 2000, 4000, 8000, 16000],
           xticklabels=['{:,.2f}k'.format(x/1000) for x in [250, 500, 1000, 2000, 4000, 8000, 16000]],
           xlim=[172, 17916])
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.set_xlabel('Frequency (Hz)', fontsize=14)
        ax.set_ylabel('Log Magnitude (dB)', fontsize=14)
        # plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig('system_freq_response.pdf', dpi=500, bbox_inches='tight')
            
    def plot_normalized_hrtfs_at_locs(self, loc_id_a=0, loc_id_b=1):
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 5))
        plt.xscale('log')
        font = {'size': 16}  
        matplotlib.rc('font', **font)
        # fig.suptitle(f'Normalized HRTF at location ({azimuth_a}, 0)')
        index = 0
        for i in range(4):
            for j in range(2):
                self.all_datasets[self.all_dataset_names[index]]._plot_normalized_hrtf_at_loc(loc_id_a, axs[0])
                self.all_datasets[self.all_dataset_names[index]]._plot_normalized_hrtf_at_loc(loc_id_b, axs[1])
                index += 1
        for ax in axs.flat:
            ax.set(xticks= [250, 1000, 4000, 16000],
                xticklabels=['{:,.2f}k'.format(x/1000) for x in [250, 1000, 4000, 16000]],
                xlim=[172, 17916])
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=16)
            ax.set_xlabel('Frequency (Hz)', fontsize=16)
            ax.set_ylabel('Log Magnitude (dB)', fontsize=16)
            # plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        for ax in axs.flat:
            ax.label_outer()
        plt.legend(bbox_to_anchor=(1.05, 0.8), loc='upper left')
        plt.savefig('system_freq_response.pdf', dpi=500, bbox_inches='tight')
            
    def plot_dataset_loc_diff(self, name_a, name_b, loc_A, loc_B):
        dataset_a = self.all_datasets[name_a]
        dataset_b = self.all_datasets[name_b]
        loc_idx_A_a = dataset_a.dataset_obj._get_locidx_from_location(loc_A)
        loc_idx_A_b = dataset_b.dataset_obj._get_locidx_from_location(loc_A)
        loc_idx_B_a = dataset_a.dataset_obj._get_locidx_from_location(loc_B)
        loc_idx_B_b = dataset_b.dataset_obj._get_locidx_from_location(loc_B)
        # avg_hrtf_A_a and avg_hrtf_A_b
        _, hrtf = dataset_a._get_hrtf(0, "all", "linear")
        hrtf_avg_A_a = np.array([0.0] * len(hrtf[0]))
        hrtf_avg_B_a = np.array([0.0] * len(hrtf[0]))
        count = 0
        for idx in range(dataset_a.__len__()):
            _, hrtfs = dataset_a._get_hrtf(idx, "all", "linear", -1)
            hrtf_avg_A_a += hrtfs[loc_idx_A_a]
            hrtf_avg_B_a += hrtfs[loc_idx_B_a]
            count += 1
        hrtf_avg_A_a /= count
        hrtf_avg_B_a /= count
        
        hrtf_avg_A_b = np.array([0.0] * len(hrtf[0]))
        hrtf_avg_B_b = np.array([0.0] * len(hrtf[0]))
        count = 0
        for idx in range(dataset_b.__len__()):
            _, hrtfs = dataset_b._get_hrtf(idx, "all", "linear", -1)
            hrtf_avg_A_b += hrtfs[loc_idx_A_b]
            hrtf_avg_B_b += hrtfs[loc_idx_B_b]
            count += 1
        hrtf_avg_A_b /= count
        hrtf_avg_B_b /= count
        
        hrtf_avg_A_a_b = 20 * np.log10(hrtf_avg_A_a / hrtf_avg_A_b)
        hrtf_avg_B_a_b = 20 * np.log10(hrtf_avg_B_a / hrtf_avg_B_b)
        
        x = np.arange(0, 128)
        x = x / 256 * 44100
        x = x[1:105]
        fig, ax = plt.subplots(figsize=(8, 4))
        plt.xscale("log")
        ax.plot(x, hrtf_avg_A_a_b, label=f'log difference from {name_a} to {name_b} at loc {loc_A}')
        ax.plot(x, hrtf_avg_B_a_b, label=f'log difference from {name_a} to {name_b} at loc {loc_B}')
        ax.set(xticks= [250, 500, 1000, 2000, 4000, 8000, 16000],
           xticklabels=['{:,.2f}k'.format(x/1000) for x in [250, 500, 1000, 2000, 4000, 8000, 16000]],
           xlim=[172, 17916])
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.set_xlabel('Frequency (Hz)', fontsize=14)
        ax.set_ylabel('Log Magnitude (dB)', fontsize=14)
        plt.legend()
        plt.savefig('system_response_loc_diff.pdf', dpi=500, bbox_inches='tight')
        
    def get_dataset_normalized_hrtf(self, name='riec'):
        return np.array(self.all_datasets[name].hrtf_normalized)


class PartialHRTFDataset(MergedHRTFDataset):
    def __init__(self, dataset_name="riec", freq=15, scale="linear", norm_way=2):
        super().__init__(dataset_name, freq, scale, norm_way)

    def __len__(self):
        return 210

    def __getitem__(self, idx):
        locs, hrtfs, _, names = self.all_data[idx]
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
    datasets = MergedHRTFDataset(["ari", "bili", "crossmod", 
                                  "hutubs", "listen", "riec", 
                                  "sadie", "3d3a"],
                                 freq=15)
    datasets.plot_normalized_hrtfs_at_locs(0, 3)