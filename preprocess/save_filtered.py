import scipy.io as scio
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import math
from matplotlib import cm  
import pandas as pd


data_path = '../../face-cognition-data/step12/'
files = os.listdir(data_path)
set = './behaviors/ERPs/info.set'
info = mne.read_epochs_eeglab(set).info

files = os.listdir(data_path)
name_lst = []
for file in files:
    if file.endswith('.mat'):
        name = file.split('.')[0].split('_')[0]
        if name not in name_lst:
            name_lst.append(name)
num_array = np.array(name_lst)

amp_max = 20
save_array = np.zeros((186, 9))
for idx, subject_name in enumerate(num_array):
    save_array[idx, 0] = subject_name
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    mat = scio.loadmat(os.path.join(data_path, '{}_Facc_PF_cc.mat'.format(subject_name)))['results12'][0, 0]
    ref_c = mat['ref_c']
    c_evoked = mne.EvokedArray(ref_c.T, info, tmin=-0.1)  
    c_Pz = c_evoked.copy().pick("Oz")
    c_Pz_filtered = c_Pz.copy().filter(l_freq=None, h_freq=10)
    ch_roi, lat_roi, amp_roi = c_Pz_filtered.get_peak(tmin=0.1, tmax=0.7, mode="pos", return_amplitude=True)
    print(c_Pz_filtered.data.shape)
    axs[0, 0].plot(ref_c[:, 27], label='Facc_PF')
    axs[0, 1].plot(c_Pz_filtered.data.T, label='Facc_PF')
    # axs[0, 1].plot(250*(lat_roi+0.1), amp_roi, marker='*', color='k')
    axs[0, 0].plot(250*(lat_roi+0.1), ref_c[int(250*(lat_roi+0.1)), 27], marker='*', color='k')
    save_array[idx, 1] = lat_roi
    save_array[idx, 2] = amp_roi/20

    mat = scio.loadmat(os.path.join(data_path, '{}_Facc_UPF_cc.mat'.format(subject_name)))['results12'][0, 0]
    ref_c = mat['ref_c']
    c_evoked = mne.EvokedArray(ref_c.T, info, tmin=-0.1)  
    c_Pz = c_evoked.copy().pick("Oz")
    c_Pz_filtered = c_Pz.copy().filter(l_freq=None, h_freq=10)
    ch_roi, lat_roi, amp_roi = c_Pz_filtered.get_peak(tmin=0.1, tmax=0.7, mode="pos", return_amplitude=True)
    print(c_Pz_filtered.data.shape)
    axs[0, 0].plot(ref_c[:, 27], label='Facc_UPF')
    axs[0, 1].plot(c_Pz_filtered.data.T, label='Facc_UPF')
    # axs[0, 1].plot(250*(lat_roi+0.1), amp_roi, marker='*', color='k')
    axs[0, 0].plot(250*(lat_roi+0.1), ref_c[int(250*(lat_roi+0.1)), 27], marker='*', color='k')
    axs[0, 0].legend(loc='lower right')
    axs[0, 1].legend(loc='lower right')
    save_array[idx, 3] = lat_roi
    save_array[idx, 4] = amp_roi/20

    mat = scio.loadmat(os.path.join(data_path, '{}_Fsp_PF_cc.mat'.format(subject_name)))['results12'][0, 0]
    ref_c = mat['ref_c']
    c_evoked = mne.EvokedArray(ref_c.T, info, tmin=-0.1)  
    c_Pz = c_evoked.copy().pick("Oz")
    c_Pz_filtered = c_Pz.copy().filter(l_freq=None, h_freq=10)
    ch_roi, lat_roi, amp_roi = c_Pz_filtered.get_peak(tmin=0.1, tmax=0.7, mode="pos", return_amplitude=True)
    print(c_Pz_filtered.data.shape)
    axs[1, 0].plot(ref_c[:, 27], label='Fsp_PF')
    axs[1, 1].plot(c_Pz_filtered.data.T, label='Fsp_PF')
    # axs[1, 1].plot(250*(lat_roi+0.1), amp_roi, marker='*', color='k')
    axs[1, 0].plot(250*(lat_roi+0.1), ref_c[int(250*(lat_roi+0.1)), 27], marker='*', color='k')
    save_array[idx, 5] = lat_roi
    save_array[idx, 6] = amp_roi/20

    mat = scio.loadmat(os.path.join(data_path, '{}_Fsp_UPF_cc.mat'.format(subject_name)))['results12'][0, 0]
    ref_c = mat['ref_c']
    c_evoked = mne.EvokedArray(ref_c.T, info, tmin=-0.1)  
    c_Pz = c_evoked.copy().pick("Oz")
    c_Pz_filtered = c_Pz.copy().filter(l_freq=None, h_freq=10)
    ch_roi, lat_roi, amp_roi = c_Pz_filtered.get_peak(tmin=0.1, tmax=0.7, mode="pos", return_amplitude=True)
    axs[1, 0].plot(ref_c[:, 27], label='Fsp_UPF')
    axs[1, 1].plot(c_Pz_filtered.data.T, label='Fsp_UPF')
    # axs[1, 1].plot(250*(lat_roi+0.1), amp_roi, marker='*', color='k')
    axs[1, 0].plot(250*(lat_roi+0.1), ref_c[int(250*(lat_roi+0.1)), 27], marker='*', color='k')
    axs[1, 0].legend(loc='lower right')
    axs[1, 1].legend(loc='lower right')
    save_array[idx, 7] = lat_roi
    save_array[idx, 8] = amp_roi/20

    # plt.show()
    fig.savefig('../../face-cognition-data/filtered_Oz/{}.png'.format(subject_name))
    plt.close()

df = pd.DataFrame(save_array)
df.to_excel('latency_shifts_original.xlsx', index=False, header=False)

