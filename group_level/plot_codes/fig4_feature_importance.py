import torch
import mne
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.io as scio
from sklearn import preprocessing
import matplotlib.ticker as mticker
from matplotlib.pyplot import MultipleLocator
from matplotlib import cm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['figure.figsize'] = (11.0, 6.0)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
x = np.linspace(-10, 1190, 300)
y = np.linspace(1, 41, 41)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
set = './ERP_results/info.set'
info = mne.read_epochs_eeglab(set).info

name_lst = ['(1) Step0 (Unseparated data)', 
            '(2) Step1 (Decompose and obtain C components)', 
            '(3) Step2 (Synchronize latency jitters per condition)', 
            '(4) Step3 (Synchronize latencies between conditions)', 
            '(5) Step4 (Align amplitudes across subjects)']

def only_importance():
    file = './RIDE/channel.mat'
    cols = scio.loadmat(file)['chanlocs']
    labels = []
    for i in np.squeeze(cols['labels']):
        labels.append(i.item())
    print(labels)
    yreal = np.linspace(1, 41, 41)
    yrange = labels[:41]

    fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(18, 18))
    for k in range(4):
        file = './latency/results/importance/step{}.pkl'.format(k)

        with open(file, "rb") as f:
            dict = pickle.load(f)
        primed, unprimed = dict[0], dict[1]
        all_attr = np.zeros((41, 300))

        for attr in primed:
            all_attr += preprocessing.MinMaxScaler().fit_transform(attr)
        for attr in unprimed:
            all_attr += (1-preprocessing.MinMaxScaler().fit_transform(attr))

        all_attr /= (len(primed)+len(unprimed))
        print(np.max(all_attr), np.min(all_attr))
        im = axs[k].pcolormesh(x, y, all_attr, cmap='seismic', shading='auto', vmin=0, vmax=1)
        axs[k].set_title(name_lst[k], loc='left', size='x-large')
        axs[k].set_yticks(yreal)
        axs[k].set_yticklabels(yrange, size=8)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.suptitle('Feature importance of Familiar-Easy task for different steps', fontsize=20, y=0.97)
    plt.xlabel('Time (ms)', fontsize=17)
    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), ticks=mticker.MultipleLocator(1), pad=0.01)
    cbar.ax.tick_params(labelsize='xx-large')
    cbar.ax.set_yticks([0, 1])
    cbar.ax.set_yticklabels(['primed \ncondition', 'unprimed \ncondition'], fontsize=15)
    # plt.show()
    plt.savefig('features.png')


def importance_erp():
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(8, 1, height_ratios=[1,2,1,2,1,2,1,2], hspace=0)
    
    ax_features = []
    ax_erp = []
    for idx in range(4):
        ax_bottom = fig.add_subplot(gs[idx*2+1])
        ax_up = fig.add_subplot(gs[idx*2], sharex=ax_bottom)
        ax_up.tick_params(axis="both", labelbottom=False, labelleft=False)

        ax_up.spines['top'].set_color('none')
        ax_up.spines['left'].set_color('none')
        ax_up.spines['right'].set_color('none')
        ax_up.spines['bottom'].set_color('none')

        ax_up.axes.xaxis.set_visible(False)
        ax_up.axes.yaxis.set_visible(False)
        if idx != 4:
            ax_bottom.axes.yaxis.set_visible(False)
            ax_bottom.axes.xaxis.set_visible(False)
        ax_up.tick_params(labelsize=8) 
        ax_features.append(ax_bottom)
        ax_erp.append(ax_up)


    for k in range(4):
        file = './latency/results/importance/step{}.pkl'.format(k)
        with open(file, "rb") as f:
            dict = pickle.load(f)
        primed, unprimed = dict[0], dict[1]
        all_attr = np.zeros((41, 300))

        for attr in primed:
            all_attr += preprocessing.MinMaxScaler().fit_transform(attr)
        # for attr in unprimed:
        #     all_attr += (1-preprocessing.MinMaxScaler().fit_transform(attr))

        # all_attr /= (len(primed)+len(unprimed))
        all_attr /= len(primed)
        print(np.max(all_attr), np.min(all_attr))
        print(all_attr.shape)

        im = ax_features[k].pcolormesh(x, y, all_attr, cmap='seismic', shading='auto', vmin=0, vmax=1)
        ax_erp[k].plot(x, np.mean(all_attr[:25], 0), alpha=0.5, label='primed ERP (\'Pz\' electrode)', color='black')      
        # ax_erp[k].plot(x, np.mean(up_latency, 1), alpha=0.5, label='unprimed ERP (\'Pz\' electrode)', color='green')  
        ax_erp[k].set_title(name_lst[k], loc='left', pad=-10, fontsize=14)  
        
    ax_features[k].set_ylabel('Channels', fontsize='large')
    ax_features[k].set_xlabel('Time (s)', fontsize=15)
    x_major_locator=MultipleLocator(500)
    ax_features[k].xaxis.set_major_locator(x_major_locator)

    # for i in range(5):
    #     p_latency = np.load('./latency/results/behavioral_results/ERP_Pz/p_step{}.npy'.format(i))
    #     up_latency = np.load('./latency/results/behavioral_results/ERP_Pz/up_step{}.npy'.format(i))
    #     ax_erp[i].plot(x, np.mean(p_latency, 1), alpha=0.5, label='primed ERP (\'Pz\' electrode)', color='black')      
    #     ax_erp[i].plot(x, np.mean(up_latency, 1), alpha=0.5, label='unprimed ERP (\'Pz\' electrode)', color='green')  
    #     ax_erp[i].set_title(name_lst[i], loc='left', pad=-10, fontsize=14)     
    # ax_erp[0].legend(loc='upper right')
    # plt.suptitle('ERP and feature importance in different steps', fontsize=18, y=0.97)
    
    cbar = fig.colorbar(im, ax=[*ax_features, *ax_erp], ticks=mticker.MultipleLocator(1), pad=0.08)
    cbar.set_label('Feature Importance', rotation=270, labelpad=-40, fontsize=20)
    cbar.ax.tick_params(labelsize='large')
    cbar.ax.set_yticks([0, 1])
    cbar.ax.set_yticklabels(['primed\ncondition', 'unprimed\ncondition'], fontsize=17)
    # xreal = [0, 500, 1000, 1500]
    # xrange = [0, 0.5, 1.0, 1.5]
    # ax_features[k].set_xticks(xreal)
    # ax_features[k].set_xticklabels(xrange, size=8)
    # plt.savefig('features.svg')
    plt.show()



def importance_with_topomaps(k=0):

    file = './latency/results/importance/step{}.pkl'.format(k)

    with open(file, "rb") as f:
        dict = pickle.load(f)
    primed, unprimed = dict[0], dict[1]
    all_attr = np.zeros((41, 300))

    for attr in primed:
        all_attr += preprocessing.MinMaxScaler().fit_transform(attr)
    for attr in unprimed:
        all_attr += (1-preprocessing.MinMaxScaler().fit_transform(attr))

    all_attr /= (len(primed)+len(unprimed))
    print(np.max(all_attr), np.min(all_attr))

    evoked = mne.EvokedArray(all_attr-0.5, info, tmin=-0.1)
    # cnorm = cm.colors.Normalize(vmax=9e5, vmin=0)
    # evoked.plot_joint(picks='eeg', times=(0.28, 0.38, 0.64), show=False, ts_args={'gfp':True}, 
    #                 topomap_args={'average':0.164})
    times = np.arange(0.2, 0.9, 0.01)
    evoked.plot_topomap(times, ch_type="eeg", ncols=8, nrows="auto", show_names=True, image_interp="cubic")
    # fig, anim = evoked.animate_topomap(times=times, ch_type="eeg", frame_rate=2, blit=False)
    # plt.show()
    plt.savefig('features_topomap_{}.png'.format(k))
        


if __name__ == '__main__':
    # only_importance()
    # importance_erp()
    importance_with_topomaps(k=4)