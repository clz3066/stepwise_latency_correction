import mne
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
import matplotlib.ticker as mticker
from matplotlib.pyplot import MultipleLocator

plt.rcParams['savefig.dpi']         = 300
plt.rcParams['svg.fonttype']        = 'none'
plt.rcParams['font.family']         = 'sans-serif'   
plt.rcParams['font.sans-serif']     = 'Arial' 
plt.rcParams['savefig.transparent'] = True
plt.rcParams['xtick.direction']     = 'in'
plt.rcParams['ytick.direction']     = 'in'

set             = './ERP_results/info.set'
info            = mne.read_epochs_eeglab(set).info
mm              = 1/25.4 # inch 和 毫米的转换
y               = np.linspace(1, 41, 41)
y_major_locator = MultipleLocator(20)


def step_importance_with_topomaps(step_idx=0):
    fig = plt.figure(figsize=(8, 4))

    grid = plt.GridSpec(3, 7, hspace=0.1, left=0.1, bottom=0.1)
    ax_topo1 = plt.subplot(grid[0, 2])
    ax_topo2 = plt.subplot(grid[0, 3]) 
    ax_features = plt.subplot(grid[1:3, :])

    file = './latency/results/importance/step{}.pkl'.format(step_idx)
    with open(file, "rb") as f:
        dict = pickle.load(f)
    primed, unprimed = dict[0], dict[1]
    all_attr = np.zeros((41, 245))

    for attr in primed:
        all_attr += preprocessing.MinMaxScaler().fit_transform(attr)
    for attr in unprimed:
        all_attr += (1-preprocessing.MinMaxScaler().fit_transform(attr))

    all_attr /= (len(primed)+len(unprimed))
    print(np.max(all_attr), np.min(all_attr))
    im = ax_features.pcolormesh(x, y, all_attr, cmap='seismic', shading='auto', vmin=0, vmax=1)
    
    xreal = [200, 400, 600, 800, 1000]
    xrange = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax_features.set_xticks(xreal)
    ax_features.set_xticklabels(xrange, size=6)
    ax_features.set_xlabel('Time (s)', fontsize=8)
    ax_features.set_ylabel('Channels', fontsize=8)
    ax_features.yaxis.set_major_locator(y_major_locator) 
    ax_features.yaxis.set_tick_params(labelsize=6)

    evoked = mne.EvokedArray(all_attr, info, tmin=0.1)
    if step_idx < 3:
        evoked.plot_topomap(ch_type="eeg", times=(0.4, 0.64), average=0.164, scalings=1, vlim=(0, 1), cmap='seismic', contours=0,
                            time_format='%0.2f s', axes=[ax_topo1, ax_topo2], colorbar=False)
    else:
        evoked.plot_topomap(ch_type="eeg", times=(0.32, 0.58), average=0.164, scalings=1, vlim=(0, 1), cmap='seismic', contours=0,
                            time_format='%0.2f s', axes=[ax_topo1, ax_topo2], colorbar=False)    


    cbar = fig.colorbar(im, ax=ax_features, ticks=mticker.MultipleLocator(1), pad=0.03)
    cbar.set_label('Feature Importance', rotation=270, labelpad=-30, fontsize=8)
    cbar.ax.set_yticks([0, 1])
    cbar.ax.set_yticklabels(['primed\ncondition', 'unprimed\ncondition'], fontsize=8)

    plt.tight_layout()
    plt.show()
    # plt.savefig('./latency/figures/fig4/features_step{}.svg'.format(step_idx), bbox_inches='tight')



def step_importance_with_Pz(step_idx=0):
    fig = plt.figure(figsize=(120*mm, 40*mm))
    grid = plt.GridSpec(3, 5, wspace=0.3, hspace=0.05, left=0.1, bottom=0.1)
    
    ax_one = plt.subplot(grid[0, 0:4])
    ax_one.spines.top.set_visible(False)
    ax_one.spines.right.set_visible(False)
    ax_one.spines.bottom.set_visible(False)
    ax_one.axes.xaxis.set_visible(False)
    ax_one.tick_params(labelsize=6) 
    ax_one.set_ylabel('$\mu V$', fontsize=8)
    x = np.arange(0.11, 1.089, 0.004)
    mat = scio.loadmat('./ERP_results/ERP/Fsp_step{}.mat'.format(step_idx))
    ax_one.plot(x, mat['p'][27, 49:294], label='primed ERP (Pz)', alpha=0.5, color='black')
    ax_one.plot(x, mat['up'][27, 49:294], label='unprimed ERP (Pz)', alpha=0.5, color='green')
    ax_one.set_xlim([0.11, 1.089])
    if step_idx == 0:
        ax_one.legend(loc='upper right', fontsize=8)
        ax_one.set_ylim([-2, 12])
        ax_one.set_title('Baseline', fontsize=8)
    elif step_idx <= 3:
        ax_one.set_ylim([-2, 12])
        ax_one.set_title('Step{}'.format(step_idx), fontsize=8)
    elif step_idx == 4: 
        ax_one.set_ylim([-0.2, 1.5])
        ax_one.set_title('Step{}'.format(step_idx), fontsize=8)

    ax_features = plt.subplot(grid[1:3, 0:4])
    file = './latency/results/importance/step{}.pkl'.format(step_idx)
    with open(file, "rb") as f:
        dict = pickle.load(f)
    primed, unprimed = dict[0], dict[1]
    all_attr = np.zeros((41, 245))

    for attr in primed:
        all_attr += preprocessing.MinMaxScaler().fit_transform(attr)
    for attr in unprimed:
        all_attr += (1-preprocessing.MinMaxScaler().fit_transform(attr))

    all_attr /= (len(primed)+len(unprimed))
    print(np.max(all_attr), np.min(all_attr))
    x = np.arange(110, 1089, 4)
    im = ax_features.pcolormesh(x, y, all_attr, cmap='seismic', shading='auto', vmin=0, vmax=1, rasterized=True)
    
    xreal = [200, 400, 600, 800, 1000]
    xrange = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax_features.set_xticks(xreal)
    ax_features.set_xticklabels(xrange, size=6)
    ax_features.set_xlabel('Time (s)', fontsize=8)
    ax_features.set_ylabel('Channels', fontsize=8)
    ax_features.yaxis.set_major_locator(y_major_locator) 
    ax_features.yaxis.set_tick_params(labelsize=6)

    ax_colorbar = plt.subplot(grid[1:3, 4])
    cbar = fig.colorbar(im, ax=ax_colorbar, ticks=mticker.MultipleLocator(1), pad=0.03)
    cbar.set_label('Feature Importance', rotation=270, labelpad=-30, fontsize=8)
    cbar.ax.set_yticks([0, 1])
    cbar.ax.set_yticklabels(['primed\ncondition', 'unprimed\ncondition'], fontsize=8)

    plt.tight_layout()
    # plt.show()
    plt.savefig('./latency/paper_figures/fig4/features_step{}.svg'.format(step_idx), bbox_inches='tight', transparent=True)
    plt.close()

step_importance_with_Pz(0)
step_importance_with_Pz(1)
step_importance_with_Pz(2)
step_importance_with_Pz(3)
step_importance_with_Pz(4)