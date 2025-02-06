import scipy.io as scio
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from matplotlib import cm
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as mticker
from scipy.stats import ttest_1samp
import pickle
from sklearn import preprocessing

mat_name_lst = {'step0':'original', 
                'step1':'st_c', 
                'step2':'st_c_synced',
                'step3':'st_c_synced_within_subject', 
                'step4':'aligned_amp',
                's_component': 'st_s',
                'r_component': 'st_r',
                'st_r_synced': 'st_r_synced',
}     

set = './ERP_results/info.set'
info = mne.read_epochs_eeglab(set).info

plt.rcParams['svg.fonttype']        = 'none'
plt.rcParams['font.family']         = 'sans-serif'   
plt.rcParams['font.sans-serif']     = 'Arial' 
plt.rcParams['savefig.transparent'] = True
plt.rcParams['xtick.direction']     = 'in'
plt.rcParams['ytick.direction']     = 'in'
mm                                  = 1/25.4 # inch 和 毫米的转换
x_major_locator                     = MultipleLocator(0.5)
colorbar_locator                    = mticker.MultipleLocator(30)
cmap                                = 'jet'
interpolation                       = 'gaussian'
data_path                           = '../../Dataset/Facc_Fsp/'
data_files                          = os.listdir(data_path)


def ERP_RIDE_components():

    fig, axs = plt.subplots(2, 3, figsize=(70*mm, 70*mm))

    mat = scio.loadmat('./ERP_results/ERP/Fsp_s_component.mat')
    evoked = mne.EvokedArray(mat['p'], info, tmin=-0.1)
    evoked.plot(picks='eeg', show=False, titles=None, scalings=dict(eeg=1), gfp=True, axes=axs[0, 0]) 
    evoked = mne.EvokedArray(mat['up'], info, tmin=-0.1)
    evoked.plot(picks='eeg', show=False, titles=None, scalings=dict(eeg=1), gfp=True, axes=axs[1, 0])   

    mat = scio.loadmat('./ERP_results/ERP/Fsp_step1.mat')
    evoked = mne.EvokedArray(mat['p'], info, tmin=-0.1)
    evoked.plot(picks='eeg', show=False, titles=None, scalings=dict(eeg=1), gfp=True, axes=axs[0, 1]) 
    evoked = mne.EvokedArray(mat['up'], info, tmin=-0.1)
    evoked.plot(picks='eeg', show=False, titles=None, scalings=dict(eeg=1), gfp=True, axes=axs[1, 1])  

    mat = scio.loadmat('./ERP_results/ERP/Fsp_r_component.mat')
    evoked = mne.EvokedArray(mat['p'], info, tmin=-0.1)
    evoked.plot(picks='eeg', show=False, titles=None, scalings=dict(eeg=1), gfp=True, axes=axs[0, 2]) 
    evoked = mne.EvokedArray(mat['up'], info, tmin=-0.1)
    evoked.plot(picks='eeg', show=False, titles=None, scalings=dict(eeg=1), gfp=True, axes=axs[1, 2])  

    # plt.show()
    plt.savefig('./latency/paper_figures/SI/fig1.svg', bbox_inches='tight', transparent=True)
    plt.close()


def ERP_step4_part1():
    '''
    only for colorbar
    '''
    mat = scio.loadmat('./ERP_results/ERP/Fsp_step4_part1.mat')
    erp = mat['erp'][:, 49:294]
    print(erp.shape)
    
    evoked = mne.EvokedArray(erp, info, tmin=0.11)
    print(np.min(evoked.data))
    print(np.max(evoked.data))

    cnorm = cm.colors.Normalize(vmax=3.8, vmin=-3.8)
    fig = evoked.plot_joint(picks='eeg', times=(0.32, 0.58), title=None, show=False,
                    ts_args={'gfp':True, 'scalings':dict(eeg=1), 'ylim':dict(eeg=[-3.8, 3.8]), 'gfp':True}, 
                    topomap_args={'average':0.124, 'scalings':dict(eeg=1), 'time_format':'%0.2f s', 'cnorm':cnorm}, 
                    highlight=[(0.26, 0.38), (0.52, 0.64)])
    fig.set_size_inches((6, 4))
   
    # plt.show()
    plt.savefig('./latency/paper_figures/SI/erp_step4_part1.svg', bbox_inches='tight', transparent=True)
    plt.close()


def get_s_r_accuracy():
    '''
    calculate scores and plot 
    '''
    scores = np.zeros((2, 10))
    for k_idx in range(10):
        score = np.load('./latency/results/classification/general/s_component/{}.npy'.format(k_idx))
        scores[0, k_idx] = score
        print(score)
        score = np.load('./latency/results/classification/general/r_component/{}.npy'.format(k_idx))
        scores[1, k_idx] = score
        print(score)
    
    print(np.mean(scores[0]), np.std(scores[0]))
    print(np.mean(scores[1]), np.std(scores[1]))

    # Data for the bar graph
    x = ['Baseline', 'Step1', 'Step2', 'Step3', 'Step4']
    y = [0.752, 0.702, 0.765, 0.687, 0.679]
    sem = [0.0062, 0.0109, 0.0086, 0.0099, 0.01]

    # Create a bar graph with error bars for SEM
    fig, ax = plt.subplots(figsize=(80*mm, 40*mm))
    bar = ax.bar(x, y, yerr=sem, capsize=3, color='black', alpha=0.7)
    plt.bar_label(bar, label_type='edge', padding=5, fontsize=8, fmt='%.2f')

    ax.set_ylim(0.5, 0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylabel('accuracy', fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    plt.show()
    # plt.savefig('./latency/paper_figures/fig3/acc.svg', bbox_inches='tight', transparent=True)
    # plt.close()

# get_s_r_accuracy()

def plot_s_r_accuracy():
    scores = np.zeros((2, 10, 216))
    for k_idx in range(10):
        score = np.load('./latency/results/classification/mvpa/s_component/ge_{}.npy'.format(k_idx))
        diag = np.diag(score)
        scores[0, k_idx] = diag

        score = np.load('./latency/results/classification/mvpa/r_component/ge_{}.npy'.format(k_idx))
        diag = np.diag(score)
        scores[1, k_idx] = diag    

    # # Create a bar graph with error bars for SEM
    fig, ax = plt.subplots(figsize=(60*mm, 40*mm))
    mean_acc = np.mean(scores, axis=1)
    std_acc = np.std(scores, axis=1)
    x = np.linspace(0, 215, num=216)
    for idx, (label, color) in enumerate(zip(['S component (0.62 $\pm$ 0.037)', 'R component (0.69 $\pm$ 0.004)'], ['#6C7BBB', '#FFA500'])):
        random = np.ones((1, 216)) * 0.5
        # Calculate the t-statistic and p-value
        t_statistic, p_value = ttest_1samp(scores[idx], random)
        p_boolean = (p_value <= 0.001)
        p_boolean = p_boolean.squeeze()
        ax.plot(x, mean_acc[idx], label=label, color=color, linewidth=2.0)
        ax.fill_between(x, mean_acc[idx]-std_acc[idx], mean_acc[idx]+std_acc[idx], where=p_boolean, facecolor=color, alpha=0.3)

    xreal = [10, 60, 110, 160, 210]
    xrange = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(xreal)
    ax.set_xticklabels(xrange, size=6)
    ax.set_ylim(0.5, 0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel('Time (s)', fontsize=8)
    ax.set_ylabel('accuracy', fontsize=8)
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    plt.legend(fontsize=8)
    plt.show()
    fig.savefig('./latency/paper_figures/SI/time.svg', bbox_inches='tight', transparent=True)
    plt.close()


def c_plot_ge():
    k_fold = 10
    fig, ax = plt.subplots(1, 2, figsize=(100*mm, 40*mm), sharex=False, sharey=True, constrained_layout=True)

    diff_max = np.abs(0.7-0.5) 
    vmin, vmax = 0.5-diff_max, 0.5+diff_max 
    
    ge_path    = './latency/results/classification/mvpa/'

    EEGNet_ge = np.zeros((k_fold, 216, 216))
    for k in range(k_fold):
        EEGNet_ge[k] = np.load('{}/s_component/ge_{}.npy'.format(ge_path, k))
    ge_mean = EEGNet_ge.mean(axis=0)
    print(ge_mean.shape)
    print(np.max(ge_mean), np.min(ge_mean))
    ax[0].imshow(ge_mean, cmap='seismic', interpolation='gaussian', origin='lower', extent=[0.22, 1.08, 0.22, 1.08], vmin=vmin, vmax=vmax)
    ax[0].xaxis.set_major_locator(MultipleLocator(0.5))
    ax[0].tick_params(labelsize=6) 

    EEGNet_ge = np.zeros((k_fold, 216, 216))
    for k in range(k_fold):
        EEGNet_ge[k] = np.load('{}/r_component/ge_{}.npy'.format(ge_path, k))
    ge_mean = EEGNet_ge.mean(axis=0)
    print(ge_mean.shape)
    print(np.max(ge_mean), np.min(ge_mean))
    im = ax[1].imshow(ge_mean, cmap='seismic', interpolation='gaussian', origin='lower', extent=[0.22, 1.08, 0.22, 1.08], vmin=vmin, vmax=vmax)
    ax[1].xaxis.set_major_locator(MultipleLocator(0.5))
    ax[1].tick_params(labelsize=6) 

    cbar = fig.colorbar(im, ax=ax[1], ticks=mticker.MultipleLocator(0.1), fraction=0.05)
    cbar.set_label('accuracy', rotation=270, labelpad=10, fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    ax[0].set_xlabel('Generalization Time (s)', fontsize=8)
    ax[0].set_ylabel('Training Time (s)', fontsize=8)
    ax[0].yaxis.set_major_locator(MultipleLocator(0.5))

    plt.show() 
    fig.savefig('./latency/paper_figures/SI/ge.svg', bbox_inches='tight', transparent=True)
    plt.close()

# c_plot_ge()


def s_r_features(component='s'):
    fig = plt.figure(figsize=(80*mm, 40*mm))
    grid = plt.GridSpec(3, 5, wspace=0.3, hspace=0.05, left=0.1, bottom=0.1)
    
    ax_one = plt.subplot(grid[0, 0:4])
    ax_one.spines.top.set_visible(False)
    ax_one.spines.right.set_visible(False)
    ax_one.spines.bottom.set_visible(False)
    ax_one.axes.xaxis.set_visible(False)
    ax_one.tick_params(labelsize=6) 
    ax_one.set_ylabel('$\mu V$', fontsize=8)
    x = np.arange(0.11, 1.089, 0.004)
    y = np.linspace(1, 41, 41)
    mat = scio.loadmat('./ERP_results/ERP/Fsp_{}_component.mat'.format(component))
    ax_one.plot(x, mat['p'][27, 49:294], label='primed ERP (Pz)', color='black')
    ax_one.plot(x, mat['up'][27, 49:294], label='unprimed ERP (Pz)', color='black', linestyle='dashed')
    ax_one.set_xlim([0.11, 1.089])

    # ax_one.set_ylim([-0.2, 1.5])
    # ax_one.set_title('Step{}'.format(step_idx), fontsize=8)

    ax_features = plt.subplot(grid[1:3, 0:4])
    all_attr = np.zeros((41, 245))
    len_primed, len_unprimed = 0, 0
    for k in range(10):
        file = './latency/results/importance/{}_component_{}.pkl'.format(component, k)
        with open(file, "rb") as f:
            dict = pickle.load(f)
            primed, unprimed = dict[0], dict[1]
            len_primed += len(primed)
            len_unprimed += len(unprimed)
            for attr in primed:
                all_attr += preprocessing.MinMaxScaler().fit_transform(attr)
            for attr in unprimed:
                all_attr += (1-preprocessing.MinMaxScaler().fit_transform(attr))

    all_attr /= (len_primed+len_unprimed)
    print(np.max(all_attr), np.min(all_attr))
    x = np.arange(110, 1089, 4)
    im = ax_features.pcolormesh(x, y, all_attr, cmap='seismic', shading='auto', vmin=0, vmax=1, rasterized=True)
    
    xreal = [200, 400, 600, 800, 1000]
    xrange = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax_features.set_xticks(xreal)
    ax_features.set_xticklabels(xrange, size=6)
    ax_features.set_xlabel('Time (s)', fontsize=8)
    ax_features.set_ylabel('Channels', fontsize=8)
    # ax_features.yaxis.set_major_locator(y_major_locator) 
    ax_features.yaxis.set_tick_params(labelsize=6)

    ax_colorbar = plt.subplot(grid[1:3, 4])
    cbar = fig.colorbar(im, ax=ax_colorbar, ticks=mticker.MultipleLocator(1), pad=0.03)
    cbar.set_label('Feature Importance', rotation=270, labelpad=-30, fontsize=8)
    cbar.ax.set_yticks([0, 1])
    cbar.ax.set_yticklabels(['primed\ncondition', 'unprimed\ncondition'], fontsize=8)

    # plt.show()
    fig.savefig('./latency/paper_figures/SI/features_{}_component.svg'.format(component), bbox_inches='tight', transparent=True)
    plt.close()

s_r_features('s')
s_r_features('r')