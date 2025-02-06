import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.stats import ttest_1samp
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as mticker

mat_name_lst = {'step0':'original', 
                'step1':'st_c', 
                'step2':'st_c_synced',
                'step3':'st_c_synced_within_subject', 
                'step4':'aligned_amp',
                's_component': 'st_s',
                'r_component': 'st_r',
                'st_r_synced': 'st_r_synced',
}     

set = './behaviors/ERPs/info.set'
info = mne.read_epochs_eeglab(set).info

plt.rcParams['savefig.dpi']         = 300
plt.rcParams['svg.fonttype']        = 'none'
plt.rcParams['font.family']         = 'sans-serif'   
plt.rcParams['font.sans-serif']     = 'Arial' 
plt.rcParams['savefig.transparent'] = True
plt.rcParams['xtick.direction']     = 'in'
plt.rcParams['ytick.direction']     = 'in'
mm                                  = 1/25.4 # inch 和 毫米的转换
cmap                                = 'jet'
interpolation                       = 'quadric'
x_major_locator                     = MultipleLocator(0.5)
y_major_locator                     = MultipleLocator(30)


def a_plot_acc_10folds():
    '''
    calculate scores and plot 
    '''
    scores = np.zeros((5, 10))
    for step_idx in range(5):
        print(step_idx)
        for k_idx in range(10):
            score = np.load('./results/group_classification/combined/step{}/Fsp_{}.npy'.format(step_idx, k_idx))
            scores[step_idx, k_idx] = score
            print(score)
        print(step_idx, np.mean(scores, 1), np.std(scores, 1))

    # Data for the bar graph
    x = ['Original', 'Step1', 'Step2', 'Step3', 'Step4']
    y = [0.752, 0.702, 0.765, 0.687, 0.679]
    sem = [0.0062, 0.0109, 0.0086, 0.0099, 0.01]

    # Create a bar graph with error bars for SEM
    fig, ax = plt.subplots(figsize=(80*mm, 40*mm))
    bar = ax.bar(x, y, yerr=sem, capsize=3, color='#696969')
    plt.bar_label(bar, label_type='edge', padding=5, fontsize=8, fmt='%.2f')

    ax.set_ylim(0.5, 0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylabel('accuracy', fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=6)
    # plt.show()
    plt.savefig('./latency/paper_figures/fig3/acc.svg', bbox_inches='tight', transparent=True)
    plt.close()

a_plot_acc_10folds()

def a_cal_diff_significance():
    '''
    calculate scores and plot 
    '''
    scores = np.zeros((5, 10))
    diff_scores = np.zeros((4, 10))
    for step_idx in range(5):
        print(step_idx)
        for k_idx in range(10):
            score = np.load('./latency/results/classification/general/step{}/{}.npy'.format(step_idx, k_idx))
            scores[step_idx, k_idx] = score
    
    for idx in range(4):
        diff_scores[idx] = scores[idx+1] - scores[idx]
        print(diff_scores[idx])
        t_test, p_value = ttest_1samp(diff_scores[idx], 0)
        # print(t_test)
        # print(p_value)
    
    diff_scores_02 = scores[2] - scores[0]
    t_test, p_value = ttest_1samp(diff_scores_02, 0)
    print(t_test)
    print(p_value)

a_cal_diff_significance()


def b_plot_temporal_decoding():
    scores = np.zeros((5, 10, 216))
    for step_idx in range(5):
        for k_idx in range(10):
            if step_idx <=3:
                score = np.load('./latency/results/classification/mvpa/step{}/ge_{}.npy'.format(step_idx, k_idx))[30:246, 30:246]
            else:
                score = np.load('./latency/results/classification/mvpa/step{}/ge_{}.npy'.format(step_idx, k_idx))
            diag = np.diag(score)
            scores[step_idx, k_idx] = diag

    # # Create a bar graph with error bars for SEM
    fig, ax = plt.subplots(figsize=(80*mm, 40*mm))
    mean_acc = np.mean(scores, axis=1)
    std_acc = np.std(scores, axis=1)
    x = np.linspace(0, 215, num=216)
    for idx, (label, color) in enumerate(zip(['Baseline', 'Step1', 'Step2', 'Step3', 'Step4'], ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC'])):
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
    ax.set_xticklabels(xrange, size=8)
    ax.set_ylim(0.5, 0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel('Time (s)', fontsize=8)
    ax.set_ylabel('accuracy', fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    plt.legend(fontsize=8)
    # plt.show()
    plt.savefig('./latency/paper_figures/fig3/time.svg', bbox_inches='tight', transparent=True)
    plt.close()


  
def c_plot_ge():
    k_fold = 10
    fig, ax = plt.subplots(1, 5, figsize=(150*mm, 40*mm), sharex=False, sharey=True, constrained_layout=True)

    diff_max = np.abs(0.7-0.5) 
    vmin, vmax = 0.5-diff_max, 0.5+diff_max 
    
    ge_path    = './latency/results/classification/mvpa/'
    for step_idx in range(0, 4):

        EEGNet_ge = np.zeros((k_fold, 271, 271))
        for k in range(k_fold):
            EEGNet_ge[k] = np.load('{}/step{}/ge_{}.npy'.format(ge_path, step_idx, k))
        
        ge_mean_all = EEGNet_ge.mean(axis=0)
        ge_mean = ge_mean_all[30:246, 30:246]
        print(ge_mean.shape)
        print(np.max(ge_mean), np.min(ge_mean))
        im = ax[step_idx].imshow(ge_mean, cmap='seismic', interpolation='gaussian', origin='lower', extent=[0.22, 1.08, 0.22, 1.08], vmin=vmin, vmax=vmax)
        ax[step_idx].xaxis.set_major_locator(MultipleLocator(0.5))
        ax[step_idx].tick_params(labelsize=6) 
        if step_idx != 0:
            ax[step_idx].axes.yaxis.set_visible(False)

    EEGNet_ge = np.zeros((k_fold, 216, 216))
    for k in range(k_fold):
        EEGNet_ge[k] = np.load('{}/step{}/ge_{}.npy'.format(ge_path, 4, k))
    ge_mean = EEGNet_ge.mean(axis=0)
    print(ge_mean.shape)
    print(np.max(ge_mean), np.min(ge_mean))
    im = ax[4].imshow(ge_mean, cmap='seismic', interpolation='gaussian', origin='lower', extent=[0.22, 1.08, 0.22, 1.08], vmin=vmin, vmax=vmax)
    ax[4].xaxis.set_major_locator(MultipleLocator(0.5))
    ax[4].tick_params(labelsize=6) 

    cbar = fig.colorbar(im, ax=ax[4], ticks=mticker.MultipleLocator(0.1), fraction=0.05)
    cbar.set_label('accuracy', rotation=270, labelpad=10, fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    ax[0].set_xlabel('Generalization Time (s)', fontsize=8)
    ax[0].set_ylabel('Training Time (s)', fontsize=8)
    ax[0].yaxis.set_major_locator(MultipleLocator(0.5))

    # plt.show() 
    plt.savefig('./latency/paper_figures/fig3/mvpa.svg', bbox_inches='tight', transparent=True)
    plt.close()



