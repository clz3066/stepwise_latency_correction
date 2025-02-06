import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import statsmodels.stats.api as sms
import mne
from matplotlib import cm

set = './behaviors/ERPs/info.set'
info = mne.read_epochs_eeglab(set).info

data_path                           = '../../Dataset/Facc_Fsp/'
mm                                  = 1/25.4 # inch 和 毫米的转换
plt.rcParams['svg.fonttype']        = 'none'
plt.rcParams['font.family']         = 'sans-serif'   
plt.rcParams['font.sans-serif']     = 'Arial' 
plt.rcParams['savefig.transparent'] = True
plt.rcParams['xtick.direction']     = 'in'
plt.rcParams['ytick.direction']     = 'in'


def a_ERP_diff(task='Fsp'):

    mat = scio.loadmat('./results/ERPs/{}_step1_400.mat'.format(task))
    erp = mat['erp'][:, 49:275]
    print(erp.shape)
    
    evoked = mne.EvokedArray(erp, info, tmin=0.1)
    print(np.min(evoked.data))
    print(np.max(evoked.data))
    # cnorm = cm.colors.Normalize(vmax=3.8, vmin=-3.8)
    fig = evoked.plot_joint(picks='eeg', times=(0.295, 0.405, 0.675), title=None, show=False,
                    ts_args={'gfp':True, 'scalings':dict(eeg=1), 'gfp':True, 'ylim':dict(eeg=[-3.8, 3.8])}, 
                    topomap_args={'average':[0.074, 0.154, 0.214], 'scalings':dict(eeg=1), 'time_format':'%0.2f s'}, 
                    highlight=[(0.26, 0.33), (0.34, 0.48), (0.55, 0.80)])
    # fig.set_size_inches((6, 4))
    plt.show()
    # fig.savefig('./individual_level/paper_figures/fig4/a_{}_erp_original.svg'.format(task), bbox_inches='tight', transparent=True)
    plt.close()


def a_ERP(task='Fsp', step='s_component'):

    mat = scio.loadmat('./results/ERPs/{}_{}_400.mat'.format(task, step))
    erp = mat['erp'][:, 49:275]
    print(erp.shape)
    
    evoked = mne.EvokedArray(erp, info, tmin=0.1)
    print(np.min(evoked.data))
    print(np.max(evoked.data))
    # cnorm = cm.colors.Normalize(vmax=3.8, vmin=-3.8)
    fig = evoked.plot_joint(picks='eeg', times=(0.295), title=None, show=False,
                    ts_args={'gfp':True, 'scalings':dict(eeg=1), 'gfp':True}, 
                    topomap_args={'average':[0.074], 'scalings':dict(eeg=1), 'time_format':'%0.2f s'}, 
                    highlight=[(0.26, 0.33)])
    # fig.set_size_inches((6, 4))
    plt.show()
    # fig.savefig('./individual_level/paper_figures/fig4/a_{}_erp_original.svg'.format(task), bbox_inches='tight', transparent=True)
    # plt.close()
# a_ERP('Fsp', 'step1')


def b_save_mvpa_acc():

    output_file = './results/individual_classification/mvpa_scores_s_r.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        for task in ['Facc', 'Fsp']:
            # for step_idx in range(5):
            for step_idx in ['s', 'r']:
                new_df = {}
                # name = 'mvpa_{}_step{}'.format(task, step_idx)
                name = 'mvpa_{}_{}_component'.format(task, step_idx)
                input_file = './results/individual_sem/{}.xlsx'.format(name)
                df = pd.read_excel(input_file)
                new_df['subjects'] = df['subjects']

                for time_idx in range(4):
                    scores = (df['t{}0'.format(time_idx)] + df['t{}1'.format(time_idx)] + df['t{}2'.format(time_idx)])/3
                    new_df['t{}'.format(time_idx)] = scores

                new_df = pd.DataFrame(new_df)
                new_df.to_excel(writer, index=False, sheet_name=name)


def b_plot_mvpa_acc_both_tasks():

    file = './results/individual_classification/mvpa_scores.xlsx'
    fig, ax = plt.subplots(1, 2, figsize=(180*mm, 40*mm))
    x = np.linspace(0, 3, num=4)
    Fsp_colors = ['#ef5f38', '#f6df84', '#77c893', '#3a8ca6', '#3d4c96', 'black']
    Facc_colors = Fsp_colors
    # Fsp_colors  = ['#d1e2ef', '#7fa9cd', '#527cb4', '#0000cd',  '#151752']
    # Facc_colors = ['#fdc0ad', '#fc7852', '#dc5c60', '#cb2e12',  '#5e1206']
    # Facc_colors = ['#fdc0ad', '#fc7852', '#dc5c60', '#cb2e12',  '#5e1206']
    for step_idx, name in enumerate([0, 1, 2, 3, 4, '_S']):
        sheet_name = 'mvpa_Fsp_step{}'.format(name)
        df = pd.read_excel(file, sheet_name=sheet_name)
        mean = []
        for time_idx in range(4):
            scores = df['t{}'.format(time_idx)]
            mean.append(np.mean(scores))
            lower, upper = sms.DescrStatsW(scores).tconfint_mean()
            print(time_idx, lower, upper)
            ax[0].plot([time_idx, time_idx], [lower, upper], color=Fsp_colors[step_idx], linewidth=1.0, alpha=0.3)
        if step_idx == 0:
            ax[0].plot(x, np.array(mean), marker='o', markersize=1, linestyle='solid', color=Fsp_colors[step_idx], label='Original'.format(step_idx), linewidth=1.0)
        else:
            ax[0].plot(x, np.array(mean), marker='o', markersize=1, linestyle='solid', color=Fsp_colors[step_idx], label='Step{}'.format(step_idx), linewidth=1.0)
        

    for step_idx, name in enumerate([0, 1, 2, 3, 4, '_S']):
        sheet_name = 'mvpa_Facc_step{}'.format(name)
        df = pd.read_excel(file, sheet_name=sheet_name)
        mean = []
        for time_idx in range(4):
            scores = df['t{}'.format(time_idx)]
            mean.append(np.mean(scores))
            lower, upper = sms.DescrStatsW(scores).tconfint_mean()
            ax[1].plot([time_idx, time_idx], [lower, upper], color=Facc_colors[step_idx], alpha=0.3)
        if step_idx == 0:
            ax[1].plot(x, np.array(mean), marker='o', markersize=1, linestyle='solid', color=Facc_colors[step_idx], label='Original'.format(step_idx), linewidth=1.0)
        else:
            ax[1].plot(x, np.array(mean), marker='o', markersize=1, linestyle='solid', color=Facc_colors[step_idx], label='Step{}'.format(step_idx), linewidth=1.0)
        

    for idx in range(2):
        ax[idx].set_ylim(0.48, 0.73)
        ax[idx].spines['top'].set_visible(False)
        ax[idx].spines['right'].set_visible(False)
        ax[idx].yaxis.set_major_locator(MultipleLocator(0.1))
        ax[idx].set_xlabel('time windows (ms)', fontsize=8)
        ax[idx].set_ylabel('classification accuracy', fontsize=8)
        xrange = ['[100, 260]', '[260, 330]', '[330, 480]', '[570, 780]']
        ax[idx].set_xticks(x)
        ax[idx].set_xticklabels(xrange, size=6)
        ax[idx].tick_params(axis='x', labelsize=6)
        ax[idx].tick_params(axis='y', labelsize=6)
    
    ax[1].legend(fontsize=8)
    plt.show()
    # fig.savefig('./individual_level/paper_figures/fig4/b_mvpa_acc.svg', bbox_inches='tight', transparent=True)
    plt.close()


def b_plot_mvpa_acc_one_task(task='Facc'):

    file = './results/individual_classification/mvpa_scores.xlsx'
    fig, ax = plt.subplots(1, 2, figsize=(180*mm, 60*mm))
    x = np.linspace(0, 2, num=3)
    Fsp_colors = ['#ef5f38', '#f6df84', '#77c893', '#3a8ca6', '#3d4c96', 'black']
    for step_idx, name in enumerate([0, 1, 2, 3, 4, '_S']):
        sheet_name = 'mvpa_{}_step{}'.format(task, name)
        df = pd.read_excel(file, sheet_name=sheet_name)
        mean = []
        for time_idx in range(3):
            scores = df['t{}'.format(time_idx)]
            mean.append(np.mean(scores))
            lower, upper = sms.DescrStatsW(scores).tconfint_mean()
            print(time_idx, lower, upper)
            ax[0].plot([time_idx, time_idx], [lower, upper], color=Fsp_colors[step_idx], linewidth=1.0, alpha=0.3)
        if step_idx == 0:
            ax[0].plot(x, np.array(mean), marker='o', markersize=1, linestyle='solid', color=Fsp_colors[step_idx], label='Original'.format(step_idx), linewidth=1.0)
        else:
            ax[0].plot(x, np.array(mean), marker='o', markersize=1, linestyle='solid', color=Fsp_colors[step_idx], label='Step{}'.format(step_idx), linewidth=1.0)
        

    for idx in range(2):
        ax[idx].set_ylim(0.48, 0.73)
        ax[idx].spines['top'].set_visible(False)
        ax[idx].spines['right'].set_visible(False)
        ax[idx].yaxis.set_major_locator(MultipleLocator(0.1))
        ax[idx].set_xlabel('time windows (ms)', fontsize=8)
        ax[idx].set_ylabel('classification accuracy', fontsize=8)
        xrange = ['[260, 330]', '[330, 480]', '[570, 780]']
        ax[idx].set_xticks(x)
        ax[idx].set_xticklabels(xrange, size=6)
        ax[idx].tick_params(axis='x', labelsize=6)
        ax[idx].tick_params(axis='y', labelsize=6)
    
    ax[0].legend(fontsize=8, ncols=2)
    plt.show()
    fig.savefig('./individual_level/paper_figures/fig5/b_mvpa_acc_{}.svg'.format(task), bbox_inches='tight', transparent=True)
    plt.close()
# b_plot_mvpa_acc_one_task()

def plot_topomaps_diff(task='Facc'):
    mat = scio.loadmat('./results/ERPs/{}_step4.mat'.format(task))
    erp = mat['erp'][:, 49:275]
    print(erp.shape)
    cnorm = cm.colors.Normalize(vmax=0.25, vmin=-0.25)
    evoked = mne.EvokedArray(erp, info, tmin=0.1)
    # times = np.arange(0.20, 0.50, 0.01)

    times = [0.295, 0.405, 0.675]
    averages = [0.074, 0.154, 0.214]
    fig = evoked.plot_topomap(times, ch_type="eeg", average=averages, scalings=1, time_unit='ms', cnorm=cnorm)
    plt.show()
    fig.savefig('./individual_level/paper_figures/fig5/{}_step4.svg'.format(task), bbox_inches='tight', transparent=True)
    plt.close()
plot_topomaps_diff('Facc')

def plot_topomaps(task='Fsp'):
    mat = scio.loadmat('./results/ERPs/{}_step4.mat'.format(task))
    erp = mat['erp'][:, 49:275]
    print(erp.shape)
    evoked = mne.EvokedArray(erp, info, tmin=0.1)
    # times = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    times = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    cnorm = cm.colors.Normalize(vmax=1, vmin=-1)
    fig = evoked.plot_topomap(times, ch_type="eeg", scalings=1, time_unit='ms', nrows=1)
    plt.show()
    # fig.savefig('./individual_level/paper_figures/fig5/{}_unprimed_s.png'.format(task), bbox_inches='tight', transparent=True)
    # plt.close()

# plot_topomaps()


# def a_plot_reaction_acc2():
#     '''
#     with distributions (not used)
#     '''
#     input_file = './individual/paper_figures/plot_codes/reaction_time.xlsx'
#     df_sheet2 = pd.read_excel(input_file, sheet_name="Sheet2")
#     fig, ax = plt.subplots(figsize=(130*mm, 70*mm))


#     # txt2 = r'easy task: $\it{r}$ : %.2f $^{***}$' % (r2) 
#     sns.jointplot(data=df_sheet2, x="reaction time difference (primed - unprimed)", y="classification accuracy", hue="task", kind="scatter", xlim=(-520, 105), ylim=(0.42, 0.88))
#     # sns.regplot(x=x2, y=y2, ax=ax, robust=True, truncate=True, marker="x", scatter_kws={"color": "black", "alpha": 0.3, "s": 10}, line_kws={"color": "black"}, label=txt2)

#     ax.set_xlim(-520, 105)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.yaxis.set_major_locator(MultipleLocator(0.1))
#     ax.set_xlabel('reaction time difference (primed - unprimed)', fontsize=10)
#     ax.set_ylabel('classification accuracy', fontsize=10)
#     ax.set_axisbelow(True)
#     plt.legend(loc='upper right', fontsize=10)
#     # plt.savefig('./individual/plot_figures/rt_acc2.svg', bbox_inches='tight', transparent=True)
#     plt.show()

#     # stat, p = stats.normaltest(df_sheet1['Fsp_PF'] - df_sheet1['Fsp_UPF'])
#     # stat, p = stats.normaltest(df_sheet1['Facc_step0'])
#     # print(p)
#     # if p > 0.05:
#     #     print('Sample looks Gaussian (fail to reject H0)')
#     # else:
#     #     print('Sample does not look Gaussian (reject H0)')
#     # difficult task does not follow normal distribution, but easy task follows.




# def c_plot_mvpa_acc(task):
#     '''
#     not used
#     '''
#     file = './individual/paper_figures/plot_codes/mvpa_scores.xlsx'
#     fig, ax = plt.subplots(figsize=(100*mm, 60*mm))
#     x = np.linspace(0, 7, num=8)
#     colors = ['#ef5f38', '#f6df84', '#77c893', '#3a8ca6', '#3d4c96']
#     for step_idx in range(5):
#         sheet_name = 'mvpa_{}_step{}'.format(task, step_idx)
#         df = pd.read_excel(file, sheet_name=sheet_name)
#         mean = []
#         std = []
#         for time_idx in range(8):
#             scores = df['t{}'.format(time_idx)]
#             mean.append(np.mean(scores))
#             std.append(np.std(scores))
#         if step_idx == 0:
#             ax.plot(x, np.array(mean), marker='*', markersize=5, linestyle='solid', color=colors[step_idx], label='Original'.format(step_idx), linewidth=2.0)
#         else:
#             ax.plot(x, np.array(mean), marker='*', markersize=5, linestyle='solid', color=colors[step_idx], label='Step{}'.format(step_idx), linewidth=2.0)
#         ax.fill_between(x, np.array(mean)-np.array(std), np.array(mean)+np.array(std), alpha=0.1)
    
#     ax.set_ylim(0.48, 0.65)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.yaxis.set_major_locator(MultipleLocator(0.1))
#     ax.set_ylabel('classification accuracy', fontsize=8)
#     # xreal = [0, 10, 110, 160, 210]
#     xrange = ['[100, 220]', '[220, 340]', '[340, 460]', '[460, 580]', '[580, 700]', '[700, 820]', '[820, 940]', '[940, 1060]']
#     ax.set_xticks(x)
#     ax.set_xticklabels(xrange, size=6)
#     ax.tick_params(axis='x', labelsize=6)
#     ax.tick_params(axis='y', labelsize=6)
#     plt.legend(fontsize=8)
#     # plt.show()
#     plt.savefig('./individual/paper_figures/fig2/mvpa_acc_{}.svg'.format(task), bbox_inches='tight', transparent=True)
#     plt.close()
