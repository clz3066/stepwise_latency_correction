import scipy.io as scio
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as mticker
import random
import pandas as pd
import statsmodels.stats.api as sms

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


def a_readin_subject_name():
    files = os.listdir(data_path)
    random.shuffle(files)

    subject_name = []
    for file in files:
        if file.endswith('.mat'):
            content = file.split('.')[0].split('_')
            subject = content[0]
            if subject not in subject_name:
                subject_name.append(subject)
    print(len(subject_name))
    return subject_name


def a_readin_error_rates():
    subject_name = a_readin_subject_name()
    files = os.listdir(data_path)
    random.shuffle(files)

    task_name_lst = ['Facc_PF', 'Facc_UPF', 'Fsp_PF', 'Fsp_UPF']
    subject_val = np.zeros((185, 5))
    for idx, subject in enumerate(subject_name):
        for i, task_name in enumerate(task_name_lst):
            file_name = os.path.join(data_path, subject+'_'+task_name+'_cc.mat')
            mat = scio.loadmat(file_name)['results']
            rt = mat[0, 0]['rt'].shape[1]

            subject_val[idx, i] = rt
            subject_val[idx, 4] = subject
    scio.savemat('./results/individual_behaviors/error_rates.mat', {"data": subject_val})


def a_readin_reaction_times():
    subject_name = a_readin_subject_name()
    files = os.listdir(data_path)
    random.shuffle(files)

    task_name_lst = ['Facc_PF', 'Facc_UPF', 'Fsp_PF', 'Fsp_UPF']
    subject_val = np.zeros((185, 5))
    for idx, subject in enumerate(subject_name):
        for i, task_name in enumerate(task_name_lst):
            file_name = os.path.join(data_path, subject+'_'+task_name+'_cc.mat')
            mat = scio.loadmat(file_name)['results']
            rt = np.mean(mat[0, 0]['rt'])

            subject_val[idx, i] = rt
            subject_val[idx, 4] = subject
    scio.savemat('./results/individual_behaviors/reaction_time.mat', {"data": subject_val})



# def a_plot_behavior():
#     boxplot_columns = ['Fsp_PF', 'Fsp_UPF', 'Facc_PF', 'Facc_UPF']
#     paired_columns = [['Fsp_PF', 'Fsp_UPF'], ['Facc_PF', 'Facc_UPF']]
#     x1, x2, x3, x4 = 1, 2, 3, 4

#     fig, ax = plt.subplots(1, 2, figsize=(175*mm, 50*mm))

#     ####################### reaction time ###############################
#     input_file = './results/individual_behaviors/reaction_time.xlsx'
#     df = pd.read_excel(input_file, sheet_name="Sheet1")
#     boxplot = ax[0].boxplot(df[boxplot_columns].values, patch_artist=True, flierprops={'marker':'x', 'markersize':2})
#     colors = ['#334c81', '#a7413c', '#334c81', '#a7413c']
#     plt.setp(boxplot['medians'], color='black')
#     for patch, color in zip(boxplot['boxes'], colors):
#         patch.set_facecolor(color)

#     for pair in paired_columns:
#         x = [boxplot_columns.index(pair[0]) + 1, boxplot_columns.index(pair[1]) + 1]
#         y = [df[pair[0]], df[pair[1]]]
#         ax[0].plot(x, y, linestyle='--', color='gray', alpha=0.5, linewidth=0.3, markersize=1)
#     ax[0].set_ylabel('reaction time (ms)', fontsize=8)
#     y, h = df['Facc_UPF'].max()+100, 30
#     ax[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c="k") 
#     ax[0].text((x1+x2)*.5, y+h, "easy\ntask", ha='center', va='center', color="k", size=8)
#     ax[0].plot([x3, x3, x4, x4], [y, y+h, y+h, y], lw=1, c="k") 
#     ax[0].text((x3+x4)*.5, y+h, "difficult\ntask", ha='center', va='center', color="k", size=8)

#     input_file = './results/individual_behaviors/error_rates.xlsx'
#     df = pd.read_excel(input_file, sheet_name="Sheet1")
#     boxplot = ax[1].boxplot(1-df[boxplot_columns].values/72, patch_artist=True, flierprops={'marker':'x', 'markersize':2})
#     colors = ['#334c81', '#a7413c', '#334c81', '#a7413c']
#     for patch, color in zip(boxplot['boxes'], colors):
#         patch.set_facecolor(color)
#     plt.setp(boxplot['medians'], color='black')
#     for pair in paired_columns:
#         x = [boxplot_columns.index(pair[0]) + 1, boxplot_columns.index(pair[1]) + 1]
#         y = [1-df[pair[0]]/72, 1-df[pair[1]]/72]
#         ax[1].plot(x, y, linestyle='--', color='gray', alpha=0.5, linewidth=0.3, markersize=1)
#     ax[1].set_ylabel('error rates (%)', fontsize=8)

    
#     y, h = (1-df['Facc_UPF'].min()/72)+0.08, 0.03
#     ax[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c="k") 
#     ax[1].text((x1+x2)*.5, y+h, "easy\ntask", ha='center', va='center', color="k", size=8)
#     ax[1].plot([x3, x3, x4, x4], [y, y+h, y+h, y], lw=1, c="k") 
#     ax[1].text((x3+x4)*.5, y+h, "difficult\ntask", ha='center', va='center', color="k", size=8)

#     for idx in range(2):
#         ax[idx].set_xticks(range(1, len(boxplot_columns) + 1))
#         ax[idx].set_xticklabels(boxplot_columns)
#         ax[idx].spines['top'].set_visible(False)
#         ax[idx].spines['right'].set_visible(False)
#         ax[idx].tick_params(axis='x', labelsize=8)
#         ax[idx].tick_params(axis='y', labelsize=6)

#     # Show the chart
#     plt.tight_layout()
#     # plt.show()
#     fig.savefig('./individual_level/paper_figures/fig2/a_behavior.svg', bbox_inches='tight', transparent=True)
#     plt.close()



def a_plot_behavior():

    file = './results/individual_behaviors/reaction_time.xlsx'
    fig, ax = plt.subplots(1, 2, figsize=(140*mm, 40*mm))
    x = np.linspace(0, 1, num=2)
    df = pd.read_excel(file, sheet_name='Sheet1')
    colors = ['#334c81', '#a7413c']
    for task_idx, task in enumerate(['Fsp', 'Facc']):
        mean = []
        for idx, name in enumerate(['{}_PF'.format(task), '{}_UPF'.format(task)]):
            scores = df[name]
            mean.append(np.mean(scores))
            lower, upper = sms.DescrStatsW(scores).tconfint_mean()
            ax[0].plot([idx, idx], [lower, upper], linewidth=1.0, color='black')
        ax[0].plot(x, np.array(mean), marker='o', markersize=4, linestyle='solid', color=colors[task_idx], label=task, linewidth=2.0)
    ax[0].set_ylim(550, 980)
    ax[0].set_ylabel('reaction time (ms)', fontsize=8)
    ax[0].yaxis.set_major_locator(MultipleLocator(100))


    file = './results/individual_behaviors/error_rates.xlsx'
    x = np.linspace(0, 1, num=2)
    df = pd.read_excel(file, sheet_name='hit rates')
    for task_idx, task in enumerate(['Fsp', 'Facc']):
        mean = []
        for idx, name in enumerate(['{}_PF'.format(task), '{}_UPF'.format(task)]):
            scores = df[name]
            mean.append(np.mean(scores))
            lower, upper = sms.DescrStatsW(scores).tconfint_mean()
            ax[1].plot([idx, idx], [lower, upper], linewidth=1.0, color='black')
        ax[1].plot(x, np.array(mean), marker='o', markersize=4, linestyle='solid', label=task, color=colors[task_idx], linewidth=2.0)
    ax[1].set_ylim(0.56, 0.74)
    ax[1].set_ylabel('hit rates (%)', fontsize=8)
    ax[1].yaxis.set_major_locator(MultipleLocator(0.05))

    for idx in range(2):
        ax[idx].set_xlim(-0.5, 1.5)
        ax[idx].spines['top'].set_visible(False)
        ax[idx].spines['right'].set_visible(False)
        
        xrange = ['primed', 'unprimed']
        ax[idx].set_xticks(x)
        ax[idx].set_xticklabels(xrange, size=8)
        ax[idx].tick_params(axis='x', labelsize=6)
        ax[idx].tick_params(axis='y', labelsize=6)
    
    ax[0].legend(fontsize=8)
    plt.show()
    fig.savefig('./individual_level/paper_figures/fig2/behavior.svg', bbox_inches='tight', transparent=True)
    plt.close()
a_plot_behavior()


