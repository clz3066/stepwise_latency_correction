import pandas as pd
import numpy as np
import seaborn as sns 
import joypy
from matplotlib import cm
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as scio
import random
from scipy.signal import find_peaks

plt.rcParams['savefig.dpi']         = 300
plt.rcParams['font.family']         = 'sans-serif'
plt.rcParams['font.sans-serif']     = 'Arial'
plt.rcParams['axes.unicode_minus']  = False  
plt.rcParams['savefig.transparent'] = True
plt.rcParams['svg.fonttype']        = 'none'
plt.rcParams['xtick.direction']     = 'in'
mm                                  = 1/25.4 # inch 和 毫米的转换

data_path                           = 'data/Facc_Fsp/'
for_illustration_path               = './latency/plot_codes/for_illustration/'
mat_name_lst                        = {'step0':'original', 
                                       'step1': 'st_c', 
                                       'step2': 'st_c_synced',
                                       'step3': 'st_c_synced_within_subject', 
                                       'step4': 'aligned_amp',
                                       's_component': 'st_s',
                                       'r_component': 'st_r'  }  


def generate_random_numbers(start, end, count):
    numbers = []
    for _ in range(count):
        number = random.uniform(start, end)
        numbers.append(number)
    return numbers



def figure_a():
    ''' 
    Fig 1a: variabilities 
    '''
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(12, 2)
    ax11 = plt.subplot(gs[4, 0])
    ax12 = plt.subplot(gs[5, 0])
    ax13 = plt.subplot(gs[6, 0])
    ax14 = plt.subplot(gs[7, 0])
    ax15 = plt.subplot(gs[8, 0])
    ax1 = [ax11, ax12, ax13, ax14, ax15]
    ax2 = plt.subplot(gs[0:4, 1])
    ax3 = plt.subplot(gs[4:8, 1])    
    ax4 = plt.subplot(gs[8:12, 1])
    # plt.subplots_adjust(hspace=-0.5)

    # plot single trials
    colors= np.flip(['#bdc3c7', '#7f8c8d', '#2c3e50', '#4B4A54', '#000000'])
    mat = scio.loadmat('{}/251104_Fsp_PF_cc.mat'.format(for_illustration_path))['results'][0, 0]
    single_trials = mat['original']

    for idx in range(5):
        ax1[idx].plot(single_trials[27, :, idx+5], color='#000000')
        ax1[idx].axis('off')
    ax1[0].axis('off')

    components = scio.loadmat('./ERP_results/ERP/Fsp_step0.mat')['p'].T
    multi_idx = np.random.randint(components.shape[1], size=5)
    print(multi_idx)
    for idx, random_electrode in enumerate(multi_idx):
        ax2.axis('off')
        ax2.plot(components[30:110, random_electrode], color=colors[idx])

    x = np.linspace(-5, 5, 100)
    means1 = generate_random_numbers(-2, 2, 5)
    print(means1)
    stds = [1]*len(means1)
    for i, (mean, std) in enumerate(zip(means1, stds)):
        y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        ax3.plot(x, y, color=colors[i])
    ax3.axis('off')

    ####################### condition 2 single trials ###############################
    amps = generate_random_numbers(0, 1, 5)
    stds = [1]*len(amps)
    for i, (amp, std) in enumerate(zip(amps, stds)):
        y = amp / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 0) / std) ** 2)
        ax4.plot(x, y, color=colors[i])
    ax4.axis('off')

    fig.savefig('./latency/paper_figures/fig1/single_trials.svg', bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()



def figure_b():
    
    ########Fig 1b case 1: smearing effects, latency jitters  ##########
    fig, ax = plt.subplots(3, 3, figsize=(120*mm, 50*mm))
    x = np.linspace(-5, 5, 100)

    ####################### two conditions single trials ############################
    y = 1 / (1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 0) / 1) ** 2)
    ax[0, 0].plot(x, y, color='k', linewidth=0.5)
    ax[0, 2].plot(x, y, color='k', label='genuine amplitude', linewidth=0.5)
    ax[0, 0].axis('off')
    
    means1 = generate_random_numbers(-2, 2, 20)
    print(means1)
    stds = [1]*len(means1)
    y1_mean = np.zeros_like(x)
    for mean, std in zip(means1, stds):
        y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        ax[0, 1].plot(x, y, color='k', alpha=0.1, linewidth=0.5)
        y1_mean += y
    ax[0, 1].plot(x, (y1_mean/len(means1)), '--', color='k', linewidth=0.5)
    ax[0, 1].axis('off')

    ####################### two conditions ERP #######################################
    ax[0, 2].plot(x, y1_mean/len(means1), '--', color='k', label='latency-jittered amplitude', linewidth=0.5)
    ax[0, 2].axis('off')
    ax[0, 2].legend(fontsize=8)

    ########  Fig 1b, case2: smearing effects, latency shifts  ############
    y0 = 1 / (1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 0) / 1) ** 2)
    ax[1, 0].plot(x, y0, color='k', linewidth=0.5)
    y1 = 0.9 / (1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 0) / 1) ** 2)
    ax[1, 0].plot(x, y1, '--', color='k', linewidth=0.5)
    ax[1, 0].axis('off')

    ax[2, 0].plot(x, y0-y1, '-.', color='k', linewidth=0.5)
    ax[2, 1].plot(x, y0-y1, '-.', color='k', label='genuine amplitude difference', linewidth=0.5)
    ax[2, 0].axis('off')
    
    y0 = 1 / (1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 0.3) / 1) ** 2)
    ax[1, 1].plot(x, y0, color='k', linewidth=0.5)
    ax[2, 1].plot(x, 0.3*y0, color='k', label='condition A amplitude', linewidth=0.5)
    y1 = 0.9 / (1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x + 0.3) / 1) ** 2)
    ax[1, 1].plot(x, y1, '--', color='k', linewidth=0.5)
    ax[2, 1].plot(x, 0.3*y1, '--', color='k', label='condition B amplitude', linewidth=0.5)
    ax[1, 1].axis('off')

    ax[2, 0].plot(x, y0-y1, ':', color='k', linewidth=0.5)
    ax[2, 1].plot(x, y0-y1, ':', color='k', label='with latency shift-induced amplitude difference', linewidth=0.5)
    ax[2, 1].axis('off')
    ax[2, 1].legend(fontsize=8)
    # plt.show()
    plt.savefig('./latency/paper_figures/fig1/b.svg', bbox_inches='tight', transparent=True)
    plt.close()



def figure_c1(step_idx):
    '''
    Fig 1c: part 1 - different steps
    0-3: ylim: 'max'
    4: ylim: (0, 20)
    '''
    data = pd.read_excel('{}/simulation.xlsx'.format(for_illustration_path), sheet_name='step{}-results'.format(step_idx))
    if step_idx != 4:
        joypy.joyplot(data, by='Name', ylabelsize=8, figsize=(5, 4), fade=False, ylim='max', fill=True, linecolor='black', overlap=2,                         
                    legend=False, color=['#ee486699', '#ee486699', '#ee486699', '#08306b99', '#08306b99', '#08306b99']) 
    else:
        joypy.joyplot(data, by='Name', ylabelsize=8, figsize=(5, 4), fade=False, ylim=(0, 20), fill=True, linecolor='black', overlap=2,                         
                    legend=False, color=['#ee486699', '#ee486699', '#ee486699', '#08306b99', '#08306b99', '#08306b99']) 
    plt.savefig('./latency/paper_figures/fig1/c{}.svg'.format(step_idx), bbox_inches='tight', transparent=True, dpi=300)
    # plt.show()
    plt.close()



def figure_c2():
    '''
    Fig 1c: part 2 - RIDE decomposing
    '''
    fig, ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(60*mm, 20*mm))
    x = np.arange(0, 200)
    mat = scio.loadmat('{}/251104_Fsp_PF_cc.mat'.format(for_illustration_path))['results'][0, 0]
    s, c, r = mat['st_s'], mat['st_c'], mat['st_r']
    s_part = np.mean(s[0:200, 33, :], axis=1)
    c_part = np.mean(c[50:250, 27, :], axis=1)
    r_part = np.mean(r[20:220, 27, :], axis=1)
    erp = s_part + c_part + r_part
    ax[0].plot(x, erp, color='black', label='single-trial ERP')
    ax[0].legend(labelcolor='black', fontsize=6)
    ax[0].axis('off')
    ax[1].plot(x, s_part, color='#4B4453', alpha=0.5, label='S component')
    ax[1].plot(x, c_part, color='#E87979', label='C component')
    ax[1].plot(x, r_part, color='#813700', label='R component')
    ax[1].axis('off')
    ax[1].legend(labelcolor=['#4B445399', '#E87979', '#813700'], fontsize=6)
    # plt.subplots_adjust(wspace=0.5, hspace=0)    
    plt.savefig('./latency/paper_figures/fig1/c_ride.svg', bbox_inches='tight', transparent=True)
    plt.show()
    # plt.close()


def readin_subject_name(path=data_path):
    files = os.listdir(path)
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


def new_figure_c():
    '''
    doesn't work
    '''

    subject_name = readin_subject_name()

    for subject in subject_name:
        fig, axs = plt.subplots(6, 5, figsize=(200*mm, 70*mm), sharey=True)
        plt.subplots_adjust(wspace=0.4, hspace=-0.2)

        primed_mat = scio.loadmat('{}/{}_Fsp_PF_cc.mat'.format(data_path, subject))['results'][0, 0]
        unprimed_mat = scio.loadmat('{}/{}_Fsp_UPF_cc.mat'.format(data_path, subject))['results'][0, 0]

        for step_idx in range(5):
            name = mat_name_lst['step{}'.format(step_idx)]
            primed_data = primed_mat[name]
            unprimed_data = unprimed_mat[name]


            for trial_idx in range(3):
                averaged_primed = np.mean(primed_data[49:294, 27, trial_idx*10: trial_idx*10+20], axis=1)
                averaged_unprimed = np.mean(unprimed_data[49:294, 27, trial_idx*10: trial_idx*10+20], axis=1)
                
                if step_idx == 3:
                    median_within_subject = primed_mat['median_within_subject']
                    axs[trial_idx, step_idx].axvline(median_within_subject-49, linestyle='dotted', color='grey', linewidth=1.0)
                    axs[trial_idx+3, step_idx].axvline(median_within_subject-49, linestyle='dotted', color='grey', linewidth=1.0)
                if step_idx == 4:
                    axs[trial_idx, step_idx].plot(averaged_primed*4, color='#334c81', linewidth=1.0)
                    axs[trial_idx+3, step_idx].plot(averaged_unprimed*4, color='#a7413c', linewidth=1.0)
                else:
                    axs[trial_idx, step_idx].plot(averaged_primed, color='#334c81', linewidth=1.0)
                    axs[trial_idx+3, step_idx].plot(averaged_unprimed, color='#a7413c', linewidth=1.0)
                axs[trial_idx, step_idx].axis('off')
                axs[trial_idx+3, step_idx].axis('off')
        # plt.show()
        plt.savefig('./latency/paper_figures/fig1/illustration/{}.svg'.format(subject), bbox_inches='tight', transparent=True)
        plt.close()



def get_subject_RIDE_figures():

    names = readin_subject_name()
    for subject_name in names:
        fig, axs = plt.subplots(2, 2, figsize=(200*mm, 80*mm), sharey=True)
        primed_mat = scio.loadmat('{}/{}_Fsp_PF_cc.mat'.format(data_path, subject_name))['results'][0, 0]
        unprimed_mat = scio.loadmat('{}/{}_Fsp_UPF_cc.mat'.format(data_path, subject_name))['results'][0, 0]

        averaged_primed = np.mean(primed_mat['original'][49:294, 27], axis=1)
        averaged_unprimed = np.mean(unprimed_mat['original'][49:294, 27], axis=1)
        axs[0, 0].plot(averaged_primed, color='k', linewidth=1.0)
        axs[1, 0].plot(averaged_unprimed, color='k', alpha=0.5, linestyle='dashed', linewidth=1.0)
    
        averaged_primed = np.mean(primed_mat['st_s'][49:294, 27], axis=1)
        averaged_unprimed = np.mean(unprimed_mat['st_s'][49:294, 27], axis=1)
        axs[0, 1].plot(averaged_primed, linewidth=1.0)
        axs[1, 1].plot(averaged_unprimed, alpha=0.5, linestyle='dashed', linewidth=1.0)

        averaged_primed = np.mean(primed_mat['st_c'][49:294, 27], axis=1)
        averaged_unprimed = np.mean(unprimed_mat['st_c'][49:294, 27], axis=1)
        axs[0, 1].plot(averaged_primed, linewidth=1.0)
        axs[1, 1].plot(averaged_unprimed, alpha=0.5, linestyle='dashed', linewidth=1.0)

        averaged_primed = np.mean(primed_mat['st_r'][49:294, 27], axis=1)
        averaged_unprimed = np.mean(unprimed_mat['st_r'][49:294, 27], axis=1)
        axs[0, 1].plot(averaged_primed, linewidth=1.0)
        axs[1, 1].plot(averaged_unprimed, alpha=0.5, linestyle='dashed', linewidth=1.0)

        # plt.show()
        plt.savefig('./latency/paper_figures/fig1/illustration/{}.png'.format(subject_name), bbox_inches='tight', transparent=True)
        plt.close()



def get_chosen_subject_RIDE_figures():

    files = os.listdir(for_illustration_path)
    for file in files:
        if file.endswith('.mat'):
            name = file.split('.')[0].split('_')[0]

            fig, axs = plt.subplots(1, 2, figsize=(130*mm, 20*mm), sharey=True)
            mat = scio.loadmat('{}/{}'.format(for_illustration_path, file))['results'][0, 0]

            averaged = np.mean(mat['original'][:, 27], axis=1)
            axs[0].plot(averaged, color='k', linewidth=1.3)
        
            averaged = np.mean(mat['st_s'][:, 27], axis=1)
            axs[1].plot(averaged, linewidth=1.3, color='#3F4C6F')

            averaged = np.mean(mat['st_c'][:, 27], axis=1)
            axs[1].plot(averaged, linewidth=1.3, color='#B2551F')

            averaged = np.mean(mat['st_r'][:, 27], axis=1)
            axs[1].plot(averaged, linewidth=1.3, color='#EDB021')

            axs[0].axis('off')
            axs[1].axis('off')
            # plt.show()
            plt.savefig('{}/{}.svg'.format(for_illustration_path, name), bbox_inches='tight', transparent=True)
            plt.close()



def a_RIDE():

    files = ['251003_Fsp_UPF_cc.mat', '252042_Fsp_UPF_cc.mat', '251098_Fsp_PF_cc.mat']
    alphas = [1.0, 1.0, 1.0]
    linewidth = 1.0
    fig, axs = plt.subplots(3, 2, figsize=(80*mm, 50*mm), sharey=True)
    for idx, (file, alpha) in enumerate(zip(files, alphas)):

        mat = scio.loadmat('{}/{}'.format(for_illustration_path, file))['results'][0, 0]

        averaged = np.mean(mat['original'][:, 27], axis=1)
        axs[idx, 0].plot(averaged, linewidth=linewidth, color='#B2551F', alpha=alpha)
    
        averaged = np.mean(mat['st_s'][:, 27], axis=1)
        axs[idx, 1].plot(averaged, linewidth=linewidth, color='#6C7BBB', alpha=alpha)

        averaged = np.mean(mat['st_c'][:, 27], axis=1)
        axs[idx, 1].plot(averaged, linewidth=linewidth, color='#000000', alpha=alpha)

        averaged = np.mean(mat['st_r'][:, 27], axis=1)
        axs[idx, 1].plot(averaged, linewidth=linewidth, color='#FFA500', alpha=alpha)

        axs[idx, 0].axis('off')
        axs[idx, 1].axis('off')
    # plt.show()
    plt.savefig('./latency/paper_figures/fig1/a_RIDE.svg', bbox_inches='tight', transparent=True)
    plt.close()


def a_variability():
    ''' 
    Fig 1a: variabilities 
    '''
    fig, axs = plt.subplots(3, 1, figsize=(35*mm, 50*mm))
    alphas = [1.0, 0.75, 0.5]
    linewidth = 1.0
    x = np.linspace(-5, 5, 100)

    stds = [0.4, 1, 1.5]
    for i, (std, alpha) in enumerate(zip(stds, alphas)):
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x - 0)**2 / (2 * std**2))
        # y = y / np.max(y)
        axs[0].plot(x, y, color='#000000', linewidth=linewidth, alpha=alpha)
    axs[0].axis('off')

    means1 = [-1.2, 0, 1.2]
    print(means1)
    stds = [1]*len(means1)
    for i, (mean, std, alpha) in enumerate(zip(means1, stds, alphas)):
        y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        axs[1].plot(x, y, color='#000000', linewidth=linewidth, alpha=alpha)
    axs[1].axis('off')

    amps = [0.3, 0.5, 0.8]
    stds = [1]*len(amps)
    for i, (amp, std, alpha) in enumerate(zip(amps, stds, alphas)):
        y = amp / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 0) / std) ** 2)
        axs[2].plot(x, y, color='#000000', linewidth=linewidth, alpha=alpha)
    axs[2].axis('off')

    # plt.show()
    fig.savefig('./latency/paper_figures/fig1/a_variability.svg', bbox_inches='tight', transparent=True)
    plt.close()


def b_case1():
    fig, ax = plt.subplots(2, 3, figsize=(120*mm, 40*mm), sharey=True)
    x = np.linspace(-5, 5, 100)
    linewidth = 1.0

    y1 = 1 / (1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 0) / 1) ** 2)
    y2 = 0.8 / (1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 0) / 1) ** 2)
    ax[0, 0].plot(x, y1, color='k', linewidth=linewidth)
    ax[0, 0].plot(x, y2, color='k', linestyle='dashed', linewidth=linewidth)
    ax[0, 0].axis('off')
    
    means1 = generate_random_numbers(-2, 2, 5)
    print(means1)
    stds = [1]*len(means1)
    y1_mean = np.zeros_like(x)
    for mean, std in zip(means1, stds):
        y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        ax[0, 1].plot(x, y, color='k', alpha=0.3, linewidth=linewidth)
        y1_mean += y
    ax[0, 1].plot(x, (y1_mean/len(means1)), color='k', linewidth=linewidth)
    ax[0, 1].axis('off')

    means2 = generate_random_numbers(-1, 1, 5)
    print(means2)
    stds = [1]*len(means2)
    y2_mean = np.zeros_like(x)
    for mean, std in zip(means2, stds):
        y = 0.8 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        ax[1, 1].plot(x, y, color='k', alpha=0.3, linewidth=linewidth, linestyle='dashed')
        y2_mean += y
    ax[1, 1].plot(x, (y2_mean/len(means1)), linestyle='dashed', color='k', linewidth=linewidth)
    ax[1, 1].axis('off')

    ax[0, 2].plot(x, y1_mean/len(means1), color='k', label='condition A', linewidth=linewidth)
    ax[0, 2].plot(x, y2_mean/len(means2), color='k', label='condition B', linewidth=linewidth, linestyle='dashed')
    ax[0, 2].legend(fontsize=8, loc='lower right')
    ax[0, 2].axis('off')
    ax[1, 0].axis('off')
    ax[1, 2].axis('off')
    
    plt.show()
    fig.savefig('./latency/paper_figures/fig1/b_case1.svg', bbox_inches='tight', transparent=True)
    plt.close()



def b_case2():
    fig, ax = plt.subplots(2, 2, figsize=(80*mm, 45*mm), sharey=True)
    x = np.linspace(-5, 5, 100)
    linewidth = 1.0

    y0 = 1 / (1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 0) / 1) ** 2)
    ax[0, 0].plot(x, y0, color='k', linewidth=linewidth)
    y1 = 0.8 / (1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 0) / 1) ** 2)
    ax[0, 0].plot(x, y1, color='k', linewidth=linewidth, linestyle='dashed')
    ax[0, 0].axis('off')

    ax[1, 0].plot(x, y0-y1, ':', color='k', linewidth=linewidth)
    ax[1, 0].axis('off')
    
    y0 = 1 / (1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 0.3) / 1) ** 2)
    ax[0, 1].plot(x, y0, color='k', linewidth=linewidth)
    y1 = 0.8 / (1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x + 0.3) / 1) ** 2)
    ax[0, 1].plot(x, y1, color='k', linewidth=linewidth, linestyle='dashed')
    ax[0, 1].axis('off')

    ax[1, 1].plot(x, 0.3*y0, color='k', label='condition A', linewidth=linewidth)
    ax[1, 1].plot(x, 0.3*y1, '--', color='k', label='condition B', linewidth=linewidth)
    ax[1, 1].plot(x, y0-y1, ':', color='k', label='amplitude difference', linewidth=linewidth)
    ax[1, 1].axis('off')
    ax[1, 1].legend(fontsize=8)
    
    plt.show()
    fig.savefig('./latency/paper_figures/fig1/b_case2.svg', bbox_inches='tight', transparent=True)
    plt.close()



def c_steps():

    files = ['251045_Fsp_PF_cc.mat', '252028_Fsp_PF_cc.mat', '251098_Fsp_PF_cc.mat',
             '251010_Fsp_UPF_cc.mat', '251037_Fsp_UPF_cc.mat', '251003_Fsp_UPF_cc.mat']
    alpha = 1.0
    linewidth = 1.0
    x = np.linspace(1, 350, 350)
    step2_shifts = [-11, -4, 15, 2, -17, 15]
    step3_shifts = [7, 14, 33, -16, -35, -3]
    linestyles = ['solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed']
    fig, axs = plt.subplots(6, 6, figsize=(180*mm, 100*mm), sharey=True)
    for idx, (file, step2_shift, step3_shift, linestyle) in enumerate(zip(files, step2_shifts, step3_shifts, linestyles)):

        mat = scio.loadmat('{}/{}'.format(for_illustration_path, file))['results'][0, 0]

        averaged = np.mean(mat['original'][:, 27], axis=1)
        axs[idx, 0].plot(averaged[:350], linewidth=linewidth, color='#B2551F', alpha=alpha, linestyle=linestyle)
    
        averaged = np.mean(mat['st_s'][:, 27], axis=1)
        axs[idx, 1].plot(averaged[:350], linewidth=linewidth, color='#6C7BBB', alpha=alpha, linestyle=linestyle)
        
        averaged = np.mean(mat['st_c'][:, 27], axis=1)
        axs[idx, 1].plot(averaged[:350], linewidth=linewidth, color='#000000', alpha=alpha, linestyle=linestyle)
        axs[idx, 2].plot(averaged[:350], linewidth=linewidth, color='#000000', alpha=alpha, linestyle=linestyle)
        peaks, _ = find_peaks(averaged[:350])        
        peak_latency = x[peaks][np.argmax(averaged[peaks])]
        print(peak_latency)
        axs[idx, 2].axvline(peak_latency, color='g', linestyle='dotted', linewidth=linewidth)

        shifted_y = np.roll(averaged, step2_shift)
        zero_padded_y = np.concatenate((np.zeros(np.abs(step2_shift)), shifted_y[np.abs(step2_shift):]))
        axs[idx, 3].plot(zero_padded_y[:350], linewidth=linewidth, color='#000000', alpha=alpha, linestyle=linestyle)
        axs[idx, 3].axvline(143, color='g', linestyle='dotted', linewidth=linewidth)
        axs[idx, 3].axvline(179, color='g', linestyle='dotted', linewidth=linewidth)

        shifted_y = np.roll(averaged, step3_shift)
        zero_padded_y = np.concatenate((np.zeros(np.abs(step3_shift)), shifted_y[np.abs(step3_shift):]))
        axs[idx, 4].plot(zero_padded_y[:350], linewidth=linewidth, color='#000000', alpha=alpha, linestyle=linestyle)
        axs[idx, 4].axvline(161, color='g', linestyle='dotted', linewidth=linewidth)
        axs[idx, 5].plot(zero_padded_y[:350]/2.5, linewidth=linewidth, color='#000000', alpha=alpha, linestyle=linestyle)

        averaged = np.mean(mat['st_r'][:, 27], axis=1)
        axs[idx, 1].plot(averaged[:350], linewidth=linewidth, color='#FFA500', alpha=alpha, linestyle=linestyle)

        # 143 PF, 179 UPF, 161 all
        axs[idx, 0].axis('off')
        axs[idx, 1].axis('off')
        axs[idx, 2].axis('off')
        axs[idx, 3].axis('off')
        axs[idx, 4].axis('off')
        axs[idx, 5].axis('off')

    plt.show()
    fig.savefig('./latency/paper_figures/fig1/c_steps.svg', bbox_inches='tight', transparent=True)
    plt.close()



def c_steps_separated():

    files = ['251045_Fsp_PF_cc.mat', '252028_Fsp_PF_cc.mat', '251098_Fsp_PF_cc.mat',
             '251010_Fsp_UPF_cc.mat', '251037_Fsp_UPF_cc.mat', '251003_Fsp_UPF_cc.mat']
    linewidth = 1.0
    x = np.linspace(1, 350, 350)

    fig, axs = plt.subplots(6, 6, figsize=(180*mm, 100*mm), sharey=True)
    for idx, file in enumerate(files):

        mat = scio.loadmat('{}/{}'.format(for_illustration_path, file))['results'][0, 0]

        averaged = np.mean(mat['original'][:, 27], axis=1)
        axs[idx, 0].plot(averaged[:350], linewidth=linewidth, color='#B2551F')
    
        averaged = np.mean(mat['st_s'][:, 27], axis=1)
        axs[idx, 1].plot(averaged[:350], linewidth=linewidth, color='#6C7BBB')
        
        averaged = np.mean(mat['st_c'][:, 27], axis=1)
        axs[idx, 1].plot(averaged[:350], linewidth=linewidth, color='#000000')

        averaged = np.mean(mat['st_r'][:, 27], axis=1)
        axs[idx, 1].plot(averaged[:350], linewidth=linewidth, color='#FFA500')

        axs[idx, 0].axis('off'); axs[idx, 1].axis('off'); axs[idx, 2].axis('off');
        axs[idx, 3].axis('off'); axs[idx, 4].axis('off'); axs[idx, 5].axis('off');

    original_shifts = [50, 30, -20, -10, 20, 38]
    step2_shifts = [-41, -14, 55, 28, -21, -7]
    step3_shifts = [-25, 2, 71, 12, -21, -23]
    for idx, (file, original_shift, step2_shift, step3_shift) in enumerate(zip(files, original_shifts, step2_shifts, step3_shifts)):

        mat = scio.loadmat('{}/{}'.format(for_illustration_path, file))['results'][0, 0]
        averaged = np.mean(mat['st_c'][:, 27], axis=1)
        shifted_averaged = np.roll(averaged, original_shift)
        padded_averaged = np.concatenate((np.zeros(np.abs(original_shift)), shifted_averaged[np.abs(original_shift):]))
        
        peaks, _ = find_peaks(padded_averaged[:350])        
        peak_latency = x[peaks][np.argmax(padded_averaged[peaks])]
        print(peak_latency)
        axs[idx, 2].plot(padded_averaged[:350], linewidth=linewidth, color='#000000')

        shifted_y = np.roll(padded_averaged, step2_shift)
        zero_padded_y = np.concatenate((np.zeros(np.abs(step2_shift)), shifted_y[np.abs(step2_shift):]))
        axs[idx, 3].plot(zero_padded_y[:350], linewidth=linewidth, color='#000000')
        axs[idx, 2].axvline(163, color='g', linestyle='dotted', linewidth=1.5)
        axs[idx, 3].axvline(163, color='g', linestyle='dotted', linewidth=1.5)
        axs[idx, 2].axvline(195, color='g', linestyle='dotted', linewidth=1.5)
        axs[idx, 3].axvline(195, color='g', linestyle='dotted', linewidth=1.5)

        shifted_y = np.roll(padded_averaged, step3_shift)
        zero_padded_y = np.concatenate((np.zeros(np.abs(step3_shift)), shifted_y[np.abs(step3_shift):]))
        axs[idx, 4].plot(zero_padded_y[:350], linewidth=linewidth, color='#000000')
        axs[idx, 4].axvline(179, color='g', linestyle='dotted', linewidth=1.5)
        axs[idx, 5].plot(zero_padded_y[:350]/2.5, linewidth=linewidth, color='#000000')


    plt.show()
    fig.savefig('./latency/paper_figures/fig1/c_steps.svg', bbox_inches='tight', transparent=True)
    plt.close()

c_steps_separated()