import scipy.io as scio
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.pyplot import MultipleLocator
from scipy.stats import ttest_1samp
import matplotlib.ticker as mticker


data_path = '../../Dataset/Facc_Fsp/'
files = os.listdir(data_path)
mat_name_lst = {'step0':'original', 
                'step1':'st_c', 
                'step2':'st_c_synced', 
                'step3':'st_c_synced_within_subject', 
                'step4':'aligned_amp'}   
plt.rcParams['savefig.dpi'] = 300
x = np.linspace(100, 1080, 245)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

cmap                            = 'jet'
interpolation                   = 'quadric'
x_major_locator                 = MultipleLocator(0.5)
y_major_locator                 = MultipleLocator(30)
colorbar_locator                = mticker.MultipleLocator(20)

img_path = './latency/figures/fig3/trials/'
imgs = os.listdir(img_path)


def plot_imagsec_one():
    sample_file_path = './252014_Fsp_PF_cc.mat'

    results = scio.loadmat(sample_file_path)['results']
    fig, ax = plt.subplots(1, 5, figsize=(12, 2), sharex=False, sharey=False)

    data_lst = []
    for i in range(5):
        data = results[0, 0][mat_name_lst['step{}'.format(i)]][:, 27, :]
        data_lst.append(data)
        ax[i].set_xlim(-100, 1500)
        ax[i].yaxis.set_major_locator(y_major_locator)
        ax[i].tick_params(labelsize=8)  

    max, min = np.max(data_lst), np.min(data_lst)
    v_min, v_max = -np.max([max, np.abs(min)]), np.max([max, np.abs(min)])
    print(v_min, v_max)

    
    rt = np.squeeze(results[0, 0]['rt'])
    trial_num = len(rt)
    x = np.arange(-100, 1500, 4)
    y = np.linspace(0, trial_num, trial_num, endpoint=False)

    # sorted by reaction time
    sorted_id = sorted(range(len(rt)), key=lambda k: rt[k], reverse=True)
    data_Pz = np.squeeze(data_lst[0])
    data_Pz = data_Pz[:, sorted_id]
    ax[0].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)
    ax[0].set_ylabel('trial (sorted by reaction time)')
    ax[0].plot(rt[sorted_id], y, c='grey')


    # sorted by latency c
    latency = np.squeeze(results[0, 0]['latency_c'])
    sorted_id = sorted(range(len(latency)), key=lambda k: latency[k], reverse=True)
    for idx in range(1, 3):
        data_Pz  = np.squeeze(data_lst[idx])
        data_Pz = data_Pz[:, sorted_id]
        ax[idx].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)
        ax[idx].set_ylabel('trial (sorted by C latency)')
    peak = np.squeeze(results[0, 0]['ref_c_peak'])
    lags = peak*4 + latency - 100
    ax[1].plot(lags[sorted_id], y, c='white')
    peak_lst = np.ones_like(lags[sorted_id]) * peak * 4 - 100
    ax[2].plot(peak_lst, y, c='white')


    # sorted by amplitude c
    amplitudes = results[0, 0]['amp_c'][:, 27]
    sorted_id = sorted(range(len(amplitudes)), key=lambda k: amplitudes[k], reverse=True)
    for idx in range(3, 5):
        data_Pz  = np.squeeze(data_lst[idx])
        data_Pz = data_Pz[:, sorted_id]
        im = ax[idx].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)
        ax[idx].set_ylabel('trial (sorted by C amplitudes)')

    cbar = fig.colorbar(im, ax=ax[4], fraction=0.05)
    
    plt.tight_layout()
    plt.show()



def plot_imagsec():
    for file in files:
        if file.endswith('.mat'):
            content = file.split('.')[0].split('_')
            print(content)
            if content[1] == 'Fsp':
                results = scio.loadmat(os.path.join(DATA_PATH, file))['results']
                fig, ax = plt.subplots(1, 5, figsize=(12, 2), sharex=True, sharey=False)

                data_lst = []
                for i in range(5):
                    data = results[0, 0][mat_name_lst['step{}'.format(i)]][49:294, 27, :]
                    data_lst.append(data)
                max, min = np.max(data_lst), np.min(data_lst)
                v_min, v_max = -np.max([max, np.abs(min)]), np.max([max, np.abs(min)])
                print(v_min, v_max)

                latency = results[0, 0]['latency_c']
                sorted_id = sorted(range(len(latency)), key=lambda k: latency[k], reverse=True)

                for idx in range(3):
                    data_Pz  = np.squeeze(data_lst[idx])
                    data_Pz = data_Pz[:, sorted_id]
                    ax[idx].pcolor(data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)

                amplitudes = results[0, 0]['amp_c'][:, 27]
                sorted_id = sorted(range(len(amplitudes)), key=lambda k: amplitudes[k], reverse=True)

                for idx in range(3, 5):
                    data_Pz  = np.squeeze(data_lst[idx])
                    data_Pz = data_Pz[:, sorted_id]
                    im = ax[idx].pcolor(data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)

                # cbar = fig.colorbar(im, ax=ax[4], fraction=0.05)
                # cbar.set_label('Accuracy', rotation=270, labelpad=20, fontsize=5)

                plt.tight_layout()
                # plt.show()
                plt.savefig('./latency/figures/trials/{}.jpg'.format(file.split('.')[0]))
                # plt.close()



def plot_imagsec_smooth():

    for file in imgs:
        if file.endswith('.jpg'):
            content = file.split('.')[0]
            name = content + '.mat'
            results = scio.loadmat(os.path.join(DATA_PATH, name))['results']
            fig, ax = plt.subplots(1, 5, figsize=(12, 2), sharex=True, sharey=False)

            data_lst = []
            for i in range(5):
                data = results[0, 0][mat_name_lst['step{}'.format(i)]][:, 27, :]
                data_lst.append(data)
                ax[i].set_xlim(-100, 1500)
                ax[i].yaxis.set_major_locator(y_major_locator)
                ax[i].tick_params(labelsize=8)  

            max, min = np.max(data_lst), np.min(data_lst)
            v_min, v_max = -np.max([max, np.abs(min)]), np.max([max, np.abs(min)])
            print(v_min, v_max)

            
            rt = np.squeeze(results[0, 0]['rt'])
            trial_num = len(rt)
            x = np.arange(-100, 1500, 4)
            y = np.linspace(0, trial_num, trial_num, endpoint=False)

            # sorted by reaction time
            sorted_id = sorted(range(len(rt)), key=lambda k: rt[k], reverse=True)
            data_Pz = np.squeeze(data_lst[0])
            data_Pz = data_Pz[:, sorted_id]
            ax[0].pcolormesh(x, y, cv2.GaussianBlur(data_Pz.T, (3, 3), 0), cmap='jet', vmin=v_min, vmax=v_max)
            ax[0].set_ylabel('trial (sorted by reaction time)')
            ax[0].plot(rt[sorted_id], y, c='grey')


            # sorted by latency c
            latency = np.squeeze(results[0, 0]['latency_c'])
            sorted_id = sorted(range(len(latency)), key=lambda k: latency[k], reverse=True)
            for idx in range(1, 3):
                data_Pz  = np.squeeze(data_lst[idx])
                data_Pz = data_Pz[:, sorted_id]
                ax[idx].pcolormesh(x, y, cv2.GaussianBlur(data_Pz.T, (3, 3), 0), cmap='jet', vmin=v_min, vmax=v_max)
                ax[idx].set_ylabel('trial (sorted by C latency)')
            peak = np.squeeze(results[0, 0]['ref_c_peak'])
            lags = peak*4 + latency - 100
            ax[1].plot(lags[sorted_id], y, c='white')
            peak_lst = np.ones_like(lags[sorted_id]) * peak * 4 - 100
            ax[2].plot(peak_lst, y, c='white')


            # sorted by amplitude c
            amplitudes = results[0, 0]['amp_c'][:, 27]
            sorted_id = sorted(range(len(amplitudes)), key=lambda k: amplitudes[k], reverse=True)
            for idx in range(3, 5):
                data_Pz  = np.squeeze(data_lst[idx])
                data_Pz = data_Pz[:, sorted_id]
                im = ax[idx].pcolormesh(x, y, cv2.GaussianBlur(data_Pz.T, (3, 3), 0), cmap='jet', vmin=v_min, vmax=v_max)
                ax[idx].set_ylabel('trial (sorted by C amplitudes)')

            cbar = fig.colorbar(im, ax=ax[4], fraction=0.05)

            plt.tight_layout()
            plt.savefig('./fig_smoothing/{}.jpg'.format(content))
            plt.close()



def plot_imagsec_one_with_hist():
    sample_file_path = './252014_Fsp_PF_cc.mat'

    results = scio.loadmat(sample_file_path)['results']
    fig = plt.figure(figsize=(15, 4))
    gs = fig.add_gridspec(2, 5, height_ratios=(1, 2), 
                        bottom=0.2, top=0.8,
                        wspace=0.4, hspace=0.0)
    
    ax = []
    ax_histx = []
    for idx in range(5):
        ax_bottom = fig.add_subplot(gs[1, idx])
        ax_up = fig.add_subplot(gs[0, idx])
        ax_up.tick_params(axis="both", labelbottom=True, labelleft=False)
        ax_up.spines['right'].set_color('none')
        ax_up.spines['top'].set_color('none')
        ax_up.spines['left'].set_color('none')
        ax_up.spines['bottom'].set_position(('axes', 0.3))
        ax_up.spines['bottom'].set_alpha(0.5)
        ax_up.axes.yaxis.set_visible(False)
        ax_up.tick_params(labelsize=8) 
        ax.append(ax_bottom)
        ax_histx.append(ax_up)

    data_lst = []
    for i in range(5):
        data = results[0, 0][mat_name_lst['step{}'.format(i)]][:, 27, :]
        data_lst.append(data)
        ax[i].set_xlim(-0.1, 1.5)
        ax[i].yaxis.set_major_locator(y_major_locator)
        ax[i].tick_params(labelsize=8)  
        ax_histx[i].set_title('Step{}'.format(i), fontsize='15')

    max, min = np.max(data_lst), np.min(data_lst)
    v_min, v_max = -np.max([max, np.abs(min)]), np.max([max, np.abs(min)])
    print(v_min, v_max)

    ############################## subplot 0 ##########################################
    rt = np.squeeze(results[0, 0]['rt'])
    trial_num = len(rt)
    x = np.arange(-0.1, 1.5, 0.004)
    y = np.linspace(0, trial_num, trial_num, endpoint=False)
    xrange = [0]

    # sorted by reaction time
    sorted_id = sorted(range(len(rt)), key=lambda k: rt[k], reverse=True)
    data_Pz = np.squeeze(data_lst[0])
    data_Pz = data_Pz[:, sorted_id]
    ax[0].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)
    ax[0].set_ylabel('trial (sorted by reaction time)')
    ax[0].plot(rt[sorted_id], y, c='black')
    ax_histx[0].hist(rt, color='black', density=True, range=(-100, 1500), bins=trial_num)
    ax_histx[0].set_xlabel('reaction\ntime (ms)', loc='right', labelpad=-20)
    xreal  = [0]
    ax_histx[0].set_xticks(xreal)
    ax_histx[0].set_xticklabels(xrange)

    ############################## subplot 1,2 ##########################################
    # sorted by C latency
    latency = np.squeeze(results[0, 0]['latency_c'])
    sorted_id = sorted(range(len(latency)), key=lambda k: latency[k], reverse=True)
    for idx in range(1, 3):
        data_Pz  = np.squeeze(data_lst[idx])
        data_Pz = data_Pz[:, sorted_id]
        ax[idx].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)
        ax[idx].set_ylabel('trial (sorted by C latency)')

    # plot latency lines
    peak = np.squeeze(results[0, 0]['ref_c_peak'])
    lags = peak*4 + latency - 100
    ax[1].plot(lags[sorted_id], y, c='grey')
    peak_lst = np.ones_like(lags[sorted_id]) * peak * 4 - 100
    ax[2].plot(peak_lst, y, c='grey')

    # upper histograms
    ax_histx[1].hist(lags, color='grey', density=False, range=(-100, 1500), bins=trial_num)
    ax_histx[2].hist(peak_lst, color='grey', range=(-100, 1500), density=True)
    xreal  = [peak*4-100]
    ax_histx[1].set_xticks(xreal)
    ax_histx[1].set_xticklabels(xrange)
    ax_histx[1].set_xlabel('latency\nlags (ms)', loc='right', labelpad=-20)
    ax_histx[2].set_xticks(xreal)
    ax_histx[2].set_xticklabels(xrange)
    ax_histx[2].set_xlabel('latency\nlags (ms)', loc='right', labelpad=-20)

    ############################## subplot 3, 4 ##########################################
    # sorted by C amplitude
    amp_Pz = np.squeeze(results[0, 0]['c'][:, 27])
    cov_lst = []
    data_Pz  = np.squeeze(data_lst[3])
    for idx in range(data_Pz.shape[1]):
        cov = np.cov(data_Pz[:, idx], amp_Pz)
        cov_lst.append(cov[0, 1])
    sorted_id = sorted(range(len(cov_lst)), key=lambda k: cov_lst[k], reverse=True)
    data_Pz = data_Pz[:, sorted_id]
    ax[3].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)
    ax[3].set_ylabel('trial (sorted by C amplitudes)')
    xreal  = [0]
    xrange = [0]
    ax_histx[3].set_xticks(xreal)
    ax_histx[3].set_xticklabels(xrange)
    ax_histx[3].hist(cov_lst, range=(-40, 100), density=True, bins=trial_num, color='red')
    ax_histx[3].set_xlabel('amplitude\nvariablity', loc='right', labelpad=-20)
    ax_histx[3].set_ylim(0, 0.1)

    cov_lst = []
    data_Pz  = np.squeeze(data_lst[4])
    new_amp_Pz = np.mean(data_Pz, axis=1)
    for idx in range(data_Pz.shape[1]):
        cov = np.cov(data_Pz[:, idx], new_amp_Pz)
        cov_lst.append(cov[0, 1])
    data_Pz = data_Pz[:, sorted_id]
    im = ax[4].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)
    ax[4].set_ylabel('trial (sorted by C amplitudes)')
    ax_histx[4].hist(cov_lst, range=(-40, 100), bins=trial_num, density=True, color='red')
    ax_histx[4].set_xticks(xreal)
    ax_histx[4].set_xticklabels(xrange)
    ax_histx[4].set_xlabel('amplitude\nvariablity', loc='right', labelpad=-20)
    ax_histx[4].set_ylim(0, 0.1)
    # cbar = fig.colorbar(im, ax=ax[4], ticks=colorbar_locator, fraction=0.05)
    # cbar.set_label('Accuracy', rotation=270, labelpad=20, fontsize=15)
    plt.tight_layout()
    plt.show()




def plot_imagsec_with_hist():
    for file in imgs:
        if file.endswith('.jpg'):
            file_name = file.split('.')[0]
            contents = file_name.split('_')
            task = contents[2]
            if task == 'PF':
                anti_task = 'UPF'
                PF_name = file_name + '.mat'
                UPF_name = contents[0] + '_' + contents[1] + '_' + anti_task + '_cc.mat'
            elif task == 'UPF':
                anti_task = 'PF'
                UPF_name = file_name + '.mat'
                PF_name = contents[0] + '_' + contents[1] + '_' + anti_task + '_cc.mat'
            
            savefig_name = './latency/figures/fig3/trials&hist_svg/{}.svg'.format(contents[0])
            if not os.path.exists(savefig_name):
            
                fig = plt.figure(figsize=(22, 6))
                gs = fig.add_gridspec(4, 7, height_ratios=(1, 2, 1, 2), 
                                    bottom=0.2, top=0.8, left=0.05, 
                                    wspace=0.5, hspace=0.0)
                
            
                results = scio.loadmat(os.path.join(data_path, PF_name))['results']
                ax = []
                ax_histx = []
                for idx in range(7):
                    ax_up = fig.add_subplot(gs[0, idx])
                    ax_up.tick_params(axis="both", labelbottom=True, labelleft=False)
                    ax_up.spines['right'].set_color('none')
                    ax_up.spines['top'].set_color('none')
                    ax_up.spines['left'].set_color('none')
                    ax_up.spines['bottom'].set_position(('axes', 0.3))
                    ax_up.spines['bottom'].set_alpha(0.5)
                    ax_up.axes.yaxis.set_visible(False)
                    ax_up.tick_params(labelsize=11) 
                    ax_bottom = fig.add_subplot(gs[1, idx])
                    ax.append(ax_bottom)
                    ax_histx.append(ax_up)
                    

                data_lst = []
                for i in range(5):
                    data = results[0, 0][mat_name_lst['step{}'.format(i)]][49:294, 27, :]
                    data_lst.append(data)
                    ax[i].xaxis.set_major_locator(x_major_locator)
                    ax[i].tick_params(labelsize=13) 
                    ax[i].set_xticks([])
                    ax[i].set_yticks([])
                    if i == 0:
                        ax_histx[i].set_title('Basis', fontsize=18)
                    else:
                        ax_histx[i].set_title('Step{}'.format(i), fontsize=18)

                max, min = np.max(data_lst), np.min(data_lst)
                v_min, v_max = -np.max([max, np.abs(min)]), np.max([max, np.abs(min)])
                print(v_min, v_max)

                ############################## subplot 0 ##########################################
                rt = np.squeeze(results[0, 0]['rt'])
                trial_num = len(rt)
                x = np.linspace(0.1, 1.08, 245)
                y = np.linspace(0, trial_num, trial_num, endpoint=False)
                xrange = [0]

                # sorted by reaction time
                sorted_id = sorted(range(len(rt)), key=lambda k: rt[k], reverse=True)
                data_Pz = np.squeeze(data_lst[0])
                data_Pz = data_Pz[:, sorted_id]
                ax[0].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)
                ax[0].set_ylabel('trials (sorted by reaction time)', fontsize=18)
                ax[0].plot(rt[sorted_id]/1000, y, c='black')
                ax_histx[0].hist(rt, color='black', density=True, range=(100, 1080), bins=trial_num)
                ax_histx[0].set_xlabel('reaction\ntime (s)', loc='right', labelpad=-25, fontsize=13)
                xreal  = [0]
                ax_histx[0].set_xticks(xreal)
                ax_histx[0].set_xticklabels(xrange)

                ############################## subplot 1,2 ##########################################
                # sorted by C latency
                latency = np.squeeze(results[0, 0]['latency_c'])
                sorted_id = sorted(range(len(latency)), key=lambda k: latency[k], reverse=True)
                for idx in range(1, 3):
                    data_Pz  = np.squeeze(data_lst[idx])
                    data_Pz = data_Pz[:, sorted_id]
                    ax[idx].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)
                    ax[idx].set_ylabel('trials (sorted by C latency)', fontsize=18)

                # plot latency lines
                peak = np.squeeze(results[0, 0]['ref_c_peak'])
                lags = (peak*4 + latency - 100)
                ax[1].plot(lags[sorted_id]/1000, y, c='grey')
                peak_lst = (np.ones_like(lags[sorted_id]) * peak * 4 - 100)
                ax[2].plot(peak_lst/1000, y, c='grey')

                # upper histograms
                ax_histx[1].hist(lags, color='grey', density=False, range=(100, 1080), bins=trial_num)
                ax_histx[2].hist(peak_lst, color='grey', range=(100, 1080), density=True)
                xreal  = [peak*4-100]
                ax_histx[1].set_xticks(xreal)
                ax_histx[1].set_xticklabels(xrange)
                ax_histx[1].set_xlabel('latency\nlags (s)', loc='right', labelpad=-25, fontsize=13)
                ax_histx[2].set_xticks(xreal)
                ax_histx[2].set_xticklabels(xrange)
                ax_histx[2].set_xlabel('latency\nlags (s)', loc='right', labelpad=-25, fontsize=13)

                ############################## subplot 3, 4 ##########################################
                # sorted by C amplitude
                # amp_Pz = np.squeeze(results[0, 0]['c'][:, 27])
                amp_Pz = results[0, 0]['amp_c'][:, 27]
                # cov_lst = []
                data_Pz  = np.squeeze(data_lst[3])
                # for idx in range(data_Pz.shape[1]):
                #     cov = np.cov(data_Pz[:, idx], amp_Pz)
                #     cov_lst.append(cov[0, 1])
                sorted_id = sorted(range(len(amp_Pz)), key=lambda k: amp_Pz[k], reverse=True)
                data_Pz = data_Pz[:, sorted_id]
                ax[3].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)
                data_Pz_step3 = data_Pz
                ax[3].set_ylabel('trials (sorted by C amplitude variability)', fontsize=18)
                xreal  = [0]
                xrange = [0]
                ax_histx[3].set_xticks(xreal)
                ax_histx[3].set_xticklabels(xrange)
                ax_histx[3].hist(amp_Pz, range=(-40, 100), density=True, bins=trial_num, color='red')
                ax_histx[3].set_xlabel('amplitude\nvariablity (a.u.)', loc='right', labelpad=-25, fontsize=13)
                ax_histx[3].set_ylim(0, 0.1)

                max, min = np.max(data_lst[4]), np.min(data_lst[4])
                v_min_step4, v_max_step4 = -np.max([max, np.abs(min)]), np.max([max, np.abs(min)])
                cov_lst = []
                data_Pz  = np.squeeze(data_lst[4])
                new_amp_Pz = np.mean(data_Pz, axis=1)
                for idx in range(data_Pz.shape[1]):
                    cov = np.cov(data_Pz[:, idx], new_amp_Pz)
                    cov_lst.append(cov[0, 1])
                data_Pz = data_Pz[:, sorted_id]
                data_Pz_step4 = data_Pz
                ax[4].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min_step4, vmax=v_max_step4)
                ax[4].set_ylabel('trials (sorted by C amplitude variability)', fontsize=18)
                ax_histx[4].hist(cov_lst, range=(-40, 100), bins=trial_num, density=True, color='red')
                ax_histx[4].set_xticks(xreal)
                ax_histx[4].set_xticklabels(xrange)
                ax_histx[4].set_xlabel('amplitude\nvariablity (a.u.)', loc='right', labelpad=-25, fontsize=13)
                ax_histx[4].set_ylim(0, 0.1)

                ############################## subplot 5 & 6 ##########################################
                im = ax[5].pcolormesh(x, y, data_Pz_step3.T, cmap='jet', vmin=v_min, vmax=v_max)
                cbar = fig.colorbar(im, ax=ax[5], ticks=colorbar_locator, fraction=0.05)
                ax_histx[5].axes.xaxis.set_visible(False)
                ax_histx[5].spines.bottom.set_visible(False)

                im = ax[6].pcolormesh(x, y, data_Pz_step4.T, cmap='jet', vmin=v_min_step4, vmax=v_max_step4)
                cbar = fig.colorbar(im, ax=ax[6], fraction=0.05)
                cbar.set_label('amplitude', rotation=270, labelpad=10, fontsize=13)
                ax_histx[6].axes.xaxis.set_visible(False)
                ax_histx[6].spines.bottom.set_visible(False)

                ############################## UPF results #######################################
                results = scio.loadmat(os.path.join(data_path, UPF_name))['results']
                ax = []
                ax_histx = []
                for idx in range(7):
                    ax_up = fig.add_subplot(gs[2, idx])
                    ax_up.tick_params(axis="both", labelbottom=True, labelleft=False)
                    ax_up.spines['right'].set_color('none')
                    ax_up.spines['top'].set_color('none')
                    ax_up.spines['left'].set_color('none')
                    ax_up.spines['bottom'].set_position(('axes', 0.3))
                    ax_up.spines['bottom'].set_alpha(0.5)
                    ax_up.axes.yaxis.set_visible(False)
                    ax_up.tick_params(labelsize=10) 
                    ax_bottom = fig.add_subplot(gs[3, idx])
                    ax.append(ax_bottom)
                    ax_histx.append(ax_up)
                    

                data_lst = []
                for i in range(5):
                    data = results[0, 0][mat_name_lst['step{}'.format(i)]][49:294, 27, :]
                    data_lst.append(data)
                    ax[i].set_xlim(0.1, 1.08)
                    ax[i].xaxis.set_major_locator(x_major_locator)
                    ax[i].tick_params(labelsize=15) 
                    ax[i].set_yticks([])

                max, min = np.max(data_lst), np.min(data_lst)
                v_min, v_max = -np.max([max, np.abs(min)]), np.max([max, np.abs(min)])
                print(v_min, v_max)

                ############################## subplot 0 ##########################################
                rt = np.squeeze(results[0, 0]['rt'])
                trial_num = len(rt)
                x = np.linspace(0.1, 1.08, 245)
                y = np.linspace(0, trial_num, trial_num, endpoint=False)
                xrange = [0]

                # sorted by reaction time
                sorted_id = sorted(range(len(rt)), key=lambda k: rt[k], reverse=True)
                data_Pz = np.squeeze(data_lst[0])
                data_Pz = data_Pz[:, sorted_id]
                ax[0].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)
                ax[0].plot(rt[sorted_id]/1000, y, c='black')
                ax_histx[0].hist(rt, color='black', density=True, range=(100, 1080), bins=trial_num)
                xreal  = [0]
                ax_histx[0].set_xticks(xreal)
                ax_histx[0].set_xticklabels(xrange)

                ############################## subplot 1,2 ##########################################
                # sorted by C latency
                latency = np.squeeze(results[0, 0]['latency_c'])
                sorted_id = sorted(range(len(latency)), key=lambda k: latency[k], reverse=True)
                for idx in range(1, 3):
                    data_Pz  = np.squeeze(data_lst[idx])
                    data_Pz = data_Pz[:, sorted_id]
                    ax[idx].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)

                # plot latency lines
                peak = np.squeeze(results[0, 0]['ref_c_peak'])
                lags = (peak*4 + latency - 100)
                ax[1].plot(lags[sorted_id]/1000, y, c='grey')
                peak_lst = (np.ones_like(lags[sorted_id]) * peak * 4 - 100)
                ax[2].plot(peak_lst/1000, y, c='grey')

                # upper histograms
                ax_histx[1].hist(lags, color='grey', density=False, range=(100, 1080), bins=trial_num)
                ax_histx[2].hist(peak_lst, color='grey', range=(100, 1080), density=True)
                xreal  = [peak*4-100]
                ax_histx[1].set_xticks(xreal)
                ax_histx[1].set_xticklabels(xrange)
                ax_histx[2].set_xticks(xreal)
                ax_histx[2].set_xticklabels(xrange)

                ############################## subplot 3, 4 ##########################################
                # sorted by C amplitude
                # amp_Pz = np.squeeze(results[0, 0]['c'][:, 27])
                amp_Pz = results[0, 0]['amp_c'][:, 27]
                # cov_lst = []
                data_Pz  = np.squeeze(data_lst[3])
                # for idx in range(data_Pz.shape[1]):
                #     cov = np.cov(data_Pz[:, idx], amp_Pz)
                #     cov_lst.append(cov[0, 1])
                sorted_id = sorted(range(len(amp_Pz)), key=lambda k: amp_Pz[k], reverse=True)
                data_Pz = data_Pz[:, sorted_id]
                data_Pz_step3 = data_Pz
                ax[3].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)
                xreal  = [0]
                xrange = [0]
                ax_histx[3].set_xticks(xreal)
                ax_histx[3].set_xticklabels(xrange)
                ax_histx[3].hist(amp_Pz, range=(-40, 100), density=True, bins=trial_num, color='red')
                ax_histx[3].set_ylim(0, 0.1)

                max, min = np.max(data_lst[4]), np.min(data_lst[4])
                v_min_step4, v_max_step4 = -np.max([max, np.abs(min)]), np.max([max, np.abs(min)])
                cov_lst = []
                data_Pz  = np.squeeze(data_lst[4])
                new_amp_Pz = np.mean(data_Pz, axis=1)
                for idx in range(data_Pz.shape[1]):
                    cov = np.cov(data_Pz[:, idx], new_amp_Pz)
                    cov_lst.append(cov[0, 1])
                data_Pz = data_Pz[:, sorted_id]
                data_Pz_step4 = data_Pz
                ax[4].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min_step4, vmax=v_max_step4)
                ax_histx[4].hist(cov_lst, range=(-40, 100), bins=trial_num, density=True, color='red')
                ax_histx[4].set_xticks(xreal)
                ax_histx[4].set_xticklabels(xrange)
                ax_histx[4].set_ylim(0, 0.1)

                ############################## subplot 5 ##########################################
                im = ax[5].pcolormesh(x, y, data_Pz_step3.T, cmap='jet', vmin=v_min, vmax=v_max)
                cbar = fig.colorbar(im, ax=ax[5], ticks=colorbar_locator, fraction=0.05)
                ax_histx[5].axes.xaxis.set_visible(False)
                ax_histx[5].spines.bottom.set_visible(False)

                im = ax[6].pcolormesh(x, y, data_Pz_step4.T, cmap='jet', vmin=v_min_step4, vmax=v_max_step4)
                cbar = fig.colorbar(im, ax=ax[6], fraction=0.05)
                cbar.set_label('amplitude', rotation=270, labelpad=10, fontsize=13)
                ax_histx[6].axes.xaxis.set_visible(False)
                ax_histx[6].spines.bottom.set_visible(False)

                plt.tight_layout()
                # plt.show()
                plt.savefig(savefig_name, bbox_inches='tight')
                plt.close()



def plot_imagsec_one_with_hist_mvpa():
    sample_file_path = './252014_Fsp_PF_cc.mat'

    results = scio.loadmat(sample_file_path)['results']
    fig = plt.figure(figsize=(15, 4))
    gs = fig.add_gridspec(3, 5, height_ratios=(1, 2, 3), 
                        bottom=0.2, top=0.8,
                        wspace=0.4, hspace=0.0)
    
    ax, ax_histx, ax_mvpa = [], [], []
    for idx in range(5):
        ax_bottom = fig.add_subplot(gs[1, idx])
        ax_up = fig.add_subplot(gs[0, idx])
        ax_up.tick_params(axis="both", labelbottom=True, labelleft=False)
        ax_up.spines['right'].set_color('none')
        ax_up.spines['top'].set_color('none')
        ax_up.spines['left'].set_color('none')
        ax_up.spines['bottom'].set_position(('axes', 0.3))
        ax_up.spines['bottom'].set_alpha(0.5)
        ax_up.axes.yaxis.set_visible(False)
        ax_up.tick_params(labelsize=8) 
        ax.append(ax_bottom)
        ax_histx.append(ax_up)
        ax_mvpa.append(fig.add_subplot(gs[2, idx]))

    data_lst = []
    for i in range(5):
        data = results[0, 0][mat_name_lst['step{}'.format(i)]][:, 27, :]
        data_lst.append(data)
        ax[i].set_xlim(-100, 1500)
        ax[i].yaxis.set_major_locator(y_major_locator)
        ax[i].tick_params(labelsize=8)  

    max, min = np.max(data_lst), np.min(data_lst)
    v_min, v_max = -np.max([max, np.abs(min)]), np.max([max, np.abs(min)])
    print(v_min, v_max)

    ############################## subplot 0 ##########################################
    rt = np.squeeze(results[0, 0]['rt'])
    trial_num = len(rt)
    x = np.arange(-100, 1500, 4)
    y = np.linspace(0, trial_num, trial_num, endpoint=False)
    xrange = [0]

    # sorted by reaction time
    sorted_id = sorted(range(len(rt)), key=lambda k: rt[k], reverse=True)
    data_Pz = np.squeeze(data_lst[0])
    data_Pz = data_Pz[:, sorted_id]
    ax[0].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)
    ax[0].set_ylabel('trial (sorted by reaction time)')
    ax[0].plot(rt[sorted_id], y, c='black')
    ax_histx[0].hist(rt, color='black', density=True, range=(-100, 1500), bins=trial_num)
    ax_histx[0].set_xlabel('reaction\ntime (ms)', loc='right', labelpad=-20)
    xreal  = [0]
    ax_histx[0].set_xticks(xreal)
    ax_histx[0].set_xticklabels(xrange)

    ############################## subplot 1,2 ##########################################
    # sorted by C latency
    latency = np.squeeze(results[0, 0]['latency_c'])
    sorted_id = sorted(range(len(latency)), key=lambda k: latency[k], reverse=True)
    for idx in range(1, 3):
        data_Pz  = np.squeeze(data_lst[idx])
        data_Pz = data_Pz[:, sorted_id]
        ax[idx].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)
        ax[idx].set_ylabel('trial (sorted by C latency)')

    # plot latency lines
    peak = np.squeeze(results[0, 0]['ref_c_peak'])
    lags = peak*4 + latency - 100
    ax[1].plot(lags[sorted_id], y, c='grey')
    peak_lst = np.ones_like(lags[sorted_id]) * peak * 4 - 100
    ax[2].plot(peak_lst, y, c='grey')

    # upper histograms
    ax_histx[1].hist(lags, color='grey', density=False, range=(-100, 1500), bins=trial_num)
    ax_histx[2].hist(peak_lst, color='grey', range=(-100, 1500), density=True)
    xreal  = [peak*4-100]
    ax_histx[1].set_xticks(xreal)
    ax_histx[1].set_xticklabels(xrange)
    ax_histx[1].set_xlabel('latency\nlags (ms)', loc='right', labelpad=-20)
    ax_histx[2].set_xticks(xreal)
    ax_histx[2].set_xticklabels(xrange)
    ax_histx[2].set_xlabel('latency\nlags (ms)', loc='right', labelpad=-20)

    ############################## subplot 3, 4 ##########################################
    # sorted by C amplitude
    amp_Pz = np.squeeze(results[0, 0]['c'][:, 27])
    cov_lst = []
    data_Pz  = np.squeeze(data_lst[3])
    for idx in range(data_Pz.shape[1]):
        cov = np.cov(data_Pz[:, idx], amp_Pz)
        cov_lst.append(cov[0, 1])
    sorted_id = sorted(range(len(cov_lst)), key=lambda k: cov_lst[k], reverse=True)
    data_Pz = data_Pz[:, sorted_id]
    ax[3].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)
    ax[3].set_ylabel('trial (sorted by C amplitudes)')
    xreal  = [0]
    xrange = [0]
    ax_histx[3].set_xticks(xreal)
    ax_histx[3].set_xticklabels(xrange)
    ax_histx[3].hist(cov_lst, range=(-40, 100), density=True, bins=trial_num, color='red')
    ax_histx[3].set_xlabel('amplitude\nvariablity', loc='right', labelpad=-20)
    ax_histx[3].set_ylim(0, 0.1)

    cov_lst = []
    data_Pz  = np.squeeze(data_lst[4])
    for idx in range(data_Pz.shape[1]):
        cov = np.cov(data_Pz[:, idx], amp_Pz)
        cov_lst.append(cov[0, 1])
    data_Pz = data_Pz[:, sorted_id]
    ax[4].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max)
    ax[4].set_ylabel('trial (sorted by C amplitudes)')
    ax_histx[4].hist(cov_lst, range=(-40, 100), bins=trial_num, density=True, color='red')
    ax_histx[4].set_xticks(xreal)
    ax_histx[4].set_xticklabels(xrange)
    ax_histx[4].set_xlabel('amplitude\nvariablity', loc='right', labelpad=-20)
    ax_histx[4].set_ylim(0, 0.1)

    ###################################  MVPA     ##########################################
    k_fold = 4

    random = np.ones((1, 271, 271)) * 0.5
    diff_max = np.abs(0.71-0.5) 
    vmin, vmax = 0.5-diff_max, 0.5+diff_max 

    mvpa_path = './latency/results/classification/mvpa/'

    EEGNet_ge = np.zeros((k_fold, 271, 271))
    for k in range(k_fold):
        EEGNet_ge[k] = np.load('{}/step{}/ge_{}.npy'.format(mvpa_path, 0, k))
    _, p_value = ttest_1samp(EEGNet_ge, random)
    p_boolean = (p_value <= 0.01)
    p_boolean = p_boolean.squeeze()

    for step in range(5):
        EEGNet_ge = np.zeros((k_fold, 271, 271))
        if step != 4:
            for k in range(k_fold):
                EEGNet_ge[k] = np.load('{}/step{}/ge_{}.npy'.format(mvpa_path, step, k))
        else:
            for k in range(k_fold):
                EEGNet_ge[k] = np.load('{}/step{}/ge_{}.npy'.format(mvpa_path, step, k))

        ge_mean = EEGNet_ge.mean(axis=0)
        mask = np.ma.masked_where(~p_boolean, ge_mean)
        print(np.max(ge_mean), np.min(ge_mean))

        im = ax_mvpa[step].imshow(ge_mean, interpolation=interpolation, cmap=cmap, origin='lower', extent=[0.06, 1.5, 0.06, 1.5], vmin=vmin, vmax=vmax)
        ax_mvpa[step].imshow(mask, interpolation=interpolation, cmap=cmap, origin='lower')
        ax_mvpa[step].set_title('Step{}'.format(step), fontsize=15, loc='center')
        ax_mvpa[step].set_xticks(xreal)
        ax_mvpa[step].set_xticklabels(xrange)
        ax_mvpa[step].axes.yaxis.set_visible(False)
        

    cbar = fig.colorbar(im, ax=ax_mvpa[4], ticks=colorbar_locator, fraction=0.05)
    cbar.set_label('Accuracy', rotation=270, labelpad=20, fontsize=15)
    ax_mvpa[0].set_title('Step0', fontsize=15, loc='center')
    ax_mvpa[0].set_xlabel('Generalization Time (s)', fontsize=16)
    ax_mvpa[0].set_ylabel('Training Time (s)', fontsize=16)
    ax_mvpa[0].set_yticks(xreal)
    ax_mvpa[0].set_yticklabels(xrange)
    ax_mvpa[0].axes.yaxis.set_visible(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_imagsec()
    # plot_imagsec_one_with_hist()
    plot_imagsec_with_hist()
    # plot_imagsec_one_with_hist_mvpa()