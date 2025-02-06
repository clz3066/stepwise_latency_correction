import scipy.io as scio
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from matplotlib import cm
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
trials_img_path                     = './latency/paper_figures/fig2/trials/'
trial_imgs_files                    = os.listdir(trials_img_path)


def b_ERP_topo(step_idx):
    '''
    only for colorbar
    '''
    mat = scio.loadmat('./ERP_results/ERP/Fsp_step{}.mat'.format(step_idx))
    erp = mat['erp'][:, 49:294]
    print(erp.shape)
    
    evoked = mne.EvokedArray(erp, info, tmin=0.11)
    print(np.min(evoked.data))
    print(np.max(evoked.data))
    if step_idx < 3:
        cnorm = cm.colors.Normalize(vmax=3.8, vmin=-3.8)
        fig = evoked.plot_joint(picks='eeg', times=(0.4, 0.64), title=None, show=False,
                        ts_args={'gfp':True, 'scalings':dict(eeg=1), 'ylim':dict(eeg=[-3.8, 3.8]), 'gfp':True}, 
                        topomap_args={'average':0.124, 'scalings':dict(eeg=1), 'time_format':'%0.2f s', 'cnorm':cnorm}, 
                        highlight=[(0.34, 0.46), (0.58, 0.70)])
        # fig.set_size_inches((6, 4))
    elif step_idx == 3:
        cnorm = cm.colors.Normalize(vmax=3.8, vmin=-3.8)
        fig = evoked.plot_joint(picks='eeg', times=(0.32, 0.58), title=None, show=False,
                        ts_args={'gfp':True, 'scalings':dict(eeg=1), 'ylim':dict(eeg=[-3.8, 3.8]), 'gfp':True}, 
                        topomap_args={'average':0.124, 'scalings':dict(eeg=1), 'time_format':'%0.2f s', 'cnorm':cnorm}, 
                        highlight=[(0.26, 0.38), (0.52, 0.64)])
        fig.set_size_inches((6, 4))
    else:
        fig = evoked.plot_joint(picks='eeg', times=(0.32, 0.58), title=None, show=False,
                        ts_args={'gfp':True, 'scalings':dict(eeg=1), 'ylim':dict(eeg=[-0.3, 0.3]), 'gfp':True}, 
                        topomap_args={'average':0.124, 'scalings':dict(eeg=1), 'time_format':'%0.2f s'}, 
                        highlight=[(0.26, 0.38), (0.52, 0.64)])
        fig.set_size_inches((6, 4))
    # plt.show()
    plt.savefig('./latency/paper_figures/fig2/erp_step{}_topo.svg'.format(step_idx), bbox_inches='tight', transparent=True)
    plt.close()

# b_ERP_topo(0)


def b_erp_topo_Pz(step_idx):
    '''
    do not use anymore
    '''

    fig = plt.figure(figsize=(6, 4))
    grid = plt.GridSpec(5, 5, wspace=0.3, hspace=0.05, left=0.1, bottom=0.1)
    
    ax_one = plt.subplot(grid[0, :])
    ax_erp = plt.subplot(grid[1:3, :])
    ax_topo1 = plt.subplot(grid[3, 1])
    ax_topo2 = plt.subplot(grid[3, 2])
    ax_one.spines.top.set_visible(False)
    ax_one.spines.right.set_visible(False)
    ax_one.spines.bottom.set_visible(False)
    ax_one.axes.xaxis.set_visible(False)

    x = np.arange(0.11, 1.089, 0.004)
    mat = scio.loadmat('./ERP_results/ERP/Fsp_step{}.mat'.format(step_idx))
    
    ax_one.plot(x, mat['p'][27, 49:294], label='primed ERP (Pz)', alpha=0.5, color='black')
    ax_one.plot(x, mat['up'][27, 49:294], label='unprimed ERP (Pz)', alpha=0.5, color='green')
    print(np.max(mat['p'][27, 49:294]), np.max(mat['up'][27, 49:294]))
    print(np.min(mat['p'][27, 49:294]), np.min(mat['up'][27, 49:294]))
    ax_one.set_xlim([0.11, 1.089])
    if step_idx == 0:
        ax_one.legend(loc='upper right')

    erp = mat['erp'][:, 49:294]
    evoked = mne.EvokedArray(erp, info, tmin=0.11)
    if step_idx < 3:
        evoked.plot(picks='eeg', show=False, axes=ax_erp, titles=None, scalings=1, ylim=dict(eeg=[-3.8, 3.8]), gfp=True, spatial_colors=True,
                    highlight=[(0.34, 0.46), (0.58, 0.7)])
        evoked.plot_topomap(ch_type="eeg", times=(0.4, 0.64), average=0.124, scalings=1, vlim=(-3.6, 3.6), 
                    time_format='%0.2f s', axes=[ax_topo1, ax_topo2], colorbar=False)
        ax_one.set_ylim([-2, 12])
    elif step_idx == 3:
        evoked.plot(picks='eeg', show=False, axes=ax_erp, titles=None, scalings=1, ylim=dict(eeg=[-3.8, 3.8]), gfp=True, spatial_colors=True, 
                    highlight=[(0.26, 0.38), (0.52, 0.64)]) 
        evoked.plot_topomap(ch_type="eeg", times=(0.32, 0.58), average=0.124, scalings=1, vlim=(-3.6, 3.6), 
                    time_format='%0.2f s', axes=[ax_topo1, ax_topo2], colorbar=False)
        ax_one.set_ylim([-2, 12])
    elif step_idx == 4:
        evoked.plot(picks='eeg', show=False, axes=ax_erp, titles=None, scalings=1, ylim=dict(eeg=[-0.3, 0.3]), gfp=True, spatial_colors=True, 
                    highlight=[(0.26, 0.38), (0.52, 0.64)])  
        evoked.plot_topomap(ch_type="eeg", times=(0.32, 0.58), average=0.124, scalings=1, 
                    time_format='%0.2f s', axes=[ax_topo1, ax_topo2], colorbar=False)    
        ax_one.set_ylim([-0.2, 1.5])


    # plt.show()
    plt.savefig('./latency/paper_figures/fig2/erp_step{}.svg'.format(step_idx), bbox_inches='tight', transparent=True)



def c_plot_trials():

    for file in data_files:
        if file.endswith('.mat'):
            content = file.split('.')[0].split('_')
            print(content)
            if content[1] == 'Fsp':
                results = scio.loadmat(os.path.join(data_path, file))['results']
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

                plt.tight_layout()
                # plt.show()
                plt.savefig('{}/{}.jpg'.format(trials_img_path, file.split('.')[0]))
                plt.close()



def c_plot_trials_with_hist():
    for file in trial_imgs_files:
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
            
            savefig_name = './latency/paper_figures/fig2/trials&hist_png/{}.png'.format(contents[0])
            if not os.path.exists(savefig_name):
            
                fig = plt.figure(figsize=(225*mm, 80*mm))
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
                    ax_up.tick_params(labelsize=6) 
                    ax_bottom = fig.add_subplot(gs[1, idx])
                    ax.append(ax_bottom)
                    ax_histx.append(ax_up)
                    

                data_lst = []
                for i in range(5):
                    data = results[0, 0][mat_name_lst['step{}'.format(i)]][49:294, 27, :]
                    data_lst.append(data)
                    ax[i].xaxis.set_major_locator(x_major_locator)
                    ax[i].tick_params(labelsize=6) 
                    ax[i].set_xticks([])
                    ax[i].set_yticks([])
                    if i == 0:
                        ax_histx[i].set_title('Baseline', fontsize=8)
                    else:
                        ax_histx[i].set_title('Step{}'.format(i), fontsize=8)

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
                ax[0].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max, rasterized=True)
                ax[0].set_ylabel('trials (sorted by reaction time)', fontsize=8)
                ax[0].plot(rt[sorted_id]/1000, y, c='black')
                ax_histx[0].hist(rt, color='black', density=True, range=(100, 1080), bins=trial_num)
                ax_histx[0].set_xlabel('reaction\ntime (s)', loc='right', labelpad=-25, fontsize=6)
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
                    ax[idx].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max, rasterized=True)
                    ax[idx].set_ylabel('trials (sorted by C latency)', fontsize=8)

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
                ax_histx[1].set_xlabel('latency\nlags (s)', loc='right', labelpad=-25, fontsize=6)
                ax_histx[2].set_xticks(xreal)
                ax_histx[2].set_xticklabels(xrange)
                ax_histx[2].set_xlabel('latency\nlags (s)', loc='right', labelpad=-25, fontsize=6)

                ############################## subplot 3, 4 ##########################################
                # sorted by C amplitude
                amp_Pz = results[0, 0]['amp_c'][:, 27]
                data_Pz  = np.squeeze(data_lst[3])
                sorted_id = sorted(range(len(amp_Pz)), key=lambda k: amp_Pz[k], reverse=True)
                data_Pz = data_Pz[:, sorted_id]
                ax[3].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max, rasterized=True)
                data_Pz_step3 = data_Pz
                ax[3].set_ylabel('trials (sorted by C amplitude variability)', fontsize=8)
                xreal  = [0]
                xrange = [0]
                ax_histx[3].set_xticks(xreal)
                ax_histx[3].set_xticklabels(xrange)
                ax_histx[3].hist(amp_Pz, range=(-40, 100), density=True, bins=trial_num, color='red')
                ax_histx[3].set_xlabel('amplitude\nvariablity (a.u.)', loc='right', labelpad=-25, fontsize=6)
                ax_histx[3].set_ylim(0, 0.1)

                max, min = np.max(data_lst[4]), np.min(data_lst[4])
                v_min_step4, v_max_step4 = -np.max([max, np.abs(min)]), np.max([max, np.abs(min)])
                cov_lst    = []
                data_Pz    = np.squeeze(data_lst[4])
                new_amp_Pz = np.mean(data_Pz, axis=1)
                for idx in range(data_Pz.shape[1]):
                    cov = np.cov(data_Pz[:, idx], new_amp_Pz)
                    cov_lst.append(cov[0, 1])
                data_Pz = data_Pz[:, sorted_id]
                data_Pz_step4 = data_Pz
                ax[4].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min_step4, vmax=v_max_step4, rasterized=True)
                ax[4].set_ylabel('trials (sorted by C amplitude variability)', fontsize=8)
                ax_histx[4].hist(cov_lst, range=(-40, 100), bins=trial_num, density=True, color='red')
                ax_histx[4].set_xticks(xreal)
                ax_histx[4].set_xticklabels(xrange)
                ax_histx[4].set_xlabel('amplitude\nvariablity (a.u.)', loc='right', labelpad=-25, fontsize=6)
                ax_histx[4].set_ylim(0, 0.1)

                ############################## subplot 5 & 6 ##########################################
                im = ax[5].pcolormesh(x, y, data_Pz_step3.T, cmap='jet', vmin=v_min, vmax=v_max, rasterized=True)
                cbar = fig.colorbar(im, ax=ax[5], ticks=colorbar_locator, fraction=0.1, aspect=13)
                cbar.ax.tick_params(labelsize=6)
                ax_histx[5].axes.xaxis.set_visible(False)
                ax_histx[5].spines.bottom.set_visible(False)

                im = ax[6].pcolormesh(x, y, data_Pz_step4.T, cmap='jet', vmin=v_min_step4, vmax=v_max_step4, rasterized=True)
                cbar = fig.colorbar(im, ax=ax[6], fraction=0.1, aspect=13)
                cbar.set_label('$\mu V$', rotation=270, labelpad=5, fontsize=6)
                cbar.ax.tick_params(labelsize=6)
                ax_histx[6].axes.xaxis.set_visible(False)
                ax_histx[6].spines.bottom.set_visible(False)

                print(v_min, v_max, v_min_step4, v_max_step4)
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
                    ax_up.tick_params(labelsize=6) 
                    ax_bottom = fig.add_subplot(gs[3, idx])
                    ax.append(ax_bottom)
                    ax_histx.append(ax_up)
                    

                data_lst = []
                for i in range(5):
                    data = results[0, 0][mat_name_lst['step{}'.format(i)]][49:294, 27, :]
                    data_lst.append(data)
                    ax[i].set_xlim(0.1, 1.08)
                    ax[i].xaxis.set_major_locator(x_major_locator)
                    ax[i].tick_params(labelsize=6) 
                    ax[i].set_yticks([])

                max, min = np.max(data_lst), np.min(data_lst)
                v_min, v_max = -np.max([max, np.abs(min)]), np.max([max, np.abs(min)])
                # print(v_min, v_max)

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
                ax[0].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max, rasterized=True)
                ax[0].plot(rt[sorted_id]/1000, y, c='black')
                ax[0].set_xlabel('Time (s)', fontsize=6)
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
                    ax[idx].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max, rasterized=True)
                    ax[idx].set_xlabel('Time (s)', fontsize=6)

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
                amp_Pz = results[0, 0]['amp_c'][:, 27]
                data_Pz  = np.squeeze(data_lst[3])
                sorted_id = sorted(range(len(amp_Pz)), key=lambda k: amp_Pz[k], reverse=True)
                data_Pz = data_Pz[:, sorted_id]
                data_Pz_step3 = data_Pz
                ax[3].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min, vmax=v_max, rasterized=True)
                ax[3].set_xlabel('Time (s)', fontsize=6)
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
                ax[4].pcolormesh(x, y, data_Pz.T, cmap='jet', vmin=v_min_step4, vmax=v_max_step4, rasterized=True)
                ax[4].set_xlabel('Time (s)', fontsize=6)
                ax_histx[4].hist(cov_lst, range=(-40, 100), bins=trial_num, density=True, color='red')
                ax_histx[4].set_xticks(xreal)
                ax_histx[4].set_xticklabels(xrange)
                ax_histx[4].set_ylim(0, 0.1)

                ############################## subplot 5 ##########################################
                im = ax[5].pcolormesh(x, y, data_Pz_step3.T, cmap='jet', vmin=v_min, vmax=v_max)
                cbar = fig.colorbar(im, ax=ax[5], ticks=colorbar_locator, fraction=0.1, aspect=13)
                cbar.ax.tick_params(labelsize=6)
                ax_histx[5].axes.xaxis.set_visible(False)
                ax_histx[5].spines.bottom.set_visible(False)

                im = ax[6].pcolormesh(x, y, data_Pz_step4.T, cmap='jet', vmin=v_min_step4, vmax=v_max_step4, rasterized=True)
                cbar = fig.colorbar(im, ax=ax[6], fraction=0.1, aspect=13)
                cbar.set_label('$\mu V$', rotation=270, labelpad=5, fontsize=6)
                ax_histx[6].axes.xaxis.set_visible(False)
                ax_histx[6].spines.bottom.set_visible(False)
                cbar.ax.tick_params(labelsize=6)

                plt.tight_layout()
                # plt.show()
                print(file_name, v_min, v_max, v_min_step4, v_max_step4)
                plt.savefig(savefig_name, bbox_inches='tight', transparent=True)
                plt.close()

c_plot_trials_with_hist()