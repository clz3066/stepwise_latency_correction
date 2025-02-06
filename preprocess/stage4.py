import scipy.io as scio
import os
import numpy as np

data_path = '../../face-cognition-data/step4/'
save_path = '../../face-cognition-data/gfp_step4/'


def gfp_step4():
    files = os.listdir(data_path)
    for file in files:
        if file.endswith('.mat'):
            print(file)
            mat = scio.loadmat(os.path.join(data_path, file))['output'][0, 0]

            trials = mat['st_c_synced_between_subjects'] 

            trial_num = trials.shape[2]
            new_single_trials_arr = np.zeros((trial_num, 400, 41))
            
            for trial_idx in range(trial_num):

                c_trial = trials[:, :, trial_idx]
                ave = np.tile(np.mean(c_trial, axis=1), (41, 1)).T
                gfp = np.tile(np.std(c_trial, axis=1), (41, 1)).T
                with np.errstate(divide='ignore', invalid='ignore'):
                    aved_c = c_trial - ave
                    c = np.true_divide(aved_c, gfp)
                    c[c == np.inf] = 0
                    single_trial = np.nan_to_num(c)
                    new_single_trials_arr[trial_idx] = single_trial

            new = new_single_trials_arr.transpose(1, 2, 0)
            print(new.shape)
            data = {'aligned_amp': new}
            scio.savemat('{}/{}'.format(save_path, file), data)

gfp_step4()