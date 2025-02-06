import os
import scipy.io as scio
import pandas as pd
import numpy as np

face_or_house = 'face'

if face_or_house == 'face':
    task_param_lst = ['Facc', 'Fsp']
elif face_or_house == 'house':
    task_param_lst = ['Hacc', 'Hsp']
familiar_param_lst = ['PF', 'UPF']
old_data_path = '../../new_dataset/{}-cognition-data/mat/'.format(face_or_house)
new_data_path = '../../new_dataset/{}-cognition-data/without_outliers/'.format(face_or_house)


def save_subject_rt(version='old'):
    if version == 'old':
        data_path = old_data_path
        save_name = './results/reaction_time/{}_subject_rt.xlsx'.format(face_or_house)
    elif version == 'new':
        data_path = new_data_path
        save_name = './results/reaction_time/{}_new_subject_rt.xlsx'.format(face_or_house)
    data_files = os.listdir(data_path)
    names = []
    for data_file in data_files:
        if data_file.endswith('.mat'):
            content = data_file.split('.')[0].split('_')
            names.append(content[0])
    names = set(names)
    subject_rt_dict = {}                                                # subject_rt_dict: {task_name: {subject: rt_num}}
    for task_param in task_param_lst:
        for familiar_param in familiar_param_lst:
            subject_rt = {}
            for data_file in data_files:
                if data_file.endswith('.mat'):
                    content = data_file.split('.')[0].split('_')
                    if content[1] == task_param and content[2] == familiar_param:
                        rt = scio.loadmat(os.path.join(data_path, data_file))['rt']
                        if version == 'old':
                            subject_rt[content[0]] = rt.shape[1]           # subject_rt: {subject: rt_num}
                        elif version == 'new':
                            subject_rt[content[0]] = rt.shape[0]

            name = task_param+'_'+familiar_param
            for item in names:
                if item not in subject_rt:
                    print(name, item)
            new_subject_rt = {key: subject_rt[key] for key in sorted(subject_rt.keys())}  # sorted according to subject's name
            subject_rt_dict[name] = new_subject_rt

    # rearrange this dict
    rt_dict = {}                 
    for key in subject_rt_dict.keys():
        rt_dict[key] = subject_rt_dict[key].values()
    rt_dict['subject'] = subject_rt_dict[key].keys()
    for key in subject_rt_dict.keys():
        print(len(subject_rt_dict[key]))
    df = pd.DataFrame(rt_dict)
    df.set_index(['subject'], inplace=True)
    df.to_excel(save_name)
    print(rt_dict)


def save_rt_in_each_task(version):

    for task_param in task_param_lst:
        for familiar_param in familiar_param_lst:
            if version == 'old':
                data_path = old_data_path
                save_name = './results/reaction_time/{}_{}.npy'.format(task_param, familiar_param)
            elif version == 'new':
                data_path = new_data_path
                save_name = './results/reaction_time/{}_{}_without_outliers.npy'.format(task_param, familiar_param)
            data_files = os.listdir(data_path)

            rt_lst = []
            num = 0
            for data_file in data_files:
                if data_file.endswith('.mat'):
                    content = data_file.split('.')[0].split('_')
                    if content[1] == task_param and content[2] == familiar_param:
                        rt = scio.loadmat(os.path.join(data_path, data_file))['rt']
                        rt_lst.extend(rt[0])
                        num += 1
            np.save(save_name, rt_lst)
            print(task_param, familiar_param)
            print(num)
            print(np.mean(rt_lst))
            print('error rate:', 1-len(rt_lst)/(72*num))       


def plot_rt_distribution():
    # # plot the reaction time distribution
    import matplotlib.pyplot as plt

    p_latency = np.load('./results/reaction_time/Fsp_PF.npy')
    up_latency = np.load('./results/reaction_time/Fsp_UPF.npy')

    print('primed max latency:{}, uprimed max latency:{}'.format(np.max(p_latency), np.max(up_latency)))
    print('primed min latency:{}, uprimed min latency:{}'.format(np.min(p_latency), np.min(up_latency)))
    print('primed length:{}, uprimed length:{}'.format(len(p_latency), len(up_latency)))
    plt.hist(p_latency, bins=2000, range=(1, 2000), density=False, cumulative=False, alpha=0.5)      
    plt.hist(up_latency, bins=2000, range=(1, 2000), density=False, cumulative=False, alpha=0.5)      
    plt.axvline(np.median(p_latency), color='k', linestyle='--')
    plt.axvline(np.median(up_latency), color='k', linestyle='--')
    plt.show()


def tukeys_method(task_name):
    
    latency = np.load('./rt_related/{}.npy'.format(task_name), allow_pickle=True)
    q1 = np.quantile(latency, 0.25)
    q3 = np.quantile(latency, 0.75)
    iqr = q3-q1
    inner_fence = 1.5*iqr
    outer_fence = 3*iqr
    
    #inner fence lower and upper end
    inner_fence_le = q1-inner_fence
    inner_fence_ue = q3+inner_fence
    
    #outer fence lower and upper end
    outer_fence_le = q1-outer_fence
    outer_fence_ue = q3+outer_fence
    
    outliers_prob = []
    outliers_poss = []
    for index, x in enumerate(latency):
        if x <= outer_fence_le or x >= outer_fence_ue:
            outliers_prob.append(index)
    for index, x in enumerate(latency):
        if x <= inner_fence_le or x >= inner_fence_ue:
            outliers_poss.append(index)
    return outliers_prob, outliers_poss, outer_fence_ue


def delete_outliers():
    data_files = os.listdir(old_data_path)
    for task_param in task_param_lst:
        for familiar_param in familiar_param_lst:
            probable_outliers, _, outer_fence_ue = tukeys_method('{}_{}'.format(task_param, familiar_param))
            rt_lst = []
            outliers_num = 0
            for data_file in data_files:
                if data_file.endswith('.mat'):
                    content = data_file.split('.')[0].split('_')
                    if content[1] == task_param and content[2] == familiar_param:
                        mat = scio.loadmat(os.path.join(old_data_path, data_file))
                        data = mat['data']
                        rt = np.squeeze(mat['rt'])
                        if rt.ndim != 0:
                            if np.max(rt) >= outer_fence_ue:
                                idx = np.where(rt >= outer_fence_ue)
                                print(idx[0].shape)
                                outliers_num += len(idx[0])
                                new_data = np.delete(data, idx, axis=2)
                                new_rt = np.delete(rt, idx, axis=0)
                                rt_lst.extend(new_rt)
                                rt = new_rt[:, np.newaxis]
                                scio.savemat(os.path.join(new_data_path, data_file), {'data': new_data, 'rt': rt})
                            else:
                                rt = rt[:, np.newaxis]
                                scio.savemat(os.path.join(new_data_path, data_file), {'data': data, 'rt': rt})

            print(task_param, familiar_param)
            # print(np.max(rt_lst))  
            assert len(probable_outliers) == outliers_num

# save_subject_rt('old')
# save_rt_in_each_task('old')
# delete_outliers()
plot_rt_distribution()
# save_rt_in_each_task('new')
# save_subject_rt('new')
