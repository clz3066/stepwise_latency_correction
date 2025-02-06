import os 
import scipy.io as scio
import random
import numpy as np
import torch.utils.data as Data
import torch
import network
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR
import argparse
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MvpaParallel():
    def __init__(self, task, name_in_mat, training_seed):
        self.task = task
        self.name_in_mat = name_in_mat
        self.training_seed = int(training_seed)
        self.mat_name_lst = {'step0':'original', 'step1':'st_c', 'step2':'st_c_synced', 
                            'step3':'st_c_synced_within_subject', 'step4':'aligned_amp',
                            's_component':'st_s', 'r_component':'st_r'}
        self.block1_kernel = 20
        self.block2_kernel = 10
        # self.data_path = '../../../../Dataset/Facc_Fsp_sample/'
        self.data_path = '/data/yilinli/Facc_Fsp/'
        # self.data_path = '/home/yilin/Facc_Fsp'


    def reset_weights(self, m):
        '''
        Try resetting model weights to avoid
        weight leakage.
        '''
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


    def seed_torch(self, seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def split_subject_name(self, seed):
        self.seed_torch(seed=seed)

        files = os.listdir(self.data_path)
        files = sorted(files)
        random.shuffle(files)
        name_lst = []
        for file in files:
            if file.endswith('.mat'):
                name = file.split('.')[0].split('_')[0]
                if name not in name_lst:
                    name_lst.append(name)
        num_array = np.array(name_lst)

        # Split the array into 10 groups
        self.groups = np.array_split(num_array, 10)

        # Print the groups
        for i, group in enumerate(self.groups):
            print(f"Group {i}: {group}")


    ############################# readin training or testing dataset  ###########################                      
    def training_dataset(self, k, for_training):
        '''
        split all subjects into 10 groups
        '''
        self.split_subject_name(seed=self.training_seed)
        files = os.listdir(self.data_path)
        group = self.groups[k]

        primed_num = 0
        unprimed_num = 0
        for file in files:
            if file.endswith('.mat'):
                content = file.split('.')[0].split('_')
                condition = content[2]
                if for_training:
                    if content[0] not in group:
                        if content[1] == self.task:
                            mat = scio.loadmat(os.path.join(self.data_path, file))['results']
                            name = self.mat_name_lst[self.name_in_mat]
                            data = np.transpose(mat[0, 0][name], (1, 0, 2))     
                            
                            if condition == 'PF':
                                primed_num += data.shape[2]                       
                            elif condition == 'UPF':
                                unprimed_num += data.shape[2]
                        else:
                            continue
                else:   # test_dataset
                    if content[0] in group:
                        if content[1] == self.task:
                            mat = scio.loadmat(os.path.join(self.data_path, file))['results']
                            name = self.mat_name_lst[self.name_in_mat]
                            data = np.transpose(mat[0, 0][name], (1, 0, 2))     
                            
                            if condition == 'PF':
                                primed_num += data.shape[2]                       
                            elif condition == 'UPF':
                                unprimed_num += data.shape[2]
                        else:
                            continue     

        channels, samples = data.shape[0], data.shape[1]

        print("primed number:   {}".format(primed_num))
        print("unprimed number: {}".format(unprimed_num))

        # primed=0 unprimed=1
        X_primed = np.empty(shape = [primed_num, channels, samples], dtype = float) 
        X_unprimed = np.empty(shape = [unprimed_num, channels, samples], dtype = float) 
        y_primed = np.zeros(shape = [primed_num,], dtype = int)
        y_unprimed = np.ones(shape = [unprimed_num,], dtype = int)

        primed_chans = 0
        unprimed_chans = 0
        for file in files:
            if file.endswith('.mat'):
                content = file.split('.')[0].split('_')
                condition = content[2]
                if for_training:                           
                    if content[0] not in group:
                        if content[1] == self.task:
                            mat = scio.loadmat(os.path.join(self.data_path, file))['results']
                            name = self.mat_name_lst[self.name_in_mat]
                            data = np.transpose(mat[0, 0][name], (1, 0, 2)) 
                            
                            if condition == "PF":                           
                                for i in range(data.shape[2]):
                                    X_primed[primed_chans] = data[:, :, i]
                                    primed_chans += 1                            
                            elif condition == 'UPF':                           
                                for i in range(data.shape[2]):
                                    X_unprimed[unprimed_chans] = data[:, :, i]
                                    unprimed_chans += 1                       
                            else:
                                continue
                        else:
                            continue
                else:       # test_dataset
                    if content[0] in group:
                        if content[1] == self.task:
                            mat = scio.loadmat(os.path.join(self.data_path, file))['results']
                            name = self.mat_name_lst[self.name_in_mat]
                            data = np.transpose(mat[0, 0][name], (1, 0, 2)) 
                             
                            if condition == "PF":                           
                                for i in range(data.shape[2]):
                                    X_primed[primed_chans] = data[:, :, i]
                                    primed_chans += 1                            
                            elif condition == 'UPF':                           
                                for i in range(data.shape[2]):
                                    X_unprimed[unprimed_chans] = data[:, :, i]
                                    unprimed_chans += 1                       
                            else:
                                continue
                        else:
                            continue
        
        assert primed_num == primed_chans
        assert unprimed_num == unprimed_chans
        
        X = np.concatenate((X_primed, X_unprimed), axis=0)
        y = np.concatenate((y_primed, y_unprimed), axis=0)
        shuffle_ix = np.random.permutation(np.arange(len(y)))
        X = X[shuffle_ix,:]
        y = y[shuffle_ix]
        
        kernels =  1
        X = X.reshape(X.shape[0], kernels, channels, samples)
        X = torch.from_numpy(X).type(torch.float32)
        y = torch.from_numpy(y).type(torch.int64)
        return X[:, :, :, 49:275], y
            

    def testing_dataset(self, subject_name):
        '''
        one subject as testing dataset
        '''
        
        p_name = subject_name + '_{}_PF_cc.mat'.format(self.task)
        mat = scio.loadmat(os.path.join(self.data_path, p_name))['results']
        name = self.mat_name_lst[self.name_in_mat]
        X_primed = np.transpose(mat[0, 0][name], (2, 1, 0))      
                        
        up_name = subject_name + '_{}_UPF_cc.mat'.format(self.task)
        mat = scio.loadmat(os.path.join(self.data_path, up_name))['results']
        name = self.mat_name_lst[self.name_in_mat]
        X_unprimed = np.transpose(mat[0, 0][name], (2, 1, 0)) 
        channels, samples = X_unprimed.shape[1], X_unprimed.shape[2]

        # primed=0 unprimed=1 
        y_primed = np.zeros(shape = [X_primed.shape[0],], dtype = int)
        y_unprimed = np.ones(shape = [X_unprimed.shape[0],], dtype = int)
        
        X = np.concatenate((X_primed, X_unprimed), axis=0)
        y = np.concatenate((y_primed, y_unprimed), axis=0)
        shuffle_ix = np.random.permutation(np.arange(len(y)))
        X = X[shuffle_ix, :]
        y = y[shuffle_ix]

        kernels =  1
        X = X.reshape(X.shape[0], kernels, channels, samples)
        X = torch.from_numpy(X).type(torch.float32)
        y = torch.from_numpy(y).type(torch.int64)
        return X[:, :, :, 49:275], y


    ############################# training or testing  ####################################                      
    def training(self):
        '''
        train the models for general purpose
        '''
        self.save_path = '../../results/individual_classification/{}/mvpa/{}/{}'.format(self.training_seed, self.task, self.name_in_mat)

        for self.k in range(10):
            X_train, Y_train = self.training_dataset(self.k, for_training=True)
            X_test, Y_test   = self.training_dataset(self.k, for_training=False)
            dataset_train    = Data.TensorDataset(X_train, Y_train)
            dataset_test     = Data.TensorDataset(X_test, Y_test)
            
            self.train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1024, pin_memory=True)
            self.test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1024, pin_memory=True)
            
            # sliding_windows = [[0,40], [40,58], [58,95], [95,150], [150,188], [188,226]]
            sliding_windows = [[0,40], [40,58], [58,95], [117, 170]]
            # sliding_windows = [[0,30], [30,60], [60,90], [90,120], [120,150], [150,180]]
            for len_idx, window in enumerate(sliding_windows):
                if len_idx > 0:
                    sample = window[1]-window[0]
                    net = network.EEGNet(sample, 4, 2, self.block1_kernel, self.block2_kernel, 0.3).to(DEVICE)
                    net.apply(self.reset_weights)
                    optimizer = torch.optim.Adam(net.parameters(), lr=0.05, weight_decay=0.001)
                    scheduler = StepLR(optimizer, step_size=40, gamma=0.8)

                    best_acc  = 0         
                    for epoch in range(1, 150):
                        net.train()
                        train_batch_loss, test_batch_loss, test_correct  = 0, 0, 0
                        for _, (data, target) in enumerate(self.train_loader):
                            data, target = data[:, :, :, window[0]:window[1]].to(DEVICE), target.to(DEVICE)
                            optimizer.zero_grad(set_to_none=True)
                            output = net(data)[0]
                            loss = F.cross_entropy(output, target, reduction='sum')
                            loss.backward()
                            optimizer.step()
                            train_batch_loss += float(loss.detach())
                        train_batch_loss /= len(dataset_train)
                        
                        net.eval()
                        with torch.no_grad():
                            for _, (data, target) in enumerate(self.test_loader):
                                data, target = data[:, :, :, window[0]:window[1]].to(DEVICE), target.to(DEVICE)
                                output = net(data)[0]
                                test_batch_loss += F.cross_entropy(output, target, reduction='sum').detach()  # sum up batch loss
                                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                                test_correct += pred.eq(target.view_as(pred)).sum().detach()
                            test_acc = test_correct / len(dataset_test)
                            test_batch_loss /= len(dataset_test)
                        scheduler.step()

                        if not os.path.exists(self.save_path):
                            print('creat new model folder:{}'.format(self.save_path))
                            os.makedirs(self.save_path)
                        best_model_path = os.path.join(self.save_path, "{}_best_model_{}.pt".format(len_idx, self.k))

                        if test_acc >= best_acc:
                            epochs_no_improve = 0
                            best_acc = test_acc
                            print('Epoch: {} | Train Loss: {:.6f} | Test Loss: {:.4f} | Test acc: {}/{} ({:.2f}%) '.format(epoch, train_batch_loss, test_batch_loss, test_correct, len(dataset_test), 100. * test_acc))
                            torch.save(net.state_dict(), best_model_path)
                        else:
                            epochs_no_improve += 1
                        torch.cuda.empty_cache()

                        if epoch > 100 and epochs_no_improve >= 20:
                            print(len_idx, best_acc, 'early stopping')
                            break
                        else:
                            continue


    def testing(self):
        ''''
        generate the temporal generalization matrix for all tasks
        '''
        subject_test_result = {}
        permutations = 6

        column_name = ['subjects']
        # sliding_windows = [[0,40], [40,58], [58,95], [95,150], [150,188], [188,226]]
        sliding_windows = [[0,40], [40,58], [58,95], [117, 170]]
        # sliding_windows = [[0,30], [30,60], [60,90], [90,120], [120,150], [150,180]]
        for testing_seed in range(permutations):
            self.split_subject_name(seed=testing_seed)
            self.save_path = '../../results/individual_classification/{}/mvpa/{}/{}'.format(testing_seed, self.task, self.name_in_mat)
            
            for self.k in range(10):
                
                group = self.groups[self.k]
                for subject_name in group:
                    X_test, Y_test = self.testing_dataset(subject_name)
                    dataset_test = Data.TensorDataset(X_test, Y_test)
                    self.test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=256, pin_memory=True)
                    
                    sliding_acc    = []
                    for model_idx, window in enumerate(sliding_windows):
                        if model_idx > 0:
                            sample = window[1]-window[0]
                            net = network.EEGNet(sample, 4, 2, self.block1_kernel, self.block2_kernel, 0.3).to(DEVICE)

                            best_model_path = os.path.join(self.save_path, "{}_best_model_{}.pt".format(model_idx, self.k))
                            loadings = torch.load(best_model_path, map_location=DEVICE)                      
                            net.load_state_dict(loadings)       

                            net.eval()
                            with torch.no_grad():
                                acc_prob_lst, batch_acc = [], 0
                                for _, (data, target) in enumerate(self.test_loader):
                                    data, target = data[:, :, :, window[0]:window[1]].to(DEVICE), target.to(DEVICE)
                                    output = net(data)[0]
                                    output = F.softmax(output, dim=1)
                                    _, max_idx_class = output.max(dim=1)  
                                                            
                                    for idx in range(output.shape[0]):
                                        acc_corr_prob = output[idx, target[idx]]
                                        acc_prob_lst.append(acc_corr_prob.item())
                                    batch_acc += (max_idx_class == target).sum().item()
                                acc = batch_acc/len(dataset_test)
                                # acc_prob = np.mean(acc_prob_lst)
                                sliding_acc.append(acc)  
                    if subject_name in subject_test_result:
                        old = subject_test_result[subject_name]
                        old_lst = list(old)
                        old_lst.extend(sliding_acc)
                        subject_test_result[subject_name] = np.array(old_lst)
                    else:
                        subject_test_result[subject_name] = np.array(sliding_acc)

        sorted_items = sorted(subject_test_result.items())
        sorted_dict = {key: value for key, value in sorted_items}
        
        data = {'Name': np.array(sorted_dict.keys())}
        print(data)
        acc_matrix = np.zeros((len(sorted_dict.keys()), len(sliding_acc)*permutations))
        for i, value in enumerate(sorted_dict.values()):
            acc_matrix[i] = value 
        for j in range(len(sliding_acc)*permutations):
            data[j] = acc_matrix[:, j]
        df = pd.DataFrame(data)
        print(df)
        
        for testing_seed in range(permutations):
            for model_idx, window in enumerate(sliding_windows):
                if model_idx > 0:
                    column_name.append('t{}{}'.format(model_idx, testing_seed))
        df.columns = column_name

        excel_file_path = 'mvpa_{}_{}.xlsx'.format(self.task, self.name_in_mat)
        df.to_excel(excel_file_path, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Facc or Fsp')
    parser.add_argument('name_in_mat', help='the data name in mat file')
    parser.add_argument('training_seed', help='training seed')
    args = parser.parse_args()
    print(args.name_in_mat)
    mvpa_models = MvpaParallel(task=args.task, name_in_mat=args.name_in_mat, training_seed=args.training_seed)
    # mvpa_models.training()
    mvpa_models.testing()
