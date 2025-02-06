'''
easy task and difficult task are trained together, but separately tested.
'''
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CombinedClassify():
    def __init__(self, name_in_mat):
        self.name_in_mat = name_in_mat
        self.mat_name_lst = {'step0':'original', 'step1':'st_c', 'step2':'st_c_synced', 
                            'step3':'st_c_synced_within_subject', 'step4':'aligned_amp', 
                            's_component':'st_s', 'r_component':'st_r'}
        self.block1_kernel = 200
        self.block2_kernel = 100
        self.data_path = '/data/Facc_Fsp/'


    def choose_save_path(self): 

        sub_folder = 'task_combined'
        self.save_path = '../../results/group_classification/{}/{}'.format(sub_folder, self.name_in_mat)


    def reset_weights(self, m):
        '''
        Try resetting model weights to avoid weight leakage.
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


    ############################# readin training or testing dataset  ###########################                      
    def training_dataset(self):

        files = os.listdir(self.data_path)

        primed_num, unprimed_num = 0, 0
        name = self.mat_name_lst[self.name_in_mat]
        for file in files:
            if file.endswith('.mat'):
                content = file.split('.')[0].split('_')
                condition = content[2]
                mat = scio.loadmat(os.path.join(self.data_path, file))['results']
                data = np.transpose(mat[0, 0][name], (1, 0, 2))  
                if condition == 'PF':
                    primed_num += data.shape[2]                       
                elif condition == 'UPF':
                    unprimed_num += data.shape[2]
                          
        channels, samples = data.shape[0], data.shape[1]

        print("primed number:   {}".format(primed_num))
        print("unprimed number: {}".format(unprimed_num))

        # primed = 0 unprimed = 1
        X_primed = np.empty(shape = [primed_num, channels, samples], dtype = float) 
        X_unprimed = np.empty(shape = [unprimed_num, channels, samples], dtype = float) 
        y_primed = np.zeros(shape = [primed_num,], dtype = int)
        y_unprimed = np.ones(shape = [unprimed_num,], dtype = int)

        primed_chans, unprimed_chans = 0, 0
        for file in files:
            if file.endswith('.mat'):
                content = file.split('.')[0].split('_')
                condition = content[2]
                mat = scio.loadmat(os.path.join(self.data_path, file))['results']
                data = np.transpose(mat[0, 0][name], (1, 0, 2)) 
                if condition == "PF":                           
                    for i in range(data.shape[2]):
                        X_primed[primed_chans] = data[:, :, i]
                        primed_chans += 1                            
                elif condition == 'UPF':                           
                    for i in range(data.shape[2]):
                        X_unprimed[unprimed_chans] = data[:, :, i]
                        unprimed_chans += 1                          
        
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
            

    def testing_dataset(self):

        files = os.listdir(self.data_path)

        primed_num, unprimed_num = 0, 0
        name = self.mat_name_lst[self.name_in_mat]
        for file in files:
            if file.endswith('.mat'):
                content = file.split('.')[0].split('_')
                condition = content[2]
                mat = scio.loadmat(os.path.join(self.data_path, file))['results']
                data = np.transpose(mat[0, 0][name], (1, 0, 2))  
                if condition == 'PF':
                    primed_num += data.shape[2]                       
                elif condition == 'UPF':
                    unprimed_num += data.shape[2]                    
                          
        channels, samples = data.shape[0], data.shape[1]

        print("primed number:   {}".format(primed_num))
        print("unprimed number: {}".format(unprimed_num))

        # primed = 0 unprimed = 1
        X_primed = np.empty(shape = [primed_num, channels, samples], dtype = float) 
        X_unprimed = np.empty(shape = [unprimed_num, channels, samples], dtype = float) 
        y_primed = np.zeros(shape = [primed_num, 2], dtype = int)
        y_unprimed = np.ones(shape = [unprimed_num, 2], dtype = int)

        primed_chans, unprimed_chans = 0, 0
        for file in files:
            if file.endswith('.mat'):
                content = file.split('.')[0].split('_')
                task = content[1]
                condition = content[2]
                mat = scio.loadmat(os.path.join(self.data_path, file))['results']
                data = np.transpose(mat[0, 0][name], (1, 0, 2)) 
                if condition == 'PF':         
                    if task == 'Facc':                  
                        for i in range(data.shape[2]):
                            X_primed[primed_chans] = data[:, :, i]
                            primed_chans += 1    
                    elif task == 'Fsp':
                        for i in range(data.shape[2]):
                            X_primed[primed_chans] = data[:, :, i]
                            y_primed[primed_chans, 1] = 1
                            primed_chans += 1                
                elif condition == 'UPF':        
                    if task == 'Facc':               
                        for i in range(data.shape[2]):
                            X_unprimed[unprimed_chans] = data[:, :, i]
                            y_unprimed[unprimed_chans, 1] = 0
                            unprimed_chans += 1   
                    elif task == 'Fsp':
                        for i in range(data.shape[2]):
                            X_unprimed[unprimed_chans] = data[:, :, i]
                            unprimed_chans += 1                                            
                       
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
    

    ############################# training or testing  ####################################                      
    def training(self):
        '''
        train the models for general purpose
        '''
        self.choose_save_path()
        X, y = self.primed_unprimed()

        kfold = KFold(n_splits=10, shuffle=True)
        dataset = Data.TensorDataset(X, y)

        for self.k, (self.train_ids, self.test_ids) in enumerate(kfold.split(np.arange(len(dataset)))):
            if self.k >= 0:
                train_subsampler = torch.utils.data.SubsetRandomSampler(self.train_ids)
                test_subsampler = torch.utils.data.SubsetRandomSampler(self.test_ids)
                
                self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=2048, sampler=train_subsampler, pin_memory=True)
                self.test_loader = torch.utils.data.DataLoader(dataset, batch_size=2048, sampler=test_subsampler, pin_memory=True)
                
                sample = X.shape[3]
                net = network.EEGNet(sample, 4, 2, self.block1_kernel, self.block2_kernel, 0.2).to(DEVICE)
                net.apply(self.reset_weights)

                optimizer = torch.optim.Adam(net.parameters(), lr=0.05, weight_decay=0.01)
                scheduler = StepLR(optimizer, step_size=50, gamma=0.8)

                best_acc  = 0         
                for epoch in range(1, 300):
                    net.train()
                    train_batch_loss, test_batch_loss, test_correct  = 0, 0, 0
                    for _, (data, target) in enumerate(self.train_loader):
                        data, target = data.to(DEVICE), target.to(DEVICE)
                        optimizer.zero_grad(set_to_none=True)
                        output = net(data)[0]
                        loss = F.cross_entropy(output, target, reduction='sum')
                        loss.backward()
                        optimizer.step()
                        train_batch_loss += float(loss.detach())
                    train_batch_loss /= len(self.train_ids)

                    net.eval()
                    with torch.no_grad():
                        for _, (data, target) in enumerate(self.test_loader):
                            data, target = data.to(DEVICE), target.to(DEVICE)
                            output = net(data)[0]
                            test_batch_loss += F.cross_entropy(output, target, reduction='sum').detach()  # sum up batch loss
                            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                            test_correct += pred.eq(target.view_as(pred)).sum().detach()
                        test_acc = test_correct / len(self.test_ids)
                        test_batch_loss /= len(self.test_ids)
                    scheduler.step()

                    print('Epoch: {} | Train Loss: {:.6f} | Test Loss: {:.4f} | Test acc: {}/{} ({:.2f}%) '.format(epoch, train_batch_loss, test_batch_loss, test_correct, len(self.test_ids), 100. * test_acc))
                    if not os.path.exists(self.save_path):
                        print('creat new model folder:{}'.format(self.save_path))
                        os.makedirs(self.save_path)
                    best_model_path = os.path.join(self.save_path, "best_model_{}.pt".format(self.k))

                    if test_acc >= best_acc:
                        epochs_no_improve = 0
                        best_acc = test_acc
                        torch.save(net.state_dict(), best_model_path)
                        print('save the best model')
                    else:
                        epochs_no_improve += 1
                    torch.cuda.empty_cache()


    def testing(self, task):
        '''
        test the models for general purpose
        '''
        self.choose_save_path()
        X, y = self.testing_dataset()

        kfold = KFold(n_splits=10, shuffle=True)
        dataset = Data.TensorDataset(X, y)

        for self.k, (self.train_ids, self.test_ids) in enumerate(kfold.split(np.arange(len(dataset)))):
            if self.k >= 0:
                test_subsampler = torch.utils.data.SubsetRandomSampler(self.test_ids)
                self.test_loader = torch.utils.data.DataLoader(dataset, batch_size=2048, sampler=test_subsampler, pin_memory=True)
                
                sample = X.shape[3]
                net = network.EEGNet(sample, 4, 2, self.block1_kernel, self.block2_kernel, 0.2).to(DEVICE)

                best_model_path = os.path.join(self.save_path, "best_model_{}.pt".format(self.k))
                loadings = torch.load(best_model_path, map_location=DEVICE)                      
                net.load_state_dict(loadings)       

                net.eval()
                with torch.no_grad():
                    batch_acc, num = 0, 0
                    for _, (mixed_data, two_labels) in enumerate(self.test_loader):
                        if task == 'Facc':
                            target = two_labels[two_labels[:, 1] == 0, 0]
                            data = mixed_data[two_labels[:, 1] == 0]
                            num += len(target)
                        elif task == 'Fsp':
                            target = two_labels[two_labels[:, 1] == 1, 0]
                            data = mixed_data[two_labels[:, 1] == 1]
                            num += len(target)
                        data, target = data.to(DEVICE), target.to(DEVICE)
                        output = net(data)[0]
                        output = F.softmax(output, dim=1)
                        _, max_idx_class = output.max(dim=1)  
                        
                        batch_acc += (max_idx_class == target).sum().item()
                    ge_acc = batch_acc/num
                print(ge_acc)
                np.save('{}/{}_{}.npy'.format(self.save_path, task, self.k), ge_acc)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name_in_mat', help='the data name in mat file')
    args = parser.parse_args()
    print(args.name_in_mat)
    
    general_models = CombinedClassify(name_in_mat=args.name_in_mat)
    general_models.training()
    general_models.testing('Facc')
    general_models.testing('Fsp')


