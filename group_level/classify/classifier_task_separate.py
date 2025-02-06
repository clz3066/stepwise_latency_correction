'''
easy task and difficult task are trained and tested separately. 
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

class GeneralClassify():
    def __init__(self, task, name_in_mat):

        self.task          = task
        self.name_in_mat   = name_in_mat
        self.mat_name_lst  = {'step0':'original', 'step1':'st_c', 'step2':'st_c_synced', 
                             'step3':'st_c_synced_within_subject', 'step4':'aligned_amp', 
                             's_component':'st_s', 'r_component': 'st_r'}
        self.block1_kernel = 200
        self.block2_kernel = 100
        self.data_path     = '/data/Facc_Fsp'                

    
    def choose_save_path(self): 

        sub_folder = 'task_separate'
        self.save_path = '../../results/group_classification/{}/{}/{}'.format(sub_folder, self.task, self.name_in_mat)


    def seed_torch(self, random_seed=0):
        
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

            
    def reset_weights(self, m):
        '''
        Try resetting model weights to avoid
        weight leakage.
        '''
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


    def primed_unprimed(self):
        '''
        extract data for general purpose, easy and difficult tasks are separated
        '''
    
        self.seed_torch(random_seed=0)
        files = os.listdir(self.data_path)
        random.shuffle(files)

        primed_num, unprimed_num = 0, 0
        name = self.mat_name_lst[self.name_in_mat] 
        for file in files:
            if file.endswith('.mat'):
                content = file.split('.')[0].split('_')
                condition = content[2]
                if content[1] == self.task:
                    mat = scio.loadmat(os.path.join(self.data_path, file))['results']
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

        primed_chans, unprimed_chans = 0, 0
        for file in files:
            if file.endswith('.mat'):
                content = file.split('.')[0].split('_')
                condition = content[2]
                if content[1] == self.task:
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
        
        return X, y
        

    def training(self):
        '''
        train the models for general purpose
        '''
        self.choose_save_path()
        X, y = self.primed_unprimed()

        kfold = KFold(n_splits=10, shuffle=True)
        dataset = Data.TensorDataset(X, y)

        for self.k, (self.train_ids, self.test_ids) in enumerate(kfold.split(np.arange(len(dataset)))):
            if self.k == 0:
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
                for epoch in range(1, 100):
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


    def testing(self):
        '''
        test the models for general purpose
        '''
        self.choose_save_path()
        X, y = self.primed_unprimed()

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
                    batch_acc= 0
                    for _, (data, target) in enumerate(self.test_loader):
                        data, target = data.to(DEVICE), target.to(DEVICE)
                        output = net(data)[0]
                        output = F.softmax(output, dim=1)
                        _, max_idx_class = output.max(dim=1)  
                        batch_acc += (max_idx_class == target).sum().item()
                    ge_acc = batch_acc/len(self.test_ids)
                print(ge_acc)
                np.save('{}/{}.npy'.format(self.save_path, self.k), ge_acc)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='task')
    parser.add_argument('name_in_mat', help='the data name in mat file')
    args = parser.parse_args()
    print(args.name_in_mat)
    
    general_models = GeneralClassify(task=args.task, name_in_mat=args.name_in_mat)
    general_models.training()
    general_models.testing()


