from numpy import pad
import torch.nn as nn
import torch.nn.functional as F
import torch

class EEGNet(nn.Module):
    def __init__(self, samples, F1=4, D=2, block1_kernel=16, block2_kernel=16, dropout=0.75):
        super(EEGNet, self).__init__()

        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((int(block1_kernel/2-1), int(block1_kernel/2), 0, 0)),
            nn.Conv2d(
                in_channels=1,                      # input shape (1, C, T)
                out_channels=F1,                    # num_filters 
                kernel_size=(1, block1_kernel),     # filter size 
                bias=False
            ),                                      # output shape (F1, C, T)
            nn.BatchNorm2d(F1),                     # output shape (F1, C, T)
            nn.Dropout(dropout)
        )
        
        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=F1,                     # input shape (F1, C, T)
                out_channels=F1*D,                  # num_filters
                kernel_size=(41, 1),                # filter size 
                groups=F1,
                bias=False
            ),                                      # output shape (F1*D, 1, T)
            nn.BatchNorm2d(F1*D),                   # output shape (F1*D, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 2)),                   # output shape (F1*D, 1, T//4)
            nn.Dropout(dropout)                     # output shape (F1*D, 1, T//4)
        )
        
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((int(block2_kernel/2-1), int(block2_kernel/2), 0, 0)),
            nn.Conv2d(
                in_channels=F1*D,                   # input shape (F1*D, 1, T//4)
                out_channels=F1*D,                  # num_filters
                kernel_size=(1, block2_kernel),     # filter size
                groups=F1*D,
                bias=False
            ),                                      # output shape (F1*D, 1, T//4)
            nn.Conv2d(
                in_channels=F1*D,                   # input shape (F1*D, 1, T//4)
                out_channels=F1*D,                  # num_filters
                kernel_size=(1, 1),                 # filter size
                bias=False
            ),                                      # output shape (F1*D, 1, T//4)
            nn.BatchNorm2d(F1*D),                   # output shape (F1*D, 1, T//4)
            nn.ELU(),
            nn.Dropout(dropout)
        )
        self.out = nn.Linear(in_features=(F1*D*(samples//2)), out_features=2)

        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, -100)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.item(), 1)

    def forward(self, x):
        block_1 = self.block_1(x)
        block_2 = self.block_2(block_1)
        block_3 = self.block_3(block_2)
        flatten = block_3.view(block_3.size(0), -1)
        # print(block_1.shape, block_2.shape, block_3.shape, flatten.shape)
        output = self.out(flatten)
        # return output
        return output, x, block_1, block_2, block_3, flatten


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        outputs, _ = self.rnn(x)
        last_time = outputs[:, -1, :]
        out = self.fc(last_time)
        return out