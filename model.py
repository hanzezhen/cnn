import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,  # input height
                      out_channels=6,  # n_filter
                      kernel_size=3,  # filter size
                      stride=1, # filter step
                      padding=2  # con2d出来的图片大小不变
                      ),  # output shape (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )
        # self.conv1 =nn.Conv2d(in_channels=1,  # input height
        #               out_channels=6,  # n_filter
        #               kernel_size=2,  # filter size
        #               stride=1,  # filter step
        #               padding=2  # con2d出来的图片大小不变
        #               ) # output shape (16,28,28)

        # self.conv1 = nn.Conv2d(in_channels=1,  # input height
        #               out_channels=6,  # n_filter
        #               kernel_size=2,  # filter size
        #               stride=1, # filter step
        #               padding=2  # con2d出来的图片大小不变
        #               )  # output shape (16,28,28)
        self.conv2 = nn.Sequential(nn.Conv2d(
            in_channels=6,  # input height
            out_channels=12,  # n_filter
            kernel_size=3,  # filter size
            stride=1,  # filter step
            padding = 2  # con2d出来的图片大小不变
        ),  # output shape (32,7,7)
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.out1 = nn.Linear(6072, 100)

        self.out2 = nn.Linear(100, 2)# 5808


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        # flat (batch_size, 32*7*7)
        output = self.out1(x)
        output2 = self.out2(output)

        return output2

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)