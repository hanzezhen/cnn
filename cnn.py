from datapre import select,myData
from torch.utils.data import DataLoader
from model import CNN,weights_init
import torch
import torch.nn as nn
from train import train_model

#交叉验证
s_piece = 8

i=2

#文件样本数
t_num = 719

num_epochs = 100

lr = 0.00005

datapath='/home/hanzezhen/文档/cnn/MICNNdata/test/a.mat'

savepath='/home/hanzezhen/文档/giit/modelsave'


train_list,test_list = select(s_piece,i,t_num,123)

data1 = myData(datapath,train_list)
data2 =  myData(datapath,test_list)

train_data = DataLoader(dataset=data1,batch_size=80,shuffle=True)
test_data = DataLoader(dataset=data2,batch_size=1,shuffle=True)

data={
    'train' :  train_data,
    'test'  :  test_data
}

data_len = {
    'train' :  len(train_list),
    'test'  :  len(test_list)
}

cnn = CNN()

if torch.cuda.is_available():cnn = cnn.cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cnn.apply(weights_init)

optimizer = torch.optim.Adam(cnn.parameters(),lr=lr)

loss_func = nn.CrossEntropyLoss()

train_model(cnn,num_epochs,data,device,optimizer,loss_func,data_len,savepath)





