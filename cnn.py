from datapre import select,myData
from torch.utils.data import DataLoader
from model import CNN,weights_init
import torch
import torch.nn as nn
from train import train_model

if __name__=='__main__':

    #交叉验证
    s_piece = 8

    i=4

    #文件样本数
    t_num = 3100

    num_epochs = 110

    lr = 0.0001

    datapath = r'C:\Users\Lenovo\Desktop\hanzezhen\giit/gan_cnndata_guiyihua.mat'

    savepath = r'C:\Users\Lenovo\Desktop\hanzezhen\giit/modelsave'

    random_seed = 123

    train_list,test_list = select(s_piece,i,t_num,random_seed)

    test_2_list = [ x for x in test_list if x<719]

    data1 = myData(datapath,train_list)
    data2 = myData(datapath,test_2_list)

    train_data = DataLoader(dataset=data1,batch_size=80,shuffle=True)
    test_data = DataLoader(dataset=data2,batch_size=1,shuffle=True)

    data = {
        'train' :  train_data,
        'test'  :  test_data
    }

    data_len = {
        'train' :  len(train_list),
        'test'  :  len(test_2_list)
    }

    cnn = CNN()

    if torch.cuda.is_available():cnn = cnn.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cnn.apply(weights_init)

    optimizer = torch.optim.Adam(cnn.parameters(),lr=lr)

    loss_func = nn.CrossEntropyLoss()

    train_model(cnn,num_epochs,data,device,optimizer,loss_func,data_len,savepath)





