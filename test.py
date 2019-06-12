from datapre import select,myData
from torch.utils.data import DataLoader
from model import CNN,weights_init
import torch
import torch.nn as nn
from train import train_model
import model
from datapre import select,myData
import datetime
from tensorboardX import SummaryWriter
import os
import scipy.io as sio
import numpy as np

def lo(train_data,the_model,criterion):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nowtim = datetime.datetime.now().strftime('%m%d-%H%M')
    fl = nowtim + 'CNN-fortest'+'.mat'

    path=os.path.dirname(os.path.realpath(__file__))

    path=os.path.join(path,fl)

    i=0

    data={}
    _loss = []

    for x1, y1 in train_data:
        x1 = torch.unsqueeze(x1, 1).float()
        x1 = x1.to(device)
        y1 = y1.to(device)
        outputs = the_model(x1)
        loss = criterion(outputs, y1)
        _loss.append(loss.item())
        i=i+1

    data['total']=np.array(_loss)

    sio.savemat(path,data)

    return True

if __name__ == '__main__':

    PATH = r'C:\Users\Lenovo\Desktop\hanzezhen\giit\modelsave\0612-1621acc81.t7'

    datapath = r'C:\Users\Lenovo\Desktop\hanzezhen\giit/gan_cnndata_guiyihua.mat'

    the_model = CNN()

    if torch.cuda.is_available():
        the_model = the_model.cuda()

    the_model.load_state_dict(torch.load(PATH)['model_state_dict'])

    test_list = [ x for x in range(719)]

    data = myData(datapath, test_list)

    train_data = DataLoader(dataset=data, batch_size=1, shuffle=False)

    loss_func = nn.CrossEntropyLoss()

    lo(train_data, the_model,loss_func)






