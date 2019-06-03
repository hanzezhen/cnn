#coding=utf-8
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import scipy.io
from tensorboardX import SummaryWriter
import random
import re

def datarand(num,r_seed):
    #num数据样本个数
    datalen = num
    random.seed(r_seed)
    ra = list(range(datalen))
    random.shuffle(ra)
    return ra

def func(listTemp, n):
    l=[]
    num = int(len(listTemp)/n)
    for i in range(0, len(listTemp), num):
        l.append(listTemp[i:i + num])
    return l

class myData(Dataset):

    def __init__(self,datapath,ordernum):
        self.matrix= scipy.io.loadmat(datapath)
        self.ordernum = ordernum
        self.lists = []
        for key in self.matrix.keys():
            f1 = re.findall('(\d+)', key)
            if f1 :
                self.lists.append(key)


    def __getitem__(self,item):
        num = self.ordernum[item]
        str1 = self.lists[num]
        data = torch.from_numpy(self.matrix[str1]).cuda()
        str2= '[0-9]+_(.*)$'
        pattern=re.compile(str2)
        label_str = re.search(pattern,str1).group(1)
        if label_str == 'one': label = 1
        else:label=0
        return data,label

    def __len__(self):
        return len(self.ordernum)


def select(s_piece,i,t_num,r_seed):
    ra = datarand(t_num,r_seed)
    l = func(ra, s_piece)
    re = []
    for line in range(len(l)):
        if line != i-1 :
            re+=l[line]
    return re,l[i-1]


