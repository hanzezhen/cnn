import scipy.io as sio
import matplotlib.pyplot as plt
import time

datapath=r'C:\Users\Lenovo\Desktop\hanzezhen\giit/gan_cnndata_guiyihua.mat'


pic = sio.loadmat(datapath)

for key,value in pic.items():

    try:
        plt.imshow(value)
        plt.show()
    except:print('1')
    time.sleep(0.1)
