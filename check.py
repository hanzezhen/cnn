import scipy.io as sio

data = sio.loadmat(r'C:\Users\Lenovo\Desktop\hanzezhen\giit\0612-1624CNN-fortest.mat')

num=0
k=0
for i in list(data['total'])[0]:

    if i>3:
        print(num,i)
        k=k+1
    num = num + 1
print('total',k)