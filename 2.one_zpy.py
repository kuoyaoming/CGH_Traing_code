import os
import numpy as np
from multiprocessing import Pool

dataset = 'chu_7_t'
ROOT = '/work/power703/cgh/data/'
train_path = ROOT+dataset+'/train/'
test_path = ROOT+dataset+'/test/'
classes = []
save_path = ROOT+dataset+'/'

for n, _, _ in os.walk(os.path.abspath(train_path)):
    classes.append(n.split('/')[-1])
    
classes = classes[1:]
classes = [int(x) for x in classes]
classes.sort()
classes = [str(x) for x in classes]
print(classes)


def load_data(npy_name):
    lable = npy_name.split('/')
    lable = classes.index(lable[-2])
    data = np.load(npy_name[:-4]+'.npy', allow_pickle=True)
    return data, lable



train_list_n = []
test_list_n = []

for root1, _, files1 in os.walk(os.path.abspath(train_path)):
    for file1 in files1:
        if('npy' in file1):
            myname = file1[:file1.rfind('_')]
            myname = myname[:myname.rfind('_')]
            train_list_n.append(os.path.join(root1, file1))

for root1, _, files1 in os.walk(os.path.abspath(test_path)):
    for file1 in files1:
        if('npy' in file1):
            myname = file1[:file1.rfind('_')]
            myname = myname[:myname.rfind('_')]
            test_list_n.append(os.path.join(root1, file1))

pool = Pool(32)
x_train = []
y_train = []
x_test = []
y_test = []

pool_outputs = pool.map(load_data, train_list_n)
for i in pool_outputs:
    x_train.append(i[0])
    y_train.append(i[1])
    # x_test.append(i[0])
    # y_test.append(i[1])

pool_outputs = pool.map(load_data, test_list_n)
for i in pool_outputs:
    x_test.append(i[0])
    y_test.append(i[1])

print('x_train len: ', len(x_train))
print('x_test len: ', len(x_test))

np.savez(
    save_path+'part_2',
    x_train=np.asarray(x_train),
    y_train=np.asarray(y_train),
    x_test=np.asarray(x_test),
    y_test=np.asarray(y_test)
)
