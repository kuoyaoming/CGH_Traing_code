import csv
import os
import numpy as np
from multiprocessing import Process, Pool
import sys
import time
from random import shuffle
import random

num_threads = 32
dataset = 'nt_student'
ROOT = '/work/power703/cgh/data/'
save_path = ROOT+dataset+'/'
classes = []
ori_path = ROOT+dataset+'/ori/'
aug_path = ROOT+dataset+'/aug/'

for n, _, _ in os.walk(os.path.abspath(ori_path)):
    classes.append(n.split('/')[-1])

classes = classes[1:]
classes = [int(x) for x in classes]
classes.sort()

classes = [str(x) for x in classes]
print(classes)

numberOfPart = 8

print('*'*100)
print('DataSet: ', dataset)
print('part: ', numberOfPart)
print('class: ', len(classes))
print('*'*100)


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


def load_data(npy_name):
    lable = npy_name[npy_name.rfind('/')+1:].split('_')
    lable = classes.index(lable[0])  # +'_'+lable[1]
    data = np.load(npy_name[:-4]+'.npy')

    return data, lable


train_list = [[] for i in range(numberOfPart)]
test_list = [[] for i in range(numberOfPart)]
origin_list = []

for n in classes:
    ori_path = ROOT+dataset+'/ori/' + str(n)+'/'
    ori_n = 0

    # add all origin file path to list
    for root2, _, files2 in os.walk(os.path.abspath(ori_path)):
        for file2 in files2:
            if('wav' in file2):
                origin_list.append(os.path.join(root2, file2))
                ori_n = ori_n+1
            if('WAV' in file2):
                origin_list.append(os.path.join(root2, file2))
                ori_n = ori_n+1

    print('Class: ', f'{n:5}', ' origin: ',
          f'{ori_n:4}')

origin_list = random.sample(origin_list, len(origin_list))
origin_list = chunkIt(origin_list, numberOfPart)
aug_list = [[] for i in range(numberOfPart)]
# add all origin file path to list
for i in range(len(origin_list)):
    tmp_list = [y[:-4] for y in origin_list[i]]
    tmp_list = [y[y.rfind('/')+1:] for y in tmp_list]
    for root1, _, files1 in os.walk(os.path.abspath(aug_path)):
        for file1 in files1:
            if('wav' in file1):
                myname = file1[:file1.rfind('_')]
                myname = myname[:myname.rfind('_')]
                if(myname in tmp_list):
                    aug_list[i].append(os.path.join(root1, file1))
            if('WAV' in file1):
                myname = file1[:file1.rfind('_')]
                myname = myname[:myname.rfind('_')]
                if(myname in tmp_list):
                    aug_list[i].append(os.path.join(root1, file1))


number_ori = np.zeros(len(classes))

# for i in range(8):
#     print(len(aug_list[i]))

for data in origin_list[0]:
    lable = data[data.rfind('/')+1:].split('_')
    lable = classes.index(lable[0])  # +'_'+lable[1]
    number_ori[lable] = number_ori[lable]+1

# print(number_ori)

# for i in range(8):
#     for j in range(16):


for n in range(numberOfPart):
    tmp_list = []
    print('origin:', len(aug_list[n]))
    aug_list[n] = random.sample(aug_list[n], len(aug_list[n]))
    number_aug = np.zeros(len(classes))
    for data in aug_list[n]:
        lable = data[data.rfind('/')+1:].split('_')
        lable = classes.index(lable[0])  # +'_'+lable[1]
        # print(data, lable)
        if (number_aug[lable] < (500 - number_ori[lable])):  # or True
            number_aug[lable] = number_aug[lable]+1
            # print('paas')
            tmp_list.append(data)
        else:
            # print('kill')
            pass
    aug_list[n] = tmp_list
    print('new:', len(aug_list[n]))


for select_part in range(numberOfPart):
    for other_part in range(numberOfPart):
        if select_part != other_part:
            train_list[select_part] = train_list[select_part] + \
                origin_list[other_part]   + aug_list[other_part]
        else:
            test_list[select_part] = origin_list[select_part]

print('*'*100)
# print(np.median([58,  66, 305, 203, 196,  57,  52,
#                  221,  47,  30,  17, 392,  57,  67, 55, 576]))

pool = Pool(num_threads)
for n in range(numberOfPart):
    print('Saving', save_path+'part_'+str(n)+'.zpy ...')
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    number_list = np.zeros(len(classes))
    tmp_list = []
    for data in test_list[n]:
        lable = data[data.rfind('/')+1:].split('_')
        lable = classes.index(lable[0])  # +'_'+lable[1]
        if (number_list[lable] < 50):  # or True
            number_list[lable] = number_list[lable]+1
            tmp_list.append(data)
    test_list[n] = tmp_list

    pool_outputs = pool.map(load_data, train_list[n])
    for i in pool_outputs:
        x_train.append(i[0])
        y_train.append(i[1])

    pool_outputs = pool.map(load_data, test_list[n])
    for i in pool_outputs:
        x_test.append(i[0])
        y_test.append(i[1])

    print('x_train len: ', len(x_train))
    print('x_test len: ',len(x_test))

    np.savez(
        save_path+'part_'+str(n),
        x_train=np.asarray(x_train),
        y_train=np.asarray(y_train),
        x_test=np.asarray(x_test),
        y_test=np.asarray(y_test)
    )
