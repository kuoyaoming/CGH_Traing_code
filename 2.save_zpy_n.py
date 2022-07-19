import os
import numpy as np
from multiprocessing import  Pool
import random

num_threads = 56
dataset = 'paper_e1'
ROOT = '/work/power703/cgh/data/'
save_path = ROOT+dataset+'/'

ori_path = ROOT+dataset+'/ori/'
aug_path = ROOT+dataset+'/aug/'
max_number_of_aug_of_each_class=1328
max_number_of_testset_of_each_class=40
numberOfPart = 5 

origin_list = []
aug_list = [[] for i in range(numberOfPart)]
train_list = [[] for i in range(numberOfPart)]
test_list = [[] for i in range(numberOfPart)]

def get_category():
    classes = []
    for n, _, _ in os.walk(os.path.abspath(ori_path)):
        classes.append(n.split('/')[-1])
    classes = classes[1:]
    classes = [int(x) for x in classes]
    classes.sort()
    classes = [str(x) for x in classes]
    print(classes)

get_category()

print('*'*100)
print('DataSet: ', dataset)
print('part: ', numberOfPart)
print('class: ', len(classes))
print('*'*100)


def randomly_divided(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


def load_data(npy_name):
    lable = npy_name.split('/')
    lable = classes.index(lable[-2])
    data = np.load(npy_name[:-4]+'.npy', allow_pickle=True)
    return data, lable

def load_origin_data_list():
    for n in classes:
        ori_path = ROOT+dataset+'/ori/' + str(n)+'/'
        ori_n = 0

        # add all origin file path to list
        for root2, _, files2 in os.walk(os.path.abspath(ori_path)):
            for file2 in files2:
                if('npy' in file2):
                    origin_list.append(os.path.join(root2, file2))
                    ori_n = ori_n+1

        print('Class: ', f'{classes[n]:5}', ' origin: ',
            f'{ori_n:4}')

    origin_list = random.sample(origin_list, len(origin_list))
    origin_list = randomly_divided(origin_list, numberOfPart)

# add all origin file path to list
for i in range(len(origin_list)):
    print('i: ',i)
    tmp_list = [y[:-4] for y in origin_list[i]]
    tmp_list = [y[y.rfind('/')+1:] for y in tmp_list]
    for root1, _, files1 in os.walk(os.path.abspath(aug_path)):
        for file1 in files1:
            if('npy' in file1):
                myname = file1[:file1.rfind('_')]
                myname = myname[:myname.rfind('_')]
                if(myname in tmp_list):
                    aug_list[i].append(os.path.join(root1, file1))

number_ori = np.zeros(len(classes))

for data in origin_list[0]:
    data=data.split('/')
    lable = data[-2]
    print('lable:',lable)
    lable = classes.index(lable)  # +'_'+lable[1]
    number_ori[lable] = number_ori[lable]+1

for n in range(numberOfPart):
    tmp_list = []
    print('origin:', len(aug_list[n]))
    aug_list[n] = random.sample(aug_list[n], len(aug_list[n]))
    number_aug = np.zeros(len(classes))
    for data in aug_list[n]:
        lable = data[data.rfind('/')+1:].split('_')
        if lable[0] != '16':
            lable = 0
        else:
            lable = 1
        if (number_aug[lable] < (max_number_of_aug_of_each_class - number_ori[lable])):
            number_aug[lable] = number_aug[lable]+1
            tmp_list.append(data)
        else:
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

        lable=data.split('/')
        print(lable)
        lable = lable[-2]
        lable = classes.index(lable)  # +'_'+lable[1]
        if (number_list[lable] < max_number_of_testset_of_each_class):  # or True
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
