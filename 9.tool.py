#!/usr/bin/env python
# coding: utf-8

# In[43]:


import os
import numpy as np
from multiprocessing import  Pool
import random


# In[44]:


num_threads=28
k_part=5
ori_data_path='/work/power703/cgh/data/paper_e2/chu_13/ori/'
aug_data_path='/work/power703/cgh/data/paper_e2/chu_13/aug/'
zpy_save_path='/work/power703/cgh/data/paper_e2/chu_13/'
origin_list = []
aug_list = [[] for i in range(k_part)]
train_data=[[] for i in range(k_part)]
test_data=[[] for i in range(k_part)]
classes = []
aug_limit=999999
test_limit=999999


# In[45]:


classes=[]
def load_class():
    for _, dirs, _ in os.walk(os.path.abspath(ori_data_path)):
        break
    dirs = [int(x) for x in dirs]
    dirs.sort()
    dirs = [str(x) for x in dirs]
    return dirs

classes=load_class()
print(classes)


# In[46]:


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


# In[47]:


def load_ori_list():
    print('Load ori list')
    folder_list=[[] for _ in range(k_part)]
    class_list=[]
    for n in classes:
        ori_path = ori_data_path + str(n)+'/'
        ori_n = 0
        tmp_list=[]
        for root2, _, files2 in os.walk(os.path.abspath(ori_path)):
            for file2 in files2:
                if('npy' in file2):
                    tmp_list.append(os.path.join(root2, file2))
                    ori_n = ori_n+1
        print('Class: ', n, ' files: ',ori_n)
        tmp_list = chunkIt(tmp_list, k_part)
        for n in range(k_part):
            print('ori part ',n,': ',len(tmp_list[n]))
            folder_list[n]=np.append(folder_list[n],tmp_list[n])
            # folder_list[n]=folder_list[n]+tmp_list[n]
    return folder_list

origin_list=load_ori_list()


# for n in range(k_part):
#     origin_list[n]=random.sample(origin_list[n], len(origin_list[n]))

# from random import shuffle 

# # foo = []
# # foo.append([1, 2, 3])
# # foo.append([4, 5, 6])
# # foo.append([7, 8, 9])

# shuffle(origin_list)
# for ii, sublist in enumerate(origin_list): 
#     shuffle(origin_list[ii],len(origin_list[ii]))

# print(origin_list)


# In[48]:


def load_aug_list():
    print('Load aug list')
    for i in range(k_part):
        tmp_list = [y[:-4] for y in origin_list[i]]
        tmp_list = [y[y.rfind('/')+1:] for y in tmp_list]
        for root1, _, files1 in os.walk(os.path.abspath(aug_data_path)):
            for file1 in files1:
                if('npy' in file1):
                    myname = file1[:file1.rfind('_')]
                    myname = myname[:myname.rfind('_')]
                    if(myname in tmp_list):
                        aug_list[i].append(os.path.join(root1, file1))
        print('part: ',i,' aug_n: ',len(aug_list[i]))

load_aug_list()


# In[49]:


def balance_aug_list():
    pass


# In[50]:


def balance_test_list():
    pass


# In[51]:


def marge_ori_aug():
    for select_part in range(k_part):
        for other_part in range(k_part):
            if select_part != other_part:
                train_data[select_part] = np.append(train_data[select_part] ,origin_list[other_part] )
                train_data[select_part]= np.append(train_data[select_part] , aug_list[other_part])
            else:
                test_data[select_part] = origin_list[select_part]
                
marge_ori_aug()


# In[52]:


def load_data(npy_name):
    lable = npy_name.split('/')
    lable = classes.index(lable[-2])
    data = np.load(npy_name[:-4]+'.npy', allow_pickle=True)
    return data, lable


# In[53]:


def save_zpy():
    pool = Pool(num_threads)
    for n in range(k_part):
        print('Saving', zpy_save_path+'part_'+str(n)+'.zpy ...')
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        number_list = np.zeros(len(classes))
        tmp_list = []
    #     for data in test_data[n]:

    #         lable=data.split('/')
    #         lable = lable[-2]
    #         lable = classes.index(lable)  # +'_'+lable[1]
    #         if (number_list[lable] < test_limit):  # or True
    #             number_list[lable] = number_list[lable]+1
    #             tmp_list.append(data)
    #     test_data[n] = tmp_list

    #     pool_outputs = pool.map(load_data, train_data[n])
    #     for i in pool_outputs:
    #         x_train.append(i[0])
    #         y_train.append(i[1])

    #     pool_outputs = pool.map(load_data, test_data[n])
    #     for i in pool_outputs:
    #         x_test.append(i[0])
    #         y_test.append(i[1])

    #     print('x_train len: ', len(x_train))
    #     print('x_test len: ',len(x_test))

    #     np.savez(
    #         zpy_save_path+'part_'+str(n),
    #         x_train=np.asarray(x_train),
    #         y_train=np.asarray(y_train),
    #         x_test=np.asarray(x_test),
    #         y_test=np.asarray(y_test)
    #     )
    # print('done')

save_zpy()

