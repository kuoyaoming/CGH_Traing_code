from os import walk
from os.path import isfile, join
from os import listdir
import logging

logging.basicConfig(level=logging.DEBUG)


DATASET_PATH = '/work/power703/cgh/data/chu_7_test/'
num_parts = 5
num_class = 2

ori_list = []
aug_list = []
parts_list = []
train_list = []
test_list = []


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


# read ori_list
for (dirpath, dirnames, filenames) in walk(DATASET_PATH+'ori'):
    if len(dirnames) == 0:
        new = []
        for _filenames in filenames:
            new.append(dirpath+'/'+_filenames)
        ori_list.append(new)
# read aug_list
for (dirpath, dirnames, filenames) in walk(DATASET_PATH+'aug'):
    if len(dirnames) == 0:
        new = []
        for _filenames in filenames:
            new.append(dirpath+'/'+_filenames)
        aug_list.append(new)

# chunk ori_list to parts_list
for i in range(num_class):
    ori_list[i] = random.sample(ori_list[i], len(ori_list[i]))
    parts_list.append(ori_list[i], num_parts)

for _num_class in range(num_class):
    train_class = []
    test_class = []
    for _num_parts in range(num_parts):
        train = []
        test = []
        for _parts_list in parts_list:
            if _parts_list != _num_parts:
                train += parts_list[num_class][_parts_list]
            else:
                test = parts_list[num_class][_parts_list]
        train_class.append(train)
        test_class.append(test)
    train_list.append(train_class)
    test_list.append(test_class)
