import numpy as np
import tensorflow as tf
from tensorflow import keras

FILE_PATH='/home/power703/work/cgh/weight/DenseNet121_paper_e2/chu_1_c2_p4_bs128_data_1228_1117.11-0.24-5.54.hdf5'

import os
print('FILE SIZE: ',os.path.getsize(FILE_PATH))
print('FILE summary: ')
model = keras.models.load_model(FILE_PATH)
model.summary()