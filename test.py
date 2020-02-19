import pandas as pd
import os
import re
import sys
import numpy as np
import datetime
from keras.layers import Dense, Dropout, Conv2D, Activation, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping
from keras.utils import to_categorical, multi_gpu_model

np.set_printoptions(threshold=sys.maxsize)

def wav2mfcc(file_path):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    mfcc = librosa.feature.mfcc(wave, sr, n_mfcc=20)
    tmp = np.zeros([20, 420])
    tmp[:mfcc.shape[0], :mfcc.shape[1]] = mfcc
    return tmp


mfcc = wav2mfcc('/home/power703/data/test_16c/class1/1_1_1_13.wav')

print(mfcc.shape)
print(mfcc)