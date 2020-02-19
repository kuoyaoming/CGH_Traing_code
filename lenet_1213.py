from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, multi_gpu_model
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Activation, MaxPooling2D, Flatten
from keras import backend as K
import datetime
import matplotlib.pyplot as plt
import numpy as np
import librosa
import wave
import sys
import re
import os


classes = 16
epochs = 500
batch_size = 128
SAVE_PATH = '/home/power703/weight/lenet_' + \
    datetime.datetime.now().strftime("%m%d_%H%M")+'.h5'
DATA_PATH = '/home/power703/data_16c/'
CSV_PATH = '/home/power703/weight/lenet_' + \
    datetime.datetime.now().strftime("%m%d_%H%M")+'.csv'
labels = ["class1", "class2", "class3", "class4", "class5", "class6", "class7", "class8",
          "class9", "class10", "class11", "class12", "class13", "class14", "class15", "class16", ]
NPY_PATH = '/home/power703/data_16c_npy/'


def wav2mfcc(file_path):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    mfcc = librosa.feature.mfcc(wave, sr, n_mfcc=20)
    tmp = np.zeros([20, 420])
    tmp[:mfcc.shape[0], :mfcc.shape[1]] = mfcc
    return tmp


def save_data_to_array(path=DATA_PATH):
    print('Saving npy files')
    for label in labels:
        mfcc_vectors = []
        wavfiles = [path + label + '/' +
                    wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in wavfiles:
            if("wav" in wavfile):
                mfcc = wav2mfcc(wavfile)
                mfcc_vectors.append(mfcc)
        print(label + ': ' + len(mfcc_vectors) + " files. done!")
        np.save(NPY_PATH + label + '.npy', mfcc_vectors)


def get_train_test(split_ratio=0.8, random_state=42):
    print(labels)
    X = np.load(NPY_PATH + labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    for i, label in enumerate(labels[1:]):
        x = np.load(NPY_PATH + label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    assert X.shape[0] == len(y)
    return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state, shuffle=True)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


X_train, X_test, y_train, y_test = get_train_test()
X_train = X_train.reshape(X_train.shape[0], 20, 420, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 420, 1)
y_train_hot = to_categorical(y_train, num_classes=classes)
y_test_hot = to_categorical(y_test, num_classes=classes)


model = Sequential()
model.add(Conv2D(input_shape=(20, 420, 1), activation='relu', filters=16,
                 kernel_size=(5, 2), strides=1, padding='same', data_format='channels_last'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=32, activation='relu', strides=1,
                 kernel_size=(5, 2), padding='same'))
model.add(Conv2D(filters=64, activation='relu', strides=1,
                 kernel_size=(5, 2), padding='same'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='same',))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64, activation='relu', strides=1,
                 kernel_size=(5, 2), padding='same'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(classes))
model.add(Activation('softmax'))
#model = multi_gpu_model(model, gpus=8)
model.summary()


adam = Adam(lr=0.00009, beta_1=0.9, beta_2=0.999,
            epsilon=1e-08, decay=0.0, amsgrad=False)

model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['acc', f1_m, precision_m, recall_m])

es = EarlyStopping(monitor='val_loss', mode='min',
                   verbose=1, patience=5)

mcp = ModelCheckpoint(SAVE_PATH, monitor='val_loss', verbose=1,
                      save_best_only=True, save_weights_only=False, mode='min', period=1)

cl = CSVLogger(CSV_PATH)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')

model.fit(X_train, y_train_hot, epochs=epochs, batch_size=batch_size,
          verbose=2, validation_data=(X_test, y_test_hot), callbacks=[es, mcp, cl])

print("lenet - " + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
