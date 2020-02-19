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


epochs = 50
batch_size = 1

SAVE_PATH = 'C:\TWCC\weight\lenet_' + \
    datetime.datetime.now().strftime("%m%d_%H%M")+'.h5'
CSV_PATH = 'C:\TWCC\weight\lenet_' + \
    datetime.datetime.now().strftime("%m%d_%H%M")+'.csv'


labels = [
    # "class1",
    "class2",
    # "class3",
    "class4",
    # "class5",
    "class6",
    "class7",
    # "class8",
    "class9"
    # "class10",
    # "class11",
    # "class12"
    # "class13",
    # "class14",
    # "class15",
    # "class16"
]

classes = len(labels)
TRAIN_PATH = 'C:/TWCC/data/data_414_new_AD/train/'
TEST_PATH = 'C:/TWCC/data/data_414_new_AD/test/'


def get_data_set(PATH):
    X = np.load(PATH + labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    for i, label in enumerate(labels[1:]):
        x = np.load(PATH + label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    assert X.shape[0] == len(y)
    return X, y


x_train, y_train = get_data_set(TRAIN_PATH)
x_test, y_test = get_data_set(TEST_PATH)
x_train = x_train.reshape(x_train.shape[0], 1025, 250, 1)
x_test = x_test.reshape(x_test.shape[0], 1025, 250, 1)
y_train_hot = to_categorical(y_train, num_classes=classes)
y_test_hot = to_categorical(y_test, num_classes=classes)


model = Sequential()
model.add(Conv2D(input_shape=(1025, 250, 1), activation='relu', filters=16,
                 kernel_size=4, strides=1, padding='same', data_format='channels_last'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=32, activation='relu', strides=1,
                 kernel_size=4, padding='same'))
model.add(Conv2D(filters=64, activation='relu', strides=1,
                 kernel_size=4, padding='same'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='same',))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64, activation='relu', strides=1,
                 kernel_size=4, padding='same'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
model.add(Conv2D(filters=24, activation='relu', strides=1,
                 kernel_size=1, padding='same'))

model.add(Dropout(0.5))
model.add(Flatten())
# model.add(Dense(1024))
# model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(classes))
model.add(Activation('softmax'))
model.summary()


adam = Adam(lr=0.00009, beta_1=0.9, beta_2=0.999,
            epsilon=1e-08, decay=0.0, amsgrad=False)

model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=10)

mcp = ModelCheckpoint(SAVE_PATH, monitor='val_loss', verbose=2,
                      save_best_only=True, save_weights_only=False, mode='min', period=1)

cl = CSVLogger(CSV_PATH)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')

model.fit(x_train, y_train_hot, epochs=epochs, batch_size=batch_size,
          verbose=2, validation_data=(x_test, y_test_hot), callbacks=[mcp, cl])

#callbacks=[ mcp, cl]
y_pred = model.predict(x_test)
print(y_pred.shape)
print(y_pred)

predict_label = np.argmax(y_pred, axis=1)

print(predict_label.shape)
print(predict_label)
print(pd.crosstab(y_test, predict_label,
                  rownames=['label'], colnames=['predict']))

print("lenet - " + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
