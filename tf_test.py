import os
import re
import sys
import wave
import numpy as np
import datetime

import tensorflow as tf
from sklearn.model_selection import train_test_split


classes = 15
epochs = 2
batch_size = 128*8
SAVE_PATH = '/home/power703/weight/lenet_' + \
    datetime.datetime.now().strftime("%m%d_%H%M")+'.h5'
DATA_PATH = '/home/power703/data_16c/'
CSV_PATH = '/home/power703/weight/lenet_' + \
    datetime.datetime.now().strftime("%m%d_%H%M")+'.csv'
labels = ["class1", "class2", "class3", "class4", "class5", "class6", "class7", "class8",
          "class9", "class10", "class11", "class12", "class13", "class14", "class15"]
NPY_PATH = '/home/power703/data_16c_npy/'
OUT_PATH = '/home/power703/weight/lenet_' + \
    datetime.datetime.now().strftime("%m%d_%H%M")+'.tflite'


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


X_train, X_test, y_train, y_test = get_train_test()

X_train = X_train.reshape(X_train.shape[0], 20, 420, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 420, 1)
# y_train = tf.keras.utils.to_categorical(y_train, num_classes=classes)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=classes)


model = tf.keras.models.Sequential([
    tf.keras.layers.Convolution2D(input_shape=(
        20, 420, 1), filters=16, kernel_size=4, strides=1, padding='same'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Convolution2D(
        filters=32, kernel_size=4, strides=1, padding='same'),
    tf.keras.layers.Convolution2D(
        filters=64, kernel_size=4, strides=1, padding='same'),
    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), strides=2, padding='same'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Convolution2D(
        filters=64, kernel_size=4, strides=1, padding='same'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(classes, activation='softmax')
])

model = tf.keras.utils.multi_gpu_model(model, 8)

adam = tf.keras.optimizers.Adam(learning_rate=0.00009, beta_1=0.9, beta_2=0.999,
            epsilon=1e-08, amsgrad=False)

model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=SAVE_PATH, save_weights_only=True)
]

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
          validation_data=(X_test, y_test), verbose=2)

print("lenet tf test")
print('SAVE_PATH: '+SAVE_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(SAVE_PATH)
tflite_model = converter.convert()
open(OUT_PATH, "wb").write(tflite_model)
