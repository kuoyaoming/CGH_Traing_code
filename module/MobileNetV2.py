from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping

BATCH_SIZE = 32
NUM_EPOCHS = 50

TRAIN_PATH = r'C:\\TWCC\\data\\data_414_new_AD\\ptrain'
TEST_PATH = r'C:\TWCC\\data\\data_414_new_AD\\ptest'

model = MobileNetV2(weights=None, classes=5)
model.compile(optimizer=Adam(lr=1e-5),
              loss="sparse_categorical_crossentropy", metrics=['acc'])

model.summary()

datagen = ImageDataGenerator()

# load and iterate training dataset
train_batches = datagen.flow_from_directory(TRAIN_PATH,
                                            target_size=(224, 224),
                                            class_mode='binary',
                                            batch_size=BATCH_SIZE)
# load and iterate validation dataset
valid_batches = datagen.flow_from_directory(TEST_PATH,
                                            target_size=(224, 224),
                                            class_mode='binary',
                                            batch_size=BATCH_SIZE)

cl = CSVLogger('log.csv')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=2)

mcp = ModelCheckpoint('test.h5', monitor='val_loss', verbose=2,
                      save_best_only=True, save_weights_only=False, mode='min')


model.fit_generator(train_batches,
                    steps_per_epoch=train_batches.samples,
                    validation_data=valid_batches,
                    validation_steps=valid_batches.samples,
                    epochs=NUM_EPOCHS,
                    callbacks=[cl, es, mcp])
