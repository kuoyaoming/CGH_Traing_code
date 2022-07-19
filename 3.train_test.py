import sys
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import mixed_precision
import os
import logging

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# logging.basicConfig(level = logging.DEBUG)

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)


classes = 4
BATCH_SIZE = 32
epochs = 20
load_last = False
dataset = 'test1'
model_name = 'EfficientNetB0'
s_n = sys.argv[1]
# s_n='0'
INPUT_X=128
INPUT_Y=64

NAME = model_name+'_'+dataset+'_c'+str(classes)+'_p'+str(s_n)+'_bs'+str(BATCH_SIZE)+'_data_' + \
    datetime.datetime.now().strftime("%m%d_%H%M")
SAVE_PATH = '/work/power703/cgh/weight/'+NAME+'.ckpt'
LOG_PATH = '/work/power703/cgh/logs/' + NAME
WEIGHT_PATH = '/work/power703/cgh/data/' + dataset + '/part_'+s_n+'.npz'

LOAD_WEIGHT = ''
class_weight = {}

logging.info('NAME       :', str(NAME))
logging.info('epochs     :', str(epochs))
logging.info('WEIGHT_PATH:', str(WEIGHT_PATH))
logging.info('SAVE_PATH  :', str(SAVE_PATH))
logging.info('LOG_PATH    :', str(LOG_PATH))
logging.info('classes    :', str(classes))


def get_compiled_model():
    base_model = tf.keras.applications.efficientnet.EfficientNetB0(
        input_shape=(INPUT_X, INPUT_Y, 3),
        weights=None,
        include_top=False
    )
    base_model.trainable = True

    # Create new model on top.
    inputs = tf.keras.Input(shape=(INPUT_X, INPUT_Y, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(4)(x)
    model = tf.keras.Model(inputs, outputs)

    opt = SGD(lr=0.01, momentum=0.9)

    if load_last:
        model.load_weights(LOAD_WEIGHT)
    model.compile(optimizer="adam",  # (lr=0.001)
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=0.0001),  # (lr=0.001)
    #             loss="BinaryCrossentropy",
    #             metrics=['BinaryAccuracy','AUC'])
    return model


def get_dataset():

    x = np.load(WEIGHT_PATH, mmap_mode='r', allow_pickle=True)
    x_train = x['x_train']
    y_train = x['y_train']
    x_test = x['x_test']
    y_test = x['y_test']

    total = len(y_train)
    unique, counts = np.unique(y_train, return_counts=True)
    class_weight = dict(zip(unique, counts))
    for l in class_weight:
        w = class_weight[l]
        new = (1 / w)*(total)/2.0
        logging.info("l:'{0}' new: '{1}'".format(l, new))
        class_weight.update({l: new})

    x_train = x_train.reshape(x_train.shape[0], INPUT_X, INPUT_Y, 3)
    y_train = to_categorical(y_train, num_classes=classes)

    x_test = x_test.reshape(x_test.shape[0], INPUT_X, INPUT_Y, 3)
    y_test = to_categorical(y_test, num_classes=classes)

    SHUFFLE_BUFFER_SIZE = len(x_train)

    return(
        tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE),
        tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)).batch(BATCH_SIZE)
    )


strategy = tf.distribute.MirroredStrategy()
logging.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    model = get_compiled_model()



model.summary()

logging.info('DATA LOADING')
train_dataset, test_dataset = get_dataset()

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_dataset = train_dataset.with_options(options)
test_dataset = test_dataset.with_options(options)

# logging.info('class_weight:', class_weight)


# model.summary()
logging.info('START TRAINING')
model.fit(
    train_dataset,
    epochs=epochs,
    verbose=1,
    callbacks=[
        TensorBoard(log_dir=LOG_PATH),
        ModelCheckpoint(
            filepath=SAVE_PATH,
            monitor='val_loss',
            verbose=1, save_best_only=True
        ),
    ],
    validation_data=test_dataset,
    class_weight=class_weight
)
