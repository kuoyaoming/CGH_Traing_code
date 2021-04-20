import sys
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.losses import categorical_crossentropy



classes = 16
BATCH_SIZE = 32
epochs = 50
load_last = False
dataset = 'nt_student'
model_name = 'EfficientNetB0'
# s_n = sys.argv[1]
s_n='0'
INPUT_X=256
INPUT_Y=256

NAME = model_name+'_'+dataset+'_c'+str(classes)+'_p'+str(s_n)+'_bs'+str(BATCH_SIZE)+'_data_' + \
    datetime.datetime.now().strftime("%m%d_%H%M")
SAVE_PATH = '/work/power703/cgh/weight/'
log_dir = '/work/power703/cgh/logs/' + NAME
WEIGHT_PATH = '/work/power703/cgh/data/' + dataset + '/part_'+s_n+'.npz'

LOAD_WEIGHT = '/work/power703/cgh/weight/EfficientNetB2_spl_chu_c5_bs1_data_0413_1158.04-1.88.hdf5'
class_weight = {}

print('NAME:', NAME)
print('epochs:', epochs)
print('WEIGHT_PATH:', WEIGHT_PATH)
print('SAVE_PATH:', SAVE_PATH)
print('log_dir:', log_dir)
print('classes:', classes)


def get_compiled_model():
    model = tf.keras.applications.EfficientNetB0(
        input_shape=(INPUT_X, INPUT_Y, 1),
        weights=None,
        classes=classes
    )
    if load_last:
        model.load_weights(LOAD_WEIGHT)
    model.compile(optimizer=Adam(),  # (lr=0.001)
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
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
        class_weight.update({l: new})

    x_train = x_train.reshape(x_train.shape[0], INPUT_X, INPUT_Y, 1)
    y_train = to_categorical(y_train, num_classes=classes)

    x_test = x_test.reshape(x_test.shape[0], INPUT_X, INPUT_Y, 1)
    y_test = to_categorical(y_test, num_classes=classes)

    SHUFFLE_BUFFER_SIZE = len(x_train)

    return(
        tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE),
        tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)).batch(BATCH_SIZE)
    )


strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    model = get_compiled_model()

train_dataset, test_dataset = get_dataset()

model.fit(
    train_dataset,
    epochs=epochs,
    verbose=1,
    callbacks=[
        TensorBoard(log_dir=log_dir),
        ModelCheckpoint(
            SAVE_PATH + NAME + '.{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor='val_accuracy',
            verbose=1, save_best_only=True
        ),
    ],
    validation_data=test_dataset,
    class_weight=class_weight
)
