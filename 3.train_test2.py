import sys
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow import keras

classes = 2
BATCH_SIZE = 128
epochs = 25
load_last = False
dataset = 'chu_y_n'
model_name = 'InceptionResNetV2'
# s_n = sys.argv[1]
s_n = '2'
INPUT_X = 256
INPUT_Y = 128

NAME = model_name+'_'+dataset+'_c'+str(classes)+'_p'+str(s_n)+'_bs'+str(BATCH_SIZE)+'_data_' + \
    datetime.datetime.now().strftime("%m%d_%H%M")
SAVE_PATH = '/work/power703/cgh/weight/'
log_dir = '/work/power703/cgh/logs/' + NAME
WEIGHT_PATH = '/work/power703/cgh/data/' + dataset + '/part_'+s_n+'.npz'

LOAD_WEIGHT = '/home/power703/work/cgh/weight/DenseNet201_chu_5_c2_p1_bs32_data_0512_2353.54-0.67-2.16.hdf5'
class_weight = {}

print('NAME:', NAME)
print('epochs:', epochs)
print('WEIGHT_PATH:', WEIGHT_PATH)
print('SAVE_PATH:', SAVE_PATH)
print('log_dir:', log_dir)
print('classes:', classes)

def build_model(num_classes):

    inputs = keras.layers.Input(shape=(INPUT_X, INPUT_Y, 3))
    # x = img_augmentation(inputs)
    model = tf.keras.applications.EfficientNetB3(
        include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = keras.layers.Dense(
        num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-5),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=METRICS)
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
        print('l: ', l, ' new: ', new)
        class_weight.update({l: new})

    x_train = x_train.reshape(x_train.shape[0], INPUT_X, INPUT_Y, 1)
    x_train = tf.keras.applications.efficientnet.preprocess_input(x_train)
    y_train = to_categorical(y_train, num_classes=classes)

    x_test = x_test.reshape(x_test.shape[0], INPUT_X, INPUT_Y, 1)
    x_test = tf.keras.applications.efficientnet.preprocess_input(x_test)
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
    model = build_model(2)
    model.summary()


# for i, layer in enumerate(model.layers):
#    print(i, layer.name)

# for layer in model.layers[:238]:
#    layer.trainable = False
# for layer in model.layers[238:]:
#    layer.trainable = True

# model.summary()
print('###################################')
print('DATA LOADING')
train_dataset, test_dataset = get_dataset()

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_dataset = train_dataset.with_options(options)
test_dataset = test_dataset.with_options(options)

# print('class_weight:', class_weight)


# model.summary()
print('###################################')
print('START TRAINING')
model.fit(
    train_dataset,
    epochs=epochs,
    verbose=1,
    callbacks=[
        TensorBoard(log_dir=log_dir),
        ModelCheckpoint(
            SAVE_PATH + NAME +
            '.{epoch:02d}-{val_accuracy:.2f}-{val_loss:.2f}.hdf5',
            monitor='val_accuracy',
            verbose=2, save_best_only=True
        ),
    ],
    validation_data=test_dataset,
    class_weight=class_weight
)
