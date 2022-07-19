
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import os
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import seaborn as sns
# matplotlib.use('Agg')

NAME = 'DenseNet121'
TEST_PATH = '/work/power703/cgh/data/paper_e1/'
LOAD_PATH = '/work/power703/cgh/data/paper_e1/part_2.npz'
WEIGHT_PATH = '/home/power703/work/cgh/weight/DenseNet121_paper_e1_c4_p4_bs128_data_0314_2054.05-0.82-0.47.hdf5'
# /home/power703/work/cgh/weight/DenseNet201_chu_13_c2_p3_bs32_data_0510_1034.33-0.74-0.86.hdf5
class_weight = {}
INPUT_X=128
INPUT_Y=256
INPUT_Z=3
BATCH_SIZE = 128
# def get_dataset():

#     x = np.load(LOAD_PATH)
#     x_test = x['x_test']
#     y_test = x['y_test']
#     # for i in y_test:
#     #     print(i)
#     x_test = x_test.reshape(x_test.shape[0], INPUT_X, INPUT_Y, 1)
#     # y_test = to_categorical(y_test, num_classes=classes)

#     return(x_test, y_test)


# classes = 4
def get_dataset():

    x = np.load(LOAD_PATH, mmap_mode='r', allow_pickle=True)
    # x_train = x['x_train']
    # y_train = x['y_train']
    x_test = x['x_test']
    y_test = x['y_test']

    x_test.astype('float16')
    y_test.astype('float16')

    # total = len(y_train)
    # unique, counts = np.unique(y_train, return_counts=True)
    # class_weight = dict(zip(unique, counts))
    # for l in class_weight:
    #     w = class_weight[l]
    #     new = (1 / w)*(total)/2.0
    #     print('l: ', l, ' new: ', new)
    #     class_weight.update({l: new})

    # x_train = x_train.reshape(x_train.shape[0], INPUT_X, INPUT_Y, INPUT_Z)
    # y_train = to_categorical(y_train, num_classes=classes)

    x_test = x_test.reshape(x_test.shape[0], INPUT_X, INPUT_Y, INPUT_Z)
    # y_test = to_categorical(y_test, num_classes=classes)

    # SHUFFLE_BUFFER_SIZE = len(x_train)

    # return(
    #     tf.data.Dataset.from_tensor_slices(
    #         (x_train, y_train)).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE),
    #     tf.data.Dataset.from_tensor_slices(
    #         (x_test, y_test)).batch(BATCH_SIZE)
    # )
    return(x_test, y_test)

classes = []
labels = [ 'Velar', 'Stopping','Affricate','Consonant vowel']


for n, _, _ in os.walk(os.path.abspath(TEST_PATH+'/ori/')):
    classes.append(n.split('/')[-1])

classes = classes[1:]
classes = [int(x) for x in classes]
classes.sort()

classes = [str(x) for x in classes]
print(classes)

# classes = ['不送氣','','','','']

x_test, y_test = get_dataset()

model = load_model(WEIGHT_PATH)
# METRICS = [
#     keras.metrics.TruePositives(name='tp'),
#     keras.metrics.FalsePositives(name='fp'),
#     keras.metrics.TrueNegatives(name='tn'),
#     keras.metrics.FalseNegatives(name='fn'),
#     keras.metrics.BinaryAccuracy(name='accuracy'),
#     keras.metrics.Precision(name='precision'),
#     keras.metrics.Recall(name='recall'),
#     keras.metrics.AUC(name='auc'),
#     keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
# ]
# model.compile(
#     optimizer=keras.optimizers.Adam(lr=1e-3),
#     loss=keras.losses.BinaryCrossentropy(),
#     metrics=metrics)


# model.summary()

y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = y_test

# print(y_true)
# print(y_pred)

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(10, 8))
# plt.rcParams.update({'font.size': 24})
print(confusion_mtx)

"""
sns.heatmap(confusion_mtx,
            xticklabels=labels,
            yticklabels=labels,
            annot=True, fmt='g',fontweight="bold")
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title(NAME)
# plt.tight_layout()
plt.savefig(NAME+'.svg')  # , bbox_inches='tight', facecolor='w'



y_true = to_categorical(y_true,len(classes))
y_pred = to_categorical(y_pred,len(classes))

m = tf.keras.metrics.Accuracy()
m.update_state(y_true, y_pred)
print('Accuracy: ',m.result().numpy())

m = tfa.metrics.F1Score(num_classes=len(classes), threshold=0.5)
m.update_state(y_true, y_pred)
print('F1Score',m.result().numpy())

m = tf.keras.metrics.Precision()
m.update_state(y_true, y_pred)
print('Precision',m.result().numpy())

m = tf.keras.metrics.Recall()
m.update_state(y_true, y_pred)
print('Recall',m.result().numpy())

m = tf.keras.metrics.AUC()
m.update_state(y_true, y_pred)
print('AUC',m.result().numpy())

# confusion_mtx===================================================






"""
"""
weighted_results = model.evaluate(
    y_test, verbose=0)

for name, value in zip(model.metrics_names, weighted_results):
  print(name, ': ', value)
print()


# def plot_cm(labels, predictions, p=0.5):
#   cm = confusion_matrix(labels, predictions > p)
#   plt.figure(figsize=(5, 5))
#   sns.heatmap(cm, annot=True, fmt="d")
#   plt.title('Confusion matrix @{:.2f}'.format(p))
#   plt.ylabel('Actual label')
#   plt.xlabel('Predicted label')

#   print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
#   print(
#       'Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
#   print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
#   print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
#   print('Total Fraudulent Transactions: ', np.sum(cm[1]))

# plot_cm(test_labels, test_predictions_weighted)

"""