
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns
matplotlib.use('Agg')

NAME = 'Confusion matrix of EfficientNetB0 form nt_student'
TEST_PATH = '/work/power703/cgh/data/nt_student/'
LOAD_PATH = '/work/power703/cgh/data/nt_student/part_0.npz'
WEIGHT_PATH = '/work/power703/cgh/weight/EfficientNetB0_nt_student_c16_p0_bs32_data_0415_1652.33-3.09.hdf5'

INPUT_X=256
INPUT_Y=256

def get_dataset():

    x = np.load(LOAD_PATH)
    x_test = x['x_test']
    y_test = x['y_test']
    # for i in y_test:
    #     print(i)
    x_test = x_test.reshape(x_test.shape[0], INPUT_X, INPUT_Y, 1)
    # y_test = to_categorical(y_test, num_classes=classes)

    return(x_test, y_test)



classes = []
for n, _, _ in os.walk(os.path.abspath(TEST_PATH+'/ori/')):
    classes.append(n.split('/')[-1])
labels = classes[1:]

# classes = ['不送氣','','','','']

x_test, y_test = get_dataset()

model = load_model(WEIGHT_PATH)
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = y_test

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=labels, yticklabels=labels,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.savefig(NAME+'.png', bbox_inches='tight', facecolor='w')
