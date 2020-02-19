
from keras.optimizers import SGD, Adam
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Dense, Dropout, Conv2D, Activation, MaxPooling2D, Flatten
from keras.models import Sequential
import sys
import time
import tensorflow as tf
import keras.backend as K
import itertools
from os import listdir
from os.path import isfile, join, isdir, abspath
from sklearn.metrics import classification_report ,confusion_matrix

def wav2mfcc(file_path):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    mfcc = librosa.feature.mfcc(wave, sr, n_mfcc=20)
    tmp = np.zeros([20, 420])
    tmp[:mfcc.shape[0], :mfcc.shape[1]] = mfcc
    return tmp



def get_labels(path):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.tight_layout()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path='/home/power703/test.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)


# In[6]:


from keras.utils import to_categorical
print('start')
mypath = '/home/power703/data/data_bace_414/test/'

labels, indices, _ = get_labels(mypath)
y_true = []
y_pred = []

classes = ['Aspirated', 'Unaspirated', 'Bilabial', 'Velar', 'Fronting', 'Stopping', 'Affricate',
                'Fricative', 'Gliding','Consonant', 'Complex vowel', 'Consonant-voewl',
                'Prenuclear Glide']

label = ['1', '2', '3', '4', '5', '6', '7', '8',
         '9', '12', '13', '14', '15']

# classes = ['Aspirated', 'Unaspirated', 'Bilabial', 'Velar', 'Fronting', 'Stopping', 'Affricate',
#                 'Fricative', 'Gliding', 'Nasal', 'Lateral', 'Consonant', 'Complex vowel', 'Consonant-voewl',
#                 'Prenuclear Glide', 'Correct']

# label = ['1', '2', '3', '4', '5', '6', '7', '8',
#          '9', '10', '11', '12', '13', '14', '15', '16']

folders = [f for f in listdir(mypath) if isdir(join(mypath, f))]
for i in folders:
    subpath = '/home/power703/data/data_bace_414/test/'+i+'/'
    files = [f for f in listdir(subpath) if isfile(join(subpath, f))]
    real = i.split('class', 2)[1]

    for f in files:
        path = subpath + f
        mfcc = wav2mfcc(path)
        mfcc_reshped = mfcc.reshape(1, 20, 420, 1)
        input_data = np.array(mfcc_reshped, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        out = np.argmax(interpreter.get_tensor(output_details[0]['index']))
        ans = labels[int(out)]
        guess = ans.split('class', 2)[1]
        y_true.append(int(real))    
        y_pred.append(int(guess))
        
    print(i)

plt.figure()
cnf_matrix = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                      title='Confusion matrix of LeNet-5 on simulation dataset')

plt.show()

print(classification_report(y_true, y_pred, target_names=label, digits=3))


# In[ ]:





# In[ ]:





# In[ ]:




