from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


BATCH_SIZE = 1
NUM_EPOCHS = 50

TEST_PATH = 'C:\TWCC\data\data_414_new_AD\ptrain'

model = load_model('C:/TWCC/weight/InceptionV3_0205_1425.h5')
model.summary()

datagen = ImageDataGenerator()

test_generator = datagen.flow_from_directory(TEST_PATH,
                                             target_size=(299, 299),
                                             class_mode='binary',
                                             batch_size=BATCH_SIZE)


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

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


y_predict = model.predict(
    test_generator, batch_size=None, verbose=0, steps=None)

y_pred = np.argmax(y_predict, axis=1)

labels = (test_generator.class_indices)
label = dict((v, k) for k, v in labels.items())
print(label)


y_true = test_generator.classes
target_names = ['class2', 'class4', 'class6', 'class7', 'class9']
# print("test " + str(month))
# print(classification_report(y_true, y_pred, target_names=target_names))
print("**************************************************************")

plt.figure()
cnf_matrix = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                      title='test confusion matrix')
plt.savefig('test.png', bbox_inches='tight',
            pad_inches=0)
plt.show()
