import numpy as np
import tensorflow as tf
import pandas as pd

labels = ["class1", "class2", "class3", "class4", "class5", "class6",
          "class7", "class8", "class9", "class12", "class13", "class14", "class15"]

classes = len(labels)

TEST_PATH = '/home/power703/data/bace2_test_npy/'


def get_data_set(PATH):
    X = np.load(PATH + labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    for i, label in enumerate(labels[1:]):
        x = np.load(PATH + label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    assert X.shape[0] == len(y)
    return X, y


def tflite_inference(input_data):
    tmp = np.zeros([1, 20, 420, 1])
    ans = np.zeros([input_data.shape[0], 13])
    interpreter = tf.lite.Interpreter(model_path='test.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(input_data.shape[0]):
        tmp[0] = input_data[i]
        input_data_32 = np.array(tmp, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data_32)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        ans[i] = output_data
        print('.', end='')

    return ans


x_test, y_test = get_data_set(TEST_PATH)
x_test = x_test.reshape(x_test.shape[0], 20, 420, 1)

tmp = np.zeros([1, 20, 420, 1])
tmp[0] = x_test[0]
# ans = tflite_inference(tmp)
# print(ans)

ans = tflite_inference(x_test)
print(ans.shape)
print(ans)
predict_label = np.argmax(ans, axis=1)

print (y_test.shape)
print (predict_label.shape)
print(predict_label)
print(pd.crosstab(y_test, predict_label,rownames=['label'], colnames=['predict']))
