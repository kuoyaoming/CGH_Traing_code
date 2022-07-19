import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import numpy as np
from numpy import savetxt

def save_spectrum_to_npy(wavfile):
    import librosa
    y, _ = librosa.load(wavfile)
    S = np.abs(librosa.stft(y, n_fft=512))
    p = librosa.amplitude_to_db(S, ref=np.max)
    tmp = np.zeros([256, 128])
    if p.shape[1] > 128:
        # print(p.shape[1])
        tmp[:256, :128] = p[:256, :128]
    else:
        tmp[:256, :p.shape[1]] = p[:256, :p.shape[1]]
    tmp = (tmp+40)
    tmp = tmp/40.0
    tmp = np.float32(tmp)
    return tmp


interpreter = tf.lite.Interpreter(model_path="/home/power703/work/cgh/tflite/chu_5_0605.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = save_spectrum_to_npy('/home/power703/work/cgh/data/chu_y_n/ori/0/1_01_3_k3_3n.wav')
savetxt('data1.csv', input_data, delimiter=',')
input_data = input_data.reshape(1, 256, 128, 1)

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = save_spectrum_to_npy('/home/power703/work/cgh/data/chu_y_n/ori/1/16_01_1_k3_1n.wav')
savetxt('data2.csv', input_data, delimiter=',')
input_data = input_data.reshape(1, 256, 128, 1)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)