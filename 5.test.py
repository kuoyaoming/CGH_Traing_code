import librosa
from multiprocessing import Pool
import tensorflow as tf
import numpy as np
import os
import visualkeras
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
LOAD_WEIGHT = '/work/power703/cgh/weight/DenseNet201_chu_7_t_c2_p2_bs128_data_0605_1247.14-0.75-1.28.hdf5'
OUTPUT_MODEL = '/work/power703/cgh/weight/chu_7_c2_p2/'
TFL_PATH = '/work/power703/cgh/tflite/'+'chu_7_c2_p2'+'.tflite'
train_path = '/home/power703/work/cgh/data/chu_7_t/test/'


model = tf.keras.applications.DenseNet121(
    input_shape=(256, 128, 1),
    weights=None,
    classes=2
)
visualkeras.layered_view(model)
# plot_model(model, to_file='model.png')





# model.load_weights(LOAD_WEIGHT)








# for root1, _, files1 in os.walk(os.path.abspath(train_path)):
#     for file1 in files1:
#         if('wav' in file1):
#             myname = file1[:file1.rfind('_')]
#             myname = myname[:myname.rfind('_')]
#             # print(os.path.join(root1, file1))
#             y, _ = librosa.load(os.path.join(root1, file1))
#             S = np.abs(librosa.stft(y, n_fft=512))
#             p = librosa.amplitude_to_db(S, ref=np.max)
#             tmp = np.zeros([256, 128])
#             if p.shape[1] > 128:
#                 # print(p.shape[1])
#                 tmp[:256, :128] = p[:256, :128]
#             else:
#                 tmp[:256, :p.shape[1]] = p[:256, :p.shape[1]]
#             tmp = (tmp+40)
#             tmp = tmp/40.0
#             tmp = tmp.reshape(1, 256, 128, 1)
#             print(file1,model.predict(tmp))

# wav_path = '/home/power703/work/cgh/wordcard01_1.wav'
# y, _ = librosa.load(wav_path)
# S = np.abs(librosa.stft(y, n_fft=512))
# p = librosa.amplitude_to_db(S, ref=np.max)
# tmp = np.zeros([256, 128])
# if p.shape[1] > 128:
#     print(p.shape[1])
#     tmp[:256, :128] = p[:256, :128]
# else:
#     tmp[:256, :p.shape[1]] = p[:256, :p.shape[1]]
# tmp = (tmp+40)
# tmp = tmp/40.0
# tmp = tmp.reshape(1, 256, 128, 1)

# print(model.predict(tmp))
