import csv
import os
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import sys
import time
import os
import pathlib
import tensorflow as tf
start_time = time.time()

num_threads = 32
PATH = '/home/power703/work/cgh/data/spl_chin/ori/'
PP = '/home/power703/work/cgh/data/split_10k/same/1/1_03_3_k3_1.WAV'
IN_PATH = []


def save_spectrum_to_npy(file_path):
    import librosa
    # audio_binary = tf.io.read_file(file_path)
    # audio, _ = tf.audio.decode_wav(audio_binary)
    # waveform = tf.squeeze(audio, axis=-1)
    # waveform = tf.cast(waveform, tf.float32)

    num_channels = 3
    window_sizes = [25, 50, 100]
    hop_sizes = [10, 25, 50]
    # centre_sec = 2.5

    # specs = []
    # for i in range(num_channels):
    #     window_length = int(round(window_sizes[i]*sr/1000))
    #     hop_length = int(round(hop_sizes[i]*sr/1000))

    # spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    # spectrogram = tf.abs(spectrogram)
    # print(spectrogram)

   
    y, sr = librosa.load(file_path, mono=True)
    for i in range(num_channels):
        window_length = int(round(window_sizes[i]*sr/1000))
        hop_length = int(round(hop_sizes[i]*sr/1000))

        spec = librosa.feature.melspectrogram(
            y=y, sr=sr,  hop_length=hop_length, win_length=window_length, n_mels=128)
        print(spec.shape)


    # D = np.abs(librosa.stft(y, n_fft=512, hop_length=64))
    # p = librosa.amplitude_to_db(D, ref=np.max)

    # tmp = np.zeros([256, 256])
    # if p.shape[1] < 256:
    #     tmp[:256, :p.shape[1]] = p[256, ]
    # else:
    #     tmp[:256, :256] = p[:256, :256]
    # print(tmp.shape)

    # tmp = (((tmp*-1)/40)-1)
    # print(tmp)
    # tmp = tmp.astype('float32')
    # np.save(wavfile[:-4]+'.npy', tmp)


if __name__ == '__main__':

    # for root2, dirs2, files2 in os.walk(os.path.abspath(PATH)):
    #     for file2 in files2:
    #         if('WAV' in file2):
    #             IN_PATH.append(os.path.join(root2, file2))
    # print('file number: ', len(IN_PATH))
    # print('num_threads: ', num_threads)
    # print('WROKING...')
    # Pool(num_threads).map(save_spectrum_to_npy, IN_PATH)
    # print('DONE...')
    save_spectrum_to_npy(PP)
