import os
import re
import sys
import librosa
import numpy as np

NPY_PATH = 'C:/TWCC/data/data_414_new_AD/test/'
DATA_PATH = 'C:/TWCC/data/data_414_new_AD/test/'
labels = [
    # "class1",
    # "class2",
    # "class3",
    "class4",
    # "class5",
    "class6",
    "class7",
    # "class8",
    "class9",
    # "class10",
    # "class11",
    # "class12"
    # "class13",
    # "class14",
    # "class15",
    # "class16"
]


# def wav2mfcc(file_path):
#     wave, sr = librosa.load(file_path, mono=True, sr=None)
#     mfcc = librosa.feature.mfcc(wave, sr, n_mfcc=20)
#     tmp = np.zeros([20, 420])
#     tmp[:mfcc.shape[0], :mfcc.shape[1]] = mfcc
#     return tmp


def wav2sftp(file_path):
    y, sr = librosa.load(file_path)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    tmp = np.zeros([1025, 250])
    tmp[:D.shape[0], :D.shape[1]] = D
    return tmp


def save_data_to_array(path=DATA_PATH):
    print('Saving npy files')
    for label in labels:
        sftp_vectors = []
        wavfiles = [path + label + '/' +
                    wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in wavfiles:
            if("wav" in wavfile):
                sftp = wav2sftp(wavfile)
                sftp_vectors.append(sftp)
        print(len(sftp_vectors))
        np.save(NPY_PATH + label + '.npy', sftp_vectors)


save_data_to_array()
