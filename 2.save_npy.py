import csv
import os
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import sys
import time
start_time = time.time()

num_threads = 32
PATH = '/home/power703/work/cgh/data/nt_student/ori/'
IN_PATH = []


def save_spectrum_to_npy(wavfile):
    import librosa
    y, _ = librosa.load(wavfile, mono=True)
    D = np.abs(librosa.stft(y, n_fft=512,hop_length=64))
    p = librosa.amplitude_to_db(D, ref=np.max)

    
    tmp = np.zeros([256, 256])
    if p.shape[1] < 256:
        tmp[:256, :p.shape[1]] = p[256, ]
    else:
        tmp[:256, :256] = p[:256, :256]
    # print(tmp.shape)
    
    # tmp = (((tmp*-1)/40)-1)
    # print(tmp)
    # tmp = tmp.astype('float32')
    np.save(wavfile[:-4]+'.npy', tmp)


if __name__ == '__main__':

    for root2, dirs2, files2 in os.walk(os.path.abspath(PATH)):
        for file2 in files2:
            if('wav' in file2):
                IN_PATH.append(os.path.join(root2, file2))
    print('file number: ', len(IN_PATH))
    print('num_threads: ', num_threads)
    print('WROKING...')
    Pool(num_threads).map(save_spectrum_to_npy, IN_PATH)
    print('DONE...')
