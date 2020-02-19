import os
import re
import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path

SAVE_PATH = 'D:/Find_feature'
DATA_PATH = 'D:/Find_feature'

SIZE = 1024


def wav2png(IN_PATH, OUT_PATH):
    y, _ = librosa.load(IN_PATH)
    plt.figure(figsize=(1, 1), dpi=SIZE)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(OUT_PATH, bbox_inches='tight',
                pad_inches=0)
    plt.close()
    print('From ', IN_PATH, ' to ', OUT_PATH)


def save_data_to_array():
    for lable in os.listdir(DATA_PATH):
        # print(lable)
        for wavfile in os.listdir(DATA_PATH+'/'+lable):
            # print(wavfile)
            if("wav" in wavfile):
                D_I = DATA_PATH+'/'+lable+'/'+wavfile
                D_O = SAVE_PATH+'/'+lable+'/' + \
                    wavfile.rsplit('.', -1)[0] + '.png'
                # print('Load:', D_I)
                # print('Save:', D_O)
                wav2png(D_I, D_O)


save_data_to_array()
