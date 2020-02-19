import os
import re
import sys
import librosa
import numpy as np
from pydub import AudioSegment, effects

f_normailze = True
f_pitch_shift = True
f_change_speed = True
DATA_PATH = r'E:\\kid_all\\'

labels = ['2','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'
]


def normailze(PATH):
    rawsound = AudioSegment.from_file(PATH)
    normalizedsound = effects.normalize(rawsound)
    normalizedsound.export(PATH, format="wav")


def pitch_shift(PATH):
    y, sr = librosa.load(PATH)
    y1 = librosa.effects.pitch_shift(y, sr, n_steps=2)
    librosa.output.write_wav(PATH[:-4]+'_p.wav', y1, sr)
    y1 = librosa.effects.pitch_shift(y, sr, n_steps=-2)
    librosa.output.write_wav(PATH[:-4]+'_n.wav', y1, sr)


def change_speed(PATH):
    y, sr = librosa.load(PATH)
    y1 = librosa.effects.time_stretch(y, 1.25)
    librosa.output.write_wav(PATH[:-4]+'_f.wav', y1, sr)
    y1 = librosa.effects.time_stretch(y, 0.75)
    librosa.output.write_wav(PATH[:-4]+'_s.wav', y1, sr)


for label in labels:
    wavfiles = [DATA_PATH + label + '/' +
                wavfile for wavfile in os.listdir(DATA_PATH + '/' + label)]
    for wavfile in wavfiles:
        if("wav" in wavfile):
            print(wavfile)
            if f_normailze:
                normailze(wavfile)
            if f_pitch_shift:
                pitch_shift(wavfile)
            if f_change_speed:
                change_speed(wavfile)
