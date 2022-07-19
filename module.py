import librosa
import numpy as np
from pydub import AudioSegment, effects
import subprocess
import random


def load_wav(PATH):
    lable = PATH[PATH.rfind('/')+1:].split('_')
    lable = lable[0]
    y, sr = librosa.load(PATH)

    return y, lable


def load_wav_aug(PATH):
    lable = PATH[PATH.rfind('/')+1:].split('_')
    lable = lable[0]
    y, sr = librosa.load(PATH)

    return y, lable


def mix(y, sr):
    y1 = pitch_shift(y, sr)
    y2 = time_shift(y, sr)
    y3 = change_speed(y, sr)
    y4 = Harmonic_Distortion(y, sr)
    y5 = addNoise(y, sr)

    return


def pitch_shift(y, sr):
    y = librosa.effects.pitch_shift(y, sr, n_steps=(random.random() - 0.5) * 4)
    return y


def time_shift(y, sr):
    shift = np.random.randint(sr * len(y)/sr*0.1)
    direction = np.random.randint(0, 2)
    if direction == 1:
        shift = -shift
    y = np.roll(y, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        y[:shift] = 0
    else:
        y[shift:] = 0
    return y


def change_speed(y, sr):
    y, sr = librosa.load(PATH)
    y = librosa.effects.time_stretch(y, 1 + (random.random() - 0.5) / 2)
    return y


# Dynamic Range Compression
def DRC(PATH):
    PATH_ = PATH[:-4]+'_drc_.wav'
    subprocess.call([
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'warning',
        '-i', PATH,
        '-filter_complex', "compand=attacks=0:points=-80/-900|-45/-15|-27/-9|0/-7|20/-7:gain=5",
        PATH_
    ])


def increase_db(PATH):
    PATH_ = PATH[:-4]+'_10db_.wav'
    subprocess.call([
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'warning',
        '-i', PATH,
        '-filter:a', "volume=10dB",
        PATH_
    ])


def Harmonic_Distortion(y, sr):
    y = np.pi * 2 * y
    for _ in range(5):
        y = np.sin(y)
    return y


def addNoise(y, sr):
    x_watt = y ** 2
    sig_avg_watts = np.mean(abs(x_watt))
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - 25
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise = 0
    noise_volts = np.random.normal(
        mean_noise, np.sqrt(noise_avg_watts), y.shape[0])
    y = y + noise_volts
    return y
