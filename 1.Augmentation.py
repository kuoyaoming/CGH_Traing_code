import os
import librosa
import numpy as np
from multiprocessing import Pool
from pydub import AudioSegment, effects
import subprocess
import soundfile as sf

DATASET = 'test'
DATA_PATH = '/home/power703/work/cgh/data/'+DATASET
num_threads = 64


def mix(PATH):
    PATH_ = PATH
    pitch_shift(PATH_)
    time_shift(PATH_)
    change_speed(PATH_)
    DRC(PATH_)
    increase_db(PATH_)
    Harmonic_Distortion(PATH_)
    addNoise(PATH_)
    # Normalize(PATH_)


def pitch_shift(PATH):
    y, sr = librosa.load(PATH)
    y1 = librosa.effects.pitch_shift(y, sr, n_steps=2)
    sf.write(PATH[:-4]+'_2p_.wav', y1, sr)
    y1 = librosa.effects.pitch_shift(y, sr, n_steps=-2)
    sf.write(PATH[:-4]+'_4p_.wav', y1, sr)


def time_shift(PATH):
    y, sr = librosa.load(PATH)
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
    sf.write(PATH[:-4]+'_ts_.wav', y, sr)


def change_speed(PATH):
    y, sr = librosa.load(PATH)
    y1 = librosa.effects.time_stretch(y, 1.25)
    sf.write(PATH[:-4]+'_f_.wav', y1, sr)
    y1 = librosa.effects.time_stretch(y, 0.75)
    sf.write(PATH[:-4]+'_s_.wav', y1, sr)


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


def Harmonic_Distortion(PATH):
    y, sr = librosa.load(PATH)
    y = np.pi * 2 * y
    for _ in range(5):
        y = np.sin(y)
    sf.write(PATH[:-4]+'_hd_.wav', y, sr)


def addNoise(PATH):
    y, sr = librosa.load(PATH)
    x_watt = y ** 2
    sig_avg_watts = np.mean(abs(x_watt))
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - 25
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise = 0
    noise_volts = np.random.normal(
        mean_noise, np.sqrt(noise_avg_watts), y.shape[0])
    y = y + noise_volts
    sf.write(PATH[:-4]+'_n_.wav', y, sr)


def Normalize(PATH):
    # rawsound = AudioSegment.from_file(PATH, "wav")
    # normalizedsound = effects.normalize(rawsound)
    # normalizedsound.export(PATH, format="wav")\
    PATH_ = PATH[:-4]+'n.wav'
    subprocess.call([
        'ffmpeg',
        '-i', PATH,
        '-filter:a', 'loudnorm',
        PATH_
    ])


if __name__ == '__main__':
    IN_PATH = []
    for root, dirs, files in os.walk(os.path.abspath(DATA_PATH)):
        # print(root, dirs, files)
        for file in files:
            if("wav" in file):
                IN_PATH.append(os.path.join(root, file))

            # if("WAV" in file):
            #     IN_PATH.append(os.path.join(root, file))
    print('file number: ', len(IN_PATH))
    print('num_threads: ', num_threads)
    print('WROKING...')
    Pool(num_threads).map(mix, IN_PATH)
