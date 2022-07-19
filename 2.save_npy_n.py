import os
import numpy as np
from multiprocessing import Pool
import joblib
import librosa
from skimage.transform import resize

num_threads = 4
PATH = '/home/power703/work/cgh/data/test'
IN_PATH = []

def save_3ch_spectrum(wavfile):
    fix_n_sec = 1
    num_channels = 3
    window_sizes = [25, 50, 100]
    hop_sizes = [10, 25, 50]
    y, sr = librosa.load(wavfile)

    # pad if len less 1 sec
#    if len(y)/sr < fix_n_sec:
#        pad_len = (fix_n_sec * sr) - len(y)
#        y = np.pad(y, (0, pad_len), 'constant', constant_values=(0))
#    else:
#        y = y[:fix_n_sec * sr]

    # audio normalizedy
    normalizedy = librosa.util.normalize(y)
    print(len(y))
    specs = []
    for i in range(num_channels):

        window_length = int(round(window_sizes[i]*sr/1000))
        hop_length = int(round(hop_sizes[i]*sr/1000))

        mel = librosa.feature.melspectrogram(
            y=normalizedy, sr=sr, n_fft=sr, hop_length=hop_length, win_length=window_length)

        mellog = np.log(mel + 1e-9)

        # normalize to (-1,1)
        spec = librosa.util.normalize(mellog)

        spec = resize(spec, (128, 256))
        spec = np.asarray(spec)
        specs.append(spec)

    # list to np array
    specs = np.asarray(specs)
    np.save(wavfile[:-4]+'.npy', specs)

for root2, dirs2, files2 in os.walk(os.path.abspath(PATH)):
    for file2 in files2:
        if('wav' in file2):
            IN_PATH.append(os.path.join(root2, file2))

print(len(IN_PATH))
n_jobs=num_threads
verbose=1
jobs = [ joblib.delayed(save_3ch_spectrum)(i) for i in IN_PATH[:10] ]
out = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)

print(out)
