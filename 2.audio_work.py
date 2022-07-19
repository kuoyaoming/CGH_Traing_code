import librosa
import numpy as np
from PIL import Image
from skimage.transform import resize
import time

start = time.time()

fix_n_sec = 1
num_channels = 3
window_sizes = [25, 50, 100]
hop_sizes = [10, 25, 50]


y, sr = librosa.load(
    'C:\\Users\\kuo\\Desktop\\data\\5_14_2_k3_1n.wav')

# pad if len less 1 sec
if len(y)/sr < fix_n_sec:
    pad_len = (fix_n_sec * sr) - len(y)
    y = np.pad(y, (0, pad_len), 'constant', constant_values=(0))
else:
    y = y[:fix_n_sec * sr]

# audio normalizedy
normalizedy = librosa.util.normalize(y)

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

end = time.time()
print(end - start)

# # (c,w,h) to (w,h,c)
# specs = np.moveaxis(specs, 0, 2)

# # (optional) to visulation
# specs = (specs * 128) + 128
# specs = specs.astype(np.uint8)
# im = Image.fromarray(specs)
# im.save("filename.jpeg")
