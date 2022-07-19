
import librosa
import numpy as np
from PIL import Image

wavfile='/home/power703/work/cgh/data/paper_e1/ori/4/4_11_3_1_k31.wav'
fix_n_sec = 1
num_channels = 3
window_sizes = [25, 50, 100]
hop_sizes = [10, 25, 50]
y=None


try:
    y, sr = librosa.load(wavfile)
    y = librosa.util.normalize(y)
except:
    print("error: ",wavfile)

if y.any():

    print('len: ',len(y)/sr)

    # pad if len less 1 sec
    if len(y)/sr > fix_n_sec:
        x = y[:fix_n_sec * sr]
        run=1
    elif len(y)/sr < 1 :
        z=int((1*sr-len(y))/2)-1
        x = np.pad(y,(z,z), 'constant', constant_values=(0, 0))
        run=1

    if run==1:

        print('audio to image : ',wavfile)

        specs = []
        for i in range(num_channels):

            window_length = int(round(window_sizes[i]*sr/1000))
            hop_length = int(round(hop_sizes[i]*sr/1000))

            mel = librosa.feature.melspectrogram(
                y=x, sr=sr,n_fft=window_length,hop_length=hop_length, win_length=window_length)
            spec = np.log(mel + 1e-9)

            spec = librosa.util.normalize(spec)
            print('channel ',i,' shape :',spec.shape)

            spec += 1.0
            spec *= 128.0
            spec = spec.astype('int8')
            
            img = Image.fromarray(spec, 'L')
            img = img.resize((128,128))
            print('save image :', '/home/power703/'+str(i)+'.png')
            img.save('/home/power703/'+str(i)+'.png')

        image = Image.merge("RGB", (Image.open('/home/power703/0.png'), Image.open('/home/power703/1.png'), Image.open('/home/power703/2.png')))
        print('save image : /home/power703/RGB.png')
        image.save('/home/power703/RGB.png')