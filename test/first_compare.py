import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import pylab
from PIL import Image
import matplotlib.cm as cm

NAME = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]     #指定受試者
INDEX = '17_3'                                  #指定單詞
CLASS = [3, 4, 5, 6, 7, 9, 12]                  #指定類別
DATA_PATH = 'E:/data_16c_all/'                  #音檔目錄
SAVE_PATH = 'E:/tmp/'                           #暫存目錄
SIZE = 1024                                     #圖片大小

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


x = 0
y = 0
fnames = []
for label in os.listdir(DATA_PATH):
    labels = DATA_PATH+label
    for class_ in CLASS:
        if(('class' + str(class_)) in labels):
            wavfiles = [labels + '/' +
                        wavfile for wavfile in os.listdir(labels)]
            for wavfile in wavfiles:
                for name in NAME:
                    if('_'+(str(name) + '.wav') in wavfile):
                        if(('_' + str(INDEX)+'_') in wavfile):
                            data_out = SAVE_PATH+str(x)+'_'+str(y)+'.png'
                            wav2png(wavfile, data_out)
                            fnames.append(data_out)
                            y += 1
            x += 1
            y = 0

f = plt.figure()
f.patch.set_facecolor('xkcd:black')
for n, fname in enumerate(fnames):
    image = Image.open(fname)
    arr = np.asarray(image)
    # this line outputs images on top of each other
    f.add_subplot(len(CLASS), len(NAME), n+1)
    # f.add_subplot(1, 2, n)  # this line outputs images side-by-side
    plt.axis('off')
    plt.imshow(arr)

plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
plt.show()

# for _ in range(square):
# 	for _ in range(square):
# 		# specify subplot and turn of axis
# 		ax = pyplot.subplot(square, square, ix)
# 		ax.set_xticks([])
# 		ax.set_yticks([])
# 		# plot filter channel in grayscale
# 		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
# 		ix += 1
# # show the figure
# pyplot.show()
