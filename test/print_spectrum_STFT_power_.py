import matplotlib.pyplot as plt
import librosa
import numpy as np
import librosa.display

y, sr = librosa.load('5_12_3_56_s.wav')
plt.figure()
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
# plt.subplot(4, 2, 1)
print(len(D))
print(len(D[0]))
librosa.display.specshow(D)#, y_axis='linear'
# plt.colorbar(format='%+2.0f dB')
# plt.title('Linear-frequency power spectrogram')


# plt.savefig('filename.png', bbox_inches='tight')
# plt.show()
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig("filename.png", bbox_inches = 'tight',
    pad_inches = 0)