import matplotlib.pyplot as plt
import librosa
import numpy as np
import librosa.display

y, sr = librosa.load('15_22_1_61.wav')
S, phase = librosa.magphase(librosa.stft(y))
rms = librosa.feature.rms(S=S)
plt.figure()
# plt.subplot(2, 1, 1)
# plt.semilogy(rms.T, label='RMS Energy')
# plt.xticks([])
# plt.xlim([0, rms.shape[-1]])
# plt.legend()
# plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(
    S, ref=np.max), y_axis='log', x_axis='time')
plt.title('log Power spectrogram')
plt.tight_layout()
S = librosa.magphase(librosa.stft(y, window=np.ones, center=False))[0]
librosa.feature.rms(S=S)
plt.show()

print(librosa.feature.rms(S=S))