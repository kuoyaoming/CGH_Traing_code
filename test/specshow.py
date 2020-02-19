import matplotlib.pyplot as plt
import librosa
import numpy as np
import librosa.display

y, sr = librosa.load('15_22_1_61.wav')
S, phase = librosa.magphase(librosa.stft(y))
rms = librosa.feature.rms(S=S)
plt.figure()=
librosa.display.specshow(librosa.amplitude_to_db(
    S, ref=np.max), y_axis='log', x_axis='time')
plt.title('log Power spectrogram')
plt.tight_layout()
S = librosa.magphase(librosa.stft(y, window=np.ones, center=False))[0]
librosa.feature.rms(S=S)
plt.show()

print(librosa.feature.rms(S=S))
