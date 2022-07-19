import librosa
import numpy as np
wav_file = '/home/power703/work/cgh/data/1201/ori/new/1/1_01_3_k5.wav'
y, sr = librosa.load(wav_file, sr=16000)
print(len(y)/sr)
a = librosa.feature.melspectrogram(y, sr=sr, n_mels=64, hop_length=160)
print(a.shape)

result = np.zeros([64, 200])
result[:a.shape[0], :a.shape[1]] = a

print(result)
