import librosa

y, sr = librosa.load('13_06_1_k1.wav')
y_fast = librosa.effects.time_stretch(y, 2.0)
librosa.output.write_wav('13_06_1_k1_fast.wav', y_fast, sr)
