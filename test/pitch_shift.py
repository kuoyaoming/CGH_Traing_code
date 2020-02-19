import librosa

y, sr = librosa.load('13_06_1_k1.wav')
y_third = librosa.effects.pitch_shift(y, sr, n_steps=-2)
librosa.output.write_wav('pitch_shift_low.wav', y_third, sr)
