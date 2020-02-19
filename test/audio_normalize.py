from pydub import AudioSegment, effects  

rawsound = AudioSegment.from_file("./13_06_1_k1.wav", "wav")  
normalizedsound = effects.normalize(rawsound)  
normalizedsound.export("./13_06_1_k1_1.wav", format="wav")