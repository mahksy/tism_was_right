import librosa
import matplotlib.pyplot as plt
filename = librosa.example('nutcracker')

waveform, samplingrate = librosa.load(filename)

tempo, beatframes = librosa.beat.beat_track(y=waveform, sr=samplingrate)

