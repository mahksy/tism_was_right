import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def compute_spectral_centroid(y, sr):
    # Compute Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    return centroid

def compute_chroma(y, sr):
    # Compute Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return chroma

# Load techno track
techno_path = 'techno_track.mp3'
techno_y, techno_sr = librosa.load(techno_path, duration=120)

# Load stoner rock track
stoner_rock_path = 'stoner_rock_track.mp3'
stoner_rock_y, stoner_rock_sr = librosa.load(stoner_rock_path, duration=120)

# Compute Spectral Centroid for techno track
techno_centroid = compute_spectral_centroid(techno_y, techno_sr)

# Compute Spectral Centroid for stoner rock track
stoner_rock_centroid = compute_spectral_centroid(stoner_rock_y, stoner_rock_sr)

# Compute Chroma Features for techno track
techno_chroma = compute_chroma(techno_y, techno_sr)

# Compute Chroma Features for stoner rock track
stoner_rock_chroma = compute_chroma(stoner_rock_y, stoner_rock_sr)

# Plot Spectral Centroid features
plt.figure(figsize=(16, 12))

plt.subplot(3, 1, 1)
plt.plot(librosa.times_like(techno_centroid), techno_centroid, label='Techno')
plt.plot(librosa.times_like(stoner_rock_centroid), stoner_rock_centroid, label='Stoner Rock')
plt.title('Spectral Centroid Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.legend()

# Plot Chroma Features
plt.subplot(3, 1, 2)
librosa.display.specshow(techno_chroma, y_axis='chroma', x_axis='time', cmap='coolwarm')
plt.colorbar()
plt.title('Chroma Features - Techno')

plt.subplot(3, 1, 3)
librosa.display.specshow(stoner_rock_chroma, y_axis='chroma', x_axis='time', cmap='coolwarm')
plt.colorbar()
plt.title('Chroma Features - Stoner Rock')

plt.tight_layout()
plt.show()
