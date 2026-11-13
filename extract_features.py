
import os
import numpy as np
import librosa
from scipy.signal import resample

DATASET_PATH = "dataset/audio"
genres = sorted(os.listdir(DATASET_PATH))

X = []
y = []

print("Extracting Mel Spectrogram features...")

for label, genre in enumerate(genres):
    folder = os.path.join(DATASET_PATH, genre)

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".wav"):
            continue

        path = os.path.join(folder, filename)

        audio, sr = librosa.load(
            path, sr=22050, mono=True, duration=30
        )

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=2048,
            hop_length=4096,
            n_mels=128
        )

        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)
        mel_db = (mel_db + 80) / 80

        mel_db = resample(mel_db, 128, axis=1)

        X.append(mel_db)
        y.append(label)

    print(f"{genre} completed")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

os.makedirs("dataset", exist_ok=True)

np.savez_compressed(
    "dataset/features.npz",
    X=X,
    y=y,
    genres=np.array(genres)
)

print("\nFEATURE EXTRACTION COMPLETE!")
print("X shape:", X.shape)
print("y shape:", y.shape)
