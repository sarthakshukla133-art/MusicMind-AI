
from datasets import load_dataset, Audio
import os
import io
import soundfile as sf

print("Loading downloaded GTZAN dataset...")

dataset = load_dataset("silky1708/GTZAN-Genre", split="train")

# Bypass TorchCodec audio decoder
dataset = dataset.cast_column("audio", Audio(decode=False))

os.makedirs("dataset/audio", exist_ok=True)

label_feature = dataset.features["label"]

for i, item in enumerate(dataset):
    audio = item["audio"]
    label = label_feature.int2str(item["label"])

    if audio["bytes"] is not None:
        data, sr = sf.read(io.BytesIO(audio["bytes"]))
    else:
        data, sr = sf.read(audio["path"])

    folder = os.path.join("dataset/audio", label)
    os.makedirs(folder, exist_ok=True)

    sf.write(os.path.join(folder, f"{i}.wav"), data, sr)

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/1000")

print("DATASET READY!")
