import os

dataset_path = "dataset/audio"

total = 0

for genre in sorted(os.listdir(dataset_path)):
    folder = os.path.join(dataset_path, genre)

    if os.path.isdir(folder):
        count = len([
            f for f in os.listdir(folder)
            if f.endswith(".wav")
        ])
        total += count
        print(f"{genre}: {count} files")

print("-" * 30)
print("Total audio files:", total)