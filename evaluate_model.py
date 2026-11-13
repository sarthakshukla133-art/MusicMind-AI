
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

data = np.load("dataset/features.npz")

X = data["X"][..., np.newaxis]
y = data["y"]
genres = data["genres"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = tf.keras.models.load_model("models/musicmind_cnn.keras")

predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

print("\nCLASSIFICATION REPORT\n")
print(classification_report(
    y_test, y_pred,
    target_names=genres,
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=genres
)

fig, ax = plt.subplots(figsize=(11, 9))
display.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("MusicMind AI - Confusion Matrix")
plt.tight_layout()

plt.savefig("models/confusion_matrix.png")
plt.show()

print("\nEvaluation completed!")
