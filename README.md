# 🎵 MusicMind AI
### Deep Learning Based Music Genre Classification

MusicMind AI is a deep learning project that uses a Convolutional Neural Network (CNN) to automatically identify the genre of an audio track.

Upload a WAV or MP3 file and let the model analyze its musical features to predict the most likely genre.

## ✨ Features

- 🎧 Upload WAV and MP3 audio files.
- 🤖 CNN-based music genre prediction.
- 🎼 Classification across 10 music genres.
- 📊 Top 3 predicted genres with confidence scores.
- 🎨 Mel Spectrogram visualization.
- 🖥️ Interactive Streamlit web interface.

## 🎶 Supported Music Genres

1. Blues
2. Classical
3. Country
4. Disco
5. Hip-Hop
6. Jazz
7. Metal
8. Pop
9. Reggae
10. Rock

## 🧠 Technology Stack

- Python
- TensorFlow / Keras
- CNN (Convolutional Neural Network)
- Librosa
- NumPy
- Matplotlib
- Streamlit
- Scikit-learn

## ⚙️ How It Works

1. User uploads an audio file.
2. Audio is loaded and preprocessed using Librosa.
3. A Mel Spectrogram is generated from the audio.
4. The processed features are passed to the trained CNN model.
5. The model predicts the most likely music genre.
6. The application displays the prediction and top 3 matches.

## 📂 Project Structure

```text
MusicMind_AI/
│
├── app.py
├── train_model.py
├── evaluate_model.py
├── extract_features.py
├── download_dataset.py
├── check_dataset.py
├── requirements.txt
│
├── models/
│   ├── musicmind_cnn.keras
│   ├── genres.npy
│   └── confusion_matrix.png
│
├── dataset/
│   ├── audio/
│   └── features.npz
│
└── README.md
```

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd MusicMind_AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

## 📊 Model Performance

The CNN model achieved approximately **66.5% test accuracy** on the held-out test split of the GTZAN dataset.

Performance may vary depending on audio quality, recording conditions and the music track.

## 🔮 Future Scope

- Expand the training dataset with additional independent music recordings.
- Improve model accuracy and generalization.
- Explore advanced CNN architectures.
- Support more music genres.
- Improve audio feature extraction and prediction.

## 📥 Download Trained Model

The trained CNN model is hosted on Google Drive due to its file size.

**Download:** [MusicMind CNN Model](https://drive.google.com/file/d/16VIDkO0XSHsVyLEIeNUoUri8dBE8cJZX/view?usp=sharing)

After downloading, place the `musicmind_cnn.keras` file inside the `models/` folder.

Make sure `genres.npy` is also present in the same folder.



## ⚠️ Disclaimer

Predictions and confidence scores are AI-generated estimates and are not guaranteed to represent the actual music genre.

## 👨‍💻 Project

**MusicMind AI**  
Deep Learning Based Music Genre Classification System

## 👨‍💻 Author

**Sarthak Shukla**

Developed using Python, TensorFlow and Streamlit.