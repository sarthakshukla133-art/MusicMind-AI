
import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import os
import zipfile
import gdown
from scipy.signal import resample


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="MusicMind AI",
    page_icon="🎵",
    layout="centered"
)


# ---------- CUSTOM DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #10101c, #19152b);
    color: white;
}

.block-container {
    max-width: 850px;
    padding-top: 3rem;
}

h1, h2, h3, p, label {
    color: #f5f3ff !important;
}

.subtitle {
    color: #aaa5c4;
    font-size: 16px;
    margin-bottom: 30px;
}

.upload-card {
    background: #211d35;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #393151;
    margin: 20px 0;
}

.result-card {
    background: linear-gradient(135deg, #30234d, #211d35);
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #6346a5;
    text-align: center;
    margin-top: 25px;
}

.genre {
    font-size: 34px;
    font-weight: 800;
    color: #c5a5ff;
    margin: 8px 0;
    text-transform: uppercase;
}

.confidence {
    color: #b9b0d2;
    font-size: 15px;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 15px;
}

.stButton button {
    background: linear-gradient(90deg, #8055d9, #a16bea);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 25px;
    font-weight: 700;
    width: 100%;
}

.stButton button:hover {
    border: 1px solid #c5a5ff;
    color: white;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():

    os.makedirs("models", exist_ok=True)

    model_path = "models/musicmind_cnn.keras"

    # Download model from Google Drive if not available
    if not os.path.exists(model_path):

        with st.spinner("Downloading trained AI model..."):

            gdown.download(
                id="16VIDkO0XSHsVyLEIeNUoUri8dBE8cJZX",
                output=model_path,
                quiet=False
            )

    # Check model file
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:

        raise RuntimeError(
            "Model download failed. Please check Google Drive access."
        )

    # Validate Keras model archive
    if not zipfile.is_zipfile(model_path):

        raise RuntimeError(
            "Downloaded model is incomplete or invalid."
        )

    # Load trained model and genre labels
    model = tf.keras.models.load_model(model_path)

    genres = np.load("genres.npy")

    return model, genres


model, genres = load_model()


# ---------- FEATURE EXTRACTION ----------
def extract_features(audio_bytes):

    y, sr = librosa.load(
        io.BytesIO(audio_bytes),
        sr=22050,
        mono=True,
        duration=30
    )

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=4096,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
        top_db=80
    )

    mel_norm = (mel_db + 80) / 80

    mel_resized = resample(
        mel_norm,
        128,
        axis=1
    )

    return mel_resized.reshape(1, 128, 128, 1), mel_db


# ---------- HEADER ----------
st.markdown(
    "<h1 style='text-align:center;'>🎵 MusicMind AI</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle' style='text-align:center;'>"
    "Discover the genre behind your music."
    "</p>",
    unsafe_allow_html=True
)


# ---------- AUDIO UPLOAD ----------
st.markdown(
    "<div class='section-title'>Upload your music</div>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3"]
)


if uploaded_file:

    audio_bytes = uploaded_file.getvalue()

    st.audio(audio_bytes)

    if st.button("✨ Analyse Music"):

        try:

            with st.spinner("Listening to your music..."):

                features, mel_db = extract_features(audio_bytes)

                predictions = model.predict(
                    features,
                    verbose=0
                )[0]

                top_indices = np.argsort(predictions)[::-1][:3]

                best_index = top_indices[0]

                best_genre = str(genres[best_index]).upper()

                confidence = float(predictions[best_index]) * 100


            # ---------- MAIN RESULT ----------
            st.markdown(f"""
            <div class="result-card">
                <div class="confidence">YOUR MUSIC GENRE</div>
                <div class="genre">{best_genre}</div>
                <div class="confidence">
                    Model confidence: {confidence:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)


            # ---------- TOP 3 MATCHES ----------
            st.markdown(
                "<div class='section-title'>Top Matches</div>",
                unsafe_allow_html=True
            )

            for index in top_indices:

                genre_name = str(genres[index]).upper()

                score = float(predictions[index]) * 100

                st.markdown(
                    f"**{genre_name}** — {score:.2f}%"
                )

                st.progress(
                    min(max(float(predictions[index]), 0.0), 1.0)
                )


            # ---------- SPECTROGRAM ----------
            with st.expander("View Music Spectrogram"):

                fig, ax = plt.subplots(figsize=(10, 3))

                librosa.display.specshow(
                    mel_db,
                    sr=22050,
                    hop_length=4096,
                    x_axis="time",
                    y_axis="mel",
                    ax=ax
                )

                ax.set_title("Mel Spectrogram")

                fig.tight_layout()

                st.pyplot(fig)

                plt.close(fig)


            st.caption(
                "Predictions are AI-generated estimates, "
                "not guaranteed genre labels."
            )


        except Exception as e:

            st.error(f"Something went wrong: {e}")


else:

    st.info("Upload a WAV or MP3 file to get started.")


# ---------- FOOTER ----------
st.markdown(
    "<p style='text-align:center;color:#77718f;"
    "margin-top:50px;font-size:13px;'>"
    "MusicMind AI • Powered by Deep Learning"
    "</p>",
    unsafe_allow_html=True
)
