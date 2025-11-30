import streamlit as st
import librosa
import numpy as np
import joblib

model = joblib.load("music_genre_model.pkl")

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=30)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr).T, axis=0)

        return np.hstack((mfccs, chroma, mel, contrast, tonnetz))
    except:
        return None

import streamlit as st
import base64

def set_background_gif(gif_path):
    with open(gif_path, "rb") as f:
        data = f.read()
    encoded_gif = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/gif;base64,{encoded_gif}");
            background-size: 100% 100%;   /* FORCE STRETCH / SQUISH */
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_gif("mybackground.gif")



st.markdown("""
<style>

/* --- Global background and text --- */

html, body, [class*="css"] {
    color: white !important;
    font-family: Arial, sans-serif !important;
}

/* Headings bold + white */
h1, h2, h3, h4, h5, h6 {
    font-weight: bold !important;
    color: white !important;
}

/* --- Buttons --- */
.stButton>button {
    background-color: black !important;
    color: white !important;
    border: 2px solid white !important;
    border-radius: 6px !important;
    padding: 8px 20px !important;
}

.stButton>button:hover {
    background-color: white !important;
    color: black !important;
    border: 2px solid white !important;
}

/* --- Input boxes (text, number, etc.) --- */
input, textarea, select {
    background-color: black !important;
    color: white !important;
    border: 1px solid white !important;
}

/* --- Sidebar black --- */
section[data-testid="stSidebar"] {
    background-color: black !important;
    border-right: 1px solid white !important;
}

</style>
""", unsafe_allow_html=True)


st.title("SonicTag: AI powered Music Genre Classification")
st.write("Upload a 30-second audio file (.wav) and let the ML model predict its genre.")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.write("Extracting audio features...")
    features = extract_features("temp.wav")

    if features is not None:
        prediction = model.predict([features])[0]
        st.success(f"### 🎶 Predicted Genre: **{prediction.capitalize()}**")
    else:
        st.error("Could not extract features. Please upload a valid audio file.")
