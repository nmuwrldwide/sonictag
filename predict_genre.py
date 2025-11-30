import joblib
import librosa
import numpy as np
from sonictag import extract_features  

model = joblib.load("music_genre_model.pkl")

def predict_genre(file_path):
    print(f"Processing: {file_path}")
    features = extract_features(file_path)
    if features is None:
        print("Could not extract features. Check if it's a valid audio file.")
        return
    prediction = model.predict([features])[0]
    print(f"Predicted Genre: {prediction}")
    return prediction

predict_genre("test_song2.wav")
