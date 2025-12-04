import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


DATASET_PATH = r"path/goes/here"


def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=30)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr).T, axis=0)

        return np.hstack((mfccs, chroma, mel, contrast, tonnetz))
    except Exception as e:
        print("Error extracting features from:", file_path, "Error:", e)
        return None


genres = os.listdir(DATASET_PATH)
features = []
labels = []

if __name__ == "__main__":
    print("Extracting features... (this may take a few minutes)\n")
    genres = os.listdir(DATASET_PATH)
    features = []
    labels = []

    for genre in genres:
        genre_path = os.path.join(DATASET_PATH, genre)
        if not os.path.isdir(genre_path):
            continue

        for file in tqdm(os.listdir(genre_path), desc=f"Processing {genre}"):
            file_path = os.path.join(genre_path, file)
            data = extract_features(file_path)
            if data is not None:
                features.append(data)
                labels.append(genre)

    X = np.array(features)
    y = np.array(labels)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\nModel Accuracy:", round(acc * 100, 2), "%\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(10, 6))
    cm = confusion_matrix(y_test, y_pred, labels=genres)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=genres, yticklabels=genres, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Music Genre Classification Confusion Matrix")
    plt.tight_layout()
    plt.show()

    import joblib
    joblib.dump(model, "music_genre_model.pkl")
    print("Model saved as 'music_genre_model.pkl'")


