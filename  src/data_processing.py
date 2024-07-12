
import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

if __name__ == "__main__":
    # Exemplu de utilizare
    features = extract_features("../data/example_audio.wav")
    print(features.shape)
