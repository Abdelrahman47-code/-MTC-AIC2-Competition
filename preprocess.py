import os
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import librosa
import numpy as np
from mltu.preprocessors import WavReader

# Functions for preprocessing
def load_audio(file_path, sr=16000):
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return audio, sample_rate

def normalize_audio(audio):
    return librosa.util.normalize(audio)

def trim_silence(audio, top_db=20):
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio

def remove_noise(audio, sr=16000):
    # Placeholder for remove_noise function
    return audio

def preprocess_audio(file_path, sr=16000):
    audio, sample_rate = load_audio(file_path, sr=sr)
    audio = normalize_audio(audio)
    audio = trim_silence(audio)
    audio = remove_noise(audio, sr=sr)
    return audio, sample_rate

def preprocess_dataset(dataset_folder, output_folder, sr=16000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    counter = 0
    
    for file_name in os.listdir(dataset_folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(dataset_folder, file_name)
            preprocessed_audio, sample_rate = preprocess_audio(file_path, sr)
            output_path = os.path.join(output_folder, file_name)
            sf.write(output_path, preprocessed_audio, sample_rate)
            counter += 1
    print(f'All {counter} Files Processed Successfully')

if __name__ == "__main__":
    # Preprocess the dataset
    chunks = ["mtc-asr-train-chunk1", "mtc-asr-train-chunk2", "mtc-asr-train-chunk3", "mtc-asr-train-chunk4"]
    output_folder = '/kaggle/working/output/train_audios'
    for chunk in chunks:
        dataset_folder = os.path.join('/kaggle/input', chunk, 'audios')
        preprocess_dataset(dataset_folder, output_folder)
