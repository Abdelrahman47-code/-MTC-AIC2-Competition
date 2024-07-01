import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from mltu.preprocessors import WavReader
from mltu.tensorflow.dataProvider import DataProvider
from mltu.transformers import SpectrogramPadding
from keras import layers
from keras.models import Model

class Configs:
    # Define configuration parameters here
    frame_length = 400
    frame_step = 160
    fft_length = 512
    batch_size = 16
    learning_rate = 1e-3
    vocab = [
        '،', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش',
        'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ـ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى', 'ي', 'ً', 'ٌ', 'ٍ',
        'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ٓ', 'ٔ', 'ٕ', '٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩', '٪', '٫', '٬',
        '٭', 'ٰ', 'ٱ', 'ٹ', 'پ', 'چ', 'ڈ', 'ڑ', 'ژ', 'ک', 'گ', 'ں', 'ھ', 'ہ', 'ۂ', 'ۃ', 'ۆ', 'ۇ', 'ۈ', 'ۋ', 'ی',
        'ے', '۔', '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹', '۾', 'ۿ', 'ﷺ'
    ]
    input_shape = [None, 257]
    
    @classmethod
    def load(cls, path):
        import json
        with open(path, "r") as f:
            data = json.load(f)
        config = cls()
        config.__dict__.update(data)
        return config

def build_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    inputs = layers.Input(shape=input_dim, name="input", dtype=tf.float32)
    input = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs)

    x = layers.Conv2D(filters=32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False)(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    x = layers.Dense(256)(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout)(x)

    output = layers.Dense(output_dim + 1, activation="softmax", dtype=tf.float32)(x)
    model = Model(inputs=inputs, outputs=output)
    return model

def decode_predictions(preds, idx_to_char):
    pred_texts = []
    for pred in preds:
        pred_text = ""
        for idx in pred:
            if idx == -1:  # Assuming -1 is used for padding
                continue
            pred_text += idx_to_char.get(idx, "")
        pred_texts.append(pred_text)
    return pred_texts

def main():
    # Load configuration
    config_path = "config.json"
    configs = Configs.load(config_path)

    # Define paths
    test_wavs_path = "/kaggle/input/mtc-aic-test-audios/test"

    # Load the test audio files
    test_audio_files = [f for f in os.listdir(test_wavs_path) if f.endswith('.wav')]
    test_dataset = [[os.path.join(test_wavs_path, f), ""] for f in test_audio_files]

    # Calculate the maximum spectrogram length for the test dataset
    max_length = 0
    for file_path, _ in test_dataset:
        wav_reader = WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length)
        spectrogram = wav_reader(file_path)
        if spectrogram.shape[0] > max_length:
            max_length = spectrogram.shape[0]

    # Update the configuration
    configs.max_spectrogram_length = max_length

    # Create a DataProvider for the test data
    test_data_provider = DataProvider(
        dataset=test_dataset,
        skip_validation=True,
        batch_size=configs.batch_size,
        data_preprocessors=[
            WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
        ],
        transformers=[
            SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
        ],
    )

    # Load the trained model with the best weights
    model = build_model(configs.input_shape, len(configs.vocab))
    checkpoint_path = "best_model_checkpoint/checkpoint.weights.h5"
    model.load_weights(checkpoint_path)

    # Compile the model before prediction
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
        loss=CTCloss()
    )

    # Create a reverse dictionary for indexing
    idx_to_char = {i: c for i, c in enumerate(configs.vocab)}

    # Predict the transcripts for the test data
    predictions = []
    for batch in tqdm(test_data_provider):
        preds = model.predict(batch[0])  # batch[0] contains the spectrograms
        decoded_preds = decode_predictions(np.argmax(preds, axis=-1), idx_to_char)
        predictions.extend(decoded_preds)

    # Create the submission dataframe
    submission_df = pd.DataFrame({
        "audio": [os.path.basename(f[0]) for f in test_dataset],
        "transcription": predictions
    })

    # Save the submission dataframe to a CSV file
    submission_df.to_csv("/kaggle/working/submission.csv", index=False)

    print("Submission file created successfully.")

if __name__ == "__main__":
    main()
