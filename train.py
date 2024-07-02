import os
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from mltu.preprocessors import WavReader
from mltu.tensorflow.dataProvider import DataProvider
from mltu.transformers import LabelIndexer, LabelPadding, SpectrogramPadding
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import TrainLogger
from mltu.tensorflow.metrics import WERMetric
from tqdm import tqdm
from config import Configs

if __name__ == "__main__":
    # Load and preprocess data
    train_dataset = []
    base_path = "/kaggle/working/output/train_audios"

    chunks = ["mtc-asr-train-chunk1", "mtc-asr-train-chunk2", "mtc-asr-train-chunk3", "mtc-asr-train-chunk4"]
    for chunk in chunks:
        dataset_path = os.path.join('/kaggle/input', chunk)
        chunk_number = chunk.split('-')[-1]
        metadata_path = os.path.join(dataset_path, f"chunk_{chunk_number[-1]}.csv")

        metadata_df = pd.read_csv(metadata_path)
        metadata_df.columns = ["audio", "transcription"]
        metadata_df = metadata_df.dropna(subset=['transcription'])

        train_dataset.extend([[os.path.join(base_path, f"{file}.wav"), label.lower()] for file, label in metadata_df.values.tolist()])

    adaptation_metadata_path = "/kaggle/input/mtc-asr-adapt-auidos/adapt.csv"
    adaptation_metadata_df = pd.read_csv(adaptation_metadata_path)
    adaptation_metadata_df.columns = ["audio", "transcription"]
    adaptation_metadata_df = adaptation_metadata_df.dropna(subset=['transcription'])
    val_dataset = [[os.path.join('/kaggle/working/output/adapt_audios', f"{file}.wav"), label.lower()] for file, label in adaptation_metadata_df.values.tolist()]

    configs = Configs()
    max_text_length, max_spectrogram_length = 0, 0

    for file_path, label in tqdm(train_dataset + val_dataset):
        spectrogram = WavReader.get_spectrogram(file_path, frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length)
        valid_label = [c for c in label if c in configs.vocab]
        max_text_length = max(max_text_length, len(valid_label))
        max_spectrogram_length = max(max_spectrogram_length, spectrogram.shape[0])

    configs.max_spectrogram_length = max_spectrogram_length
    configs.max_text_length = max_text_length
    configs.save()

    try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
    except: pass

    data_provider = DataProvider(
        dataset=train_dataset,
        preprocessor=WavReader(configs.frame_length, configs.frame_step, configs.fft_length),
        transformers=[
            LabelIndexer(configs.vocab),
            LabelPadding(configs.max_text_length),
            SpectrogramPadding(configs.max_spectrogram_length)
        ],
        batch_size=configs.batch_size,
        shuffle=True,
        drop_remainder=True
    )

    val_data_provider = DataProvider(
        dataset=val_dataset,
        preprocessor=WavReader(configs.frame_length, configs.frame_step, configs.fft_length),
        transformers=[
            LabelIndexer(configs.vocab),
            LabelPadding(configs.max_text_length),
            SpectrogramPadding(configs.max_spectrogram_length)
        ],
        batch_size=configs.batch_size,
        shuffle=True,
        drop_remainder=True
    )

    model = train_model(configs.input_shape, len(configs.vocab))
    optimizer = tf.keras.optimizers.Adam(learning_rate=configs.learning_rate)
    model.compile(optimizer=optimizer, loss=CTCloss(), metrics=[WERMetric()])

    callbacks = [
        TrainLogger(configs.model_path),
        EarlyStopping(monitor="val_WER", patience=10, verbose=1, mode="min"),
        ReduceLROnPlateau(monitor="val_WER", factor=0.5, patience=5, verbose=1, mode="min"),
        ModelCheckpoint(os.path.join(configs.model_path, "best_model.h5"), monitor="val_WER", save_best_only=True, mode="min", verbose=1),
        TensorBoard(log_dir=os.path.join(configs.model_path, "logs"))
    ]

    model.fit(
        data_provider,
        validation_data=val_data_provider,
        epochs=configs.train_epochs,
        callbacks=callbacks,
        workers=configs.train_workers
    )

    model.load_weights(os.path.join(configs.model_path, "best_model.h5"))
