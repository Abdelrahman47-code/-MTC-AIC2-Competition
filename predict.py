import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from mltu.preprocessors import WavReader
from config import Configs

if __name__ == "__main__":
    # Prediction
    test_metadata = pd.read_csv("/kaggle/input/mtc-asr-test/test.csv")
    test_metadata.columns = ["audio"]
    test_dataset = [[os.path.join('/kaggle/working/output/test_audios', f"{file}.wav"), file] for file in test_metadata["audio"].tolist()]

    submission = []
    for file_path, file_name in tqdm(test_dataset):
        try:
            prediction = predict(file_path, model)
            prediction = "".join([c for c in prediction if c in configs.vocab])
            submission.append([file_name, prediction])
        except:
            submission.append([file_name, ""])

    submission_df = pd.DataFrame(submission, columns=["audio", "transcription"])
    submission_df.to_csv("/kaggle/working/submission.csv", index=False)
