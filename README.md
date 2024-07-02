# Automatic Speech Recognition (ASR) System for Egyptian Dialects

This repository contains the code and documentation for our Automatic Speech Recognition (ASR) system developed specifically for Egyptian dialects, created as a challenge of the **MTC AIC-2 Competition**. The dataset consists of a collection of Egyptian Arabic speech recordings, segmented into chunks for efficient processing and training. The system preprocesses audio data, trains a deep learning model, and provides transcription capabilities tailored to the nuances of Egyptian Arabic dialects.

![image](https://miro.medium.com/v2/resize:fit:828/format:webp/1*KndD0pTUyTHI8p9-f_u_6g.jpeg)

## System Architecture:
- **Dataset Preparation:** The training dataset, consisting of over 50,000 WAV files, was divided into chunks of 5,000 files each for easier management and uploading to Kaggle.
- **Preprocessing:** Audio files were preprocessed using librosa for loading, normalizing, trimming silence, and reducing noise.
- **Model Training:** Used TensorFlow/Keras for training a speech-to-text model. The model architecture included custom preprocessing layers and a deep learning model optimized for sequence-to-sequence tasks.
- **Training Details:** Training involved data augmentation, such as spectrogram padding and label indexing, to handle varying audio lengths and transcription labels.
- **Evaluation:** Evaluated using Word Error Rate (WER) metric and other custom metrics to assess transcription accuracy.
- **Inference:** Inference scripts were developed to predict transcriptions on a test dataset and generate a submission file.

---

## Methodologies Employed:
- **Data Preprocessing:** Standardized audio data preprocessing steps to enhance model robustness.
- **Model Design:** Designed a deep learning architecture tailored for speech recognition tasks, optimizing for both accuracy and efficiency.
- **Training Strategy:** Employed early stopping, learning rate reduction, and model checkpointing during training to prevent overfitting and improve performance.
- **Reproducibility:** Provided all necessary scripts, configurations, and model checkpoints to ensure reproducibility of results.
- **Supplementary Option:** `arbml/wav2vec2-large-xlsr-53-arabic-egyptian` Used for Arabic (Egyptian dialect) speech recognition  to supplement and augment the overall system's capabilities.
  
---

## Technical Details:
- **Frameworks and Libraries:** Utilized TensorFlow/Keras for deep learning, librosa for audio processing, and various utility libraries for data handling and evaluation.
- **Code Structure:** Organized codebase into modular components for data preprocessing, model training, evaluation, and inference.
- **Folder Structure:** Utilized chunked folder organization for the train dataset to manage large data volumes efficiently.

---

## Team Members:
- **Team Leader:** [Abdelrahman Eldaba](https://www.linkedin.com/in/abdelrahmaneldaba)
- **Team Members:** [Ahmed Selem](https://www.linkedin.com/in/ahmed-selim-230953221?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app), [Mohammed Hamza](https://www.linkedin.com/in/mohammed-hamza-4184b2251?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app), [Osama Abo-Bakr](https://www.linkedin.com/in/osama-abo-bakr-293614259/?lipi=urn%3Ali%3Apage%3Ad_flagship3_people_connections%3BFTWf1FHzSFe6LQtj0fPnpQ%3D%3D), [Sayed Ali](https://www.linkedin.com/in/sayed-ali-482668262/?lipi=urn%3Ali%3Apage%3Ad_flagship3_people_connections%3BFTWf1FHzSFe6LQtj0fPnpQ%3D%3D)

