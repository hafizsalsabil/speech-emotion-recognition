# 🎙️ Speech Emotion Recognition Using CNN with Mel-Spectrogram

A web-based Speech Emotion Recognition (SER) application that classifies human emotions from speech audio using a Convolutional Neural Network (CNN) and Mel-Spectrogram feature representation.

This project was developed as an undergraduate thesis in the Information Systems Program at Universitas Gunadarma.

---

## 📌 Features

- Upload one or multiple audio files (.wav, .mp3, .ogg, .m4a)
- Automatic audio conversion to WAV format
- Mel-Spectrogram generation
- CNN-based emotion prediction
- Prediction confidence for each emotion
- Download prediction results as CSV
- Responsive web interface using Flask

---

## 🧠 Predicted Emotions

The model classifies speech into four emotions:

- 😠 Anger
- 😀 Happiness
- 😐 Neutral
- 😢 Sadness

---

## 🛠 Technologies

- Python
- Flask
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Bootstrap 5
- HTML
- CSS

---

## 📂 Project Structure

```
SER_Skripsi/
│
├── app.py
├── audio_processing.py
├── prediction.py
├── requirements.txt
├── ser_model_augmented.keras
│
├── docs/
│   ├── MelSpectrogram_Anger.png
│   ├── MelSpectrogram_Happiness.png
│   ├── MelSpectrogram_Neutral.png
│   └── MelSpectrogram_Sadness.png
│
├── notebook/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_audio_augmentation.ipynb
│   ├── 03_cnn_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_error_analysis.ipynb
│   ├── 06_prediction_original_dataset.ipynb
│   ├── 07_dataset_audit.ipynb
│   └── 08_generate_melspectrogram.ipynb
│
├── static/
│   ├── css/
│   └── uploads/
│
└── templates/
    └── index.html
```

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/hafizsalsabil/speech-emotion-recognition.git
```

Move into the project

```bash
cd speech-emotion-recognition
```

Create virtual environment

```bash
python -m venv .venv
```

Activate environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

## 📊 Workflow

1. Upload audio
2. Convert audio to WAV
3. Load audio signal
4. Generate Mel-Spectrogram
5. Extract features
6. CNN prediction
7. Display emotion probabilities
8. Export results to CSV

---

## 📈 Model

Architecture:

- Convolutional Neural Network (CNN)

Feature Extraction:

- Mel-Spectrogram

Framework:

- TensorFlow / Keras

Output Classes:

- Anger
- Happiness
- Neutral
- Sadness

---

## 📁 Dataset

The model was trained using an Indonesian Speech Emotion Recognition dataset.

The dataset consists of speech recordings labeled into four emotional categories:

- Anger
- Happiness
- Neutral
- Sadness

---

## 🖼 Mel-Spectrogram Examples

Examples are available in the `docs/` directory.

---

## 👨‍💻 Author

**Ananda Hafiz Salsabil**

Information Systems Undergraduate Student

Universitas Gunadarma

GitHub:
https://github.com/hafizsalsabil

---

## 📄 License

This project was developed for educational and research purposes.