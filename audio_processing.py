import os
import numpy as np
import librosa
import librosa.display
import matplotlib
import tempfile

from pydub import AudioSegment

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ==========================================
# LOAD AUDIO
# ==========================================

def load_audio(audio_file):

    audio, sr = librosa.load(
        audio_file,
        sr=16000
    )

    duration = round(
        librosa.get_duration(
            y=audio,
            sr=sr
        ),
        2
    )

    return audio, sr, duration

# ==========================================
# CONVERT TO WAV
# ==========================================

def convert_to_wav(audio_file):

    audio = AudioSegment.from_file(audio_file)

    audio = audio.set_channels(1)

    audio = audio.set_frame_rate(16000)

    temp_file = tempfile.NamedTemporaryFile(

        suffix=".wav",

        delete=False

    )

    audio.export(

        temp_file.name,

        format="wav"

    )

    return temp_file.name

# ==========================================
# CREATE MEL SPECTROGRAM
# ==========================================

def create_mel_spectrogram(audio, sr):

    mel = librosa.feature.melspectrogram(

        y=audio,

        sr=sr,

        n_mels=128

    )

    mel_db = librosa.power_to_db(

        mel,

        ref=np.max

    )

    return mel_db

# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_feature(mel_db):

    max_length = 215

    pad_width = max_length - mel_db.shape[1]

    if pad_width > 0:

        mel_db = np.pad(
            mel_db,
            ((0,0),(0,pad_width)),
            mode="constant"
        )

    else:

        mel_db = mel_db[:, :215]

    X = np.array([mel_db])

    X = X[..., np.newaxis]

    X = X.astype("float32")

    X = (
        X - X.min()
    ) / (
        X.max() - X.min()
    )

    return X