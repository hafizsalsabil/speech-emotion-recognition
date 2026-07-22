import numpy as np
from tensorflow.keras.models import load_model

# ==========================================
# LOAD MODEL
# ==========================================

model = load_model(
    "ser_model_augmented.h5"
)

emotion_names = [
    "Anger",
    "Sadness",
    "Neutral",
    "Happiness"
]


# ==========================================
# SINGLE PREDICTION
# ==========================================

def predict_emotion(X):

    prediction = model.predict(
        X,
        verbose=0
    )[0]

    predicted_class = np.argmax(
        prediction
    )

    emotion = emotion_names[
        predicted_class
    ]

    confidence = round(
        float(
            prediction[predicted_class]
        ) * 100,
        2
    )

    probabilities = {

        emotion_names[i]: round(
            float(prediction[i]) * 100,
            2
        )

        for i in range(
            len(emotion_names)
        )

    }

    return {

        "emotion": emotion,

        "confidence": confidence,

        "probabilities": probabilities

    }