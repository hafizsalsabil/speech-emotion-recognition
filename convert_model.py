from tensorflow.keras.models import load_model

model = load_model("ser_model_augmented.keras")

model.save("ser_model_augmented.h5")