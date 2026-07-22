# ==========================================================
# IMPORT LIBRARY
# ==========================================================

import os

import pandas as pd

from flask import (

    Flask,

    render_template,

    request,

    send_file

)

from prediction import (

    predict_emotion

)

from audio_processing import (
    convert_to_wav,
    load_audio,
    create_mel_spectrogram,
    extract_feature
)

# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)

latest_csv_path = None

# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

def create_output_folder():

    folder = os.path.join(

        "static",

        "uploads"

    )

    os.makedirs(

        folder,

        exist_ok=True

    )

    return folder

# ==========================================================
# HOME
# ==========================================================

@app.route("/")

def home():

    return render_template(

        "index.html",

        results=None,

        total_files=0,

        average_confidence=0,

        emotion_summary={

            "Anger": 0,

            "Sadness": 0,

            "Neutral": 0,

            "Happiness": 0

        },

        success_count=0,

        failed_count=0

    )

# ==========================================================
# MULTIPLE PREDICTION
# ==========================================================

@app.route(

    "/predict",

    methods=["POST"]

)

def predict():

    audio_files = request.files.getlist(

        "multiple_audio"

    )

    if len(audio_files) == 0:

        return render_template(

            "index.html",

            error="Silakan pilih minimal satu file audio."

        )

    if audio_files[0].filename == "":

        return render_template(

            "index.html",

            error="Silakan pilih minimal satu file audio."

        )

    results = []

    # ==========================================================
    # PROCESS ALL AUDIO FILES
    # ==========================================================

    for audio_file in audio_files:

        wav_path = None

        try:

            # Load Audio
            wav_path = convert_to_wav(audio_file)

            audio, sr, duration = load_audio(wav_path)

            # Create Mel Spectrogram
            mel_db = create_mel_spectrogram(
                audio,
                sr
            )

            # Feature Extraction
            X = extract_feature(
                mel_db
            )

            # Prediction
            prediction = predict_emotion(
                X
            )

            results.append({

                "filename": audio_file.filename,

                "emotion": prediction["emotion"],

                "confidence": prediction["confidence"],

                "probabilities": prediction["probabilities"],

                "duration": duration,

                "status": "Success"

            })

        except Exception as e:

            print(e)

            results.append({

                "filename": audio_file.filename,

                "emotion": "-",

                "confidence": "-",

                "probabilities": None,

                "duration": "-",

                "status": "Failed"

            })
        
        finally:

            if wav_path and os.path.exists(wav_path):

                os.remove(wav_path)

    # ==========================================================
    # SUMMARY
    # ==========================================================

    total_files = len(results)

    valid_results = [

        r for r in results

        if r["status"] == "Success"

    ]

    valid_confidence = [

        r["confidence"]

        for r in valid_results

    ]

    average_confidence = 0

    if valid_confidence:

        average_confidence = round(

            sum(valid_confidence)

            / len(valid_confidence),

            2

        )

    success_count = len(valid_results)

    failed_count = total_files - success_count

    emotion_summary = {

        "Anger": 0,

        "Sadness": 0,

        "Neutral": 0,

        "Happiness": 0

    }

    for r in valid_results:

        emotion_summary[

            r["emotion"]

        ] += 1

    # ==========================================================
    # SAVE CSV
    # ==========================================================

    csv_folder = create_output_folder()

    # ==========================================================
    # AUTO FILE NAME
    # ==========================================================

    file_number = 1

    while True:

        file_name = f"prediction_result_{file_number}.csv"

        csv_path = os.path.join(

            csv_folder,

            file_name

        )

        if not os.path.exists(csv_path):

            break

        file_number += 1

    csv_results = []

    for r in results:

        row = {

            "Filename": r["filename"],

            "Prediction": r["emotion"],

            "Confidence (%)": r["confidence"],

            "Duration (s)": r["duration"],

            "Status": r["status"]

        }

        if r["probabilities"]:

            row["Anger (%)"] = r["probabilities"]["Anger"]

            row["Sadness (%)"] = r["probabilities"]["Sadness"]

            row["Neutral (%)"] = r["probabilities"]["Neutral"]

            row["Happiness (%)"] = r["probabilities"]["Happiness"]

        else:

            row["Anger (%)"] = "-"

            row["Sadness (%)"] = "-"

            row["Neutral (%)"] = "-"

            row["Happiness (%)"] = "-"

        csv_results.append(row)

    df = pd.DataFrame(csv_results)

    df.to_csv(

        csv_path,

        index=False

    )

    global latest_csv_path

    latest_csv_path = csv_path

    # ==========================================================
    # RETURN
    # ==========================================================

    return render_template(

        "index.html",

        results=results,

        total_files=total_files,

        average_confidence=average_confidence,

        emotion_summary=emotion_summary,

        success_count=success_count,

        failed_count=failed_count

    )

# ==========================================================
# DOWNLOAD CSV
# ==========================================================

@app.route("/download_csv")
def download_csv():

    global latest_csv_path

    if latest_csv_path is None:

        return "CSV file not found.", 404

    return send_file(

        latest_csv_path,

        as_attachment=True

    )

# ==========================================================
# RUN FLASK
# ==========================================================

if __name__ == "__main__":

    app.run(

        debug=True

    )