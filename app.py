
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_model.pkl")

model = joblib.load(MODEL_PATH)

app = Flask(__name__, template_folder="HTML pages")

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/find-your-crop")
def findyourcrop():
    return render_template("find_your_crop.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        nitrogen = float(request.form["Nitrogen"])
        phosphorous = float(request.form["Phosphorous"])
        potassium = float(request.form["Potassium"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        input_df = pd.DataFrame(
            [[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]],
            columns=FEATURE_COLUMNS
        )

        crop = str(model.predict(input_df)[0]).title()

        if hasattr(model, "predict_proba"):
            confidence = round(
                float(np.max(model.predict_proba(input_df))) * 100, 2
            )
            prediction_text = f"Recommended Crop: {crop} ({confidence}% confidence)"
        else:
            prediction_text = f"Recommended Crop: {crop}"

        return render_template(
            "find_your_crop.html",
            prediction_text=prediction_text
        )

    except KeyError as e:
        return render_template(
            "find_your_crop.html",
            prediction_text=f"Error: Missing form field {e}."
        )
    except ValueError:
        return render_template(
            "find_your_crop.html",
            prediction_text="Error: Please enter valid numeric values for all fields."
        )


if __name__ == "__main__":
    app.run(debug=True)
