from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("flood_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    rainfall = float(request.form["rainfall"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    visibility = float(request.form["cloud_visibility"])

    data = pd.DataFrame(
        [[rainfall, temperature, humidity, visibility]],
        columns=[
            "rainfall",
            "temperature",
            "humidity",
            "cloud_visibility"
        ]
    )

    result = model.predict(data)[0]

    if result == 1:
        return render_template("flood.html")
    else:
        return render_template("no_flood.html")


if __name__ == "__main__":
    app.run(debug=True)
