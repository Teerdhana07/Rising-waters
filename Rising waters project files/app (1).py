from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'floods.save')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')

model = None
scaler = None


def load_artifacts():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run the model training script first.")


# Load the model when the application starts
load_artifacts()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict')
def predict():
    return render_template('input.html')


@app.route('/result', methods=['POST'])
def result():
    try:
        # Get input values
        annual_rainfall = float(request.form['annual_rainfall'])
        cloud_coverage = float(request.form['cloud_coverage'])
        jun_sep = float(request.form['jun_sep'])
        mar_may = float(request.form['mar_may'])
        oct_dec = float(request.form['oct_dec'])
        jan_feb = float(request.form['jan_feb'])

        # Create feature array
        features = np.array([[annual_rainfall,
                              cloud_coverage,
                              jun_sep,
                              mar_may,
                              oct_dec,
                              jan_feb]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]

        if prediction == 1:
            return render_template('result_flood.html')
        else:
            return render_template('result_no_flood.html')

    except Exception as e:
        return f"Error: {str(e)}", 400


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)