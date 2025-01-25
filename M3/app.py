import joblib
import numpy as np
from flask import Flask, request, jsonify

# Load the trained model
model = joblib.load("model.joblib")

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from POST request
    data = request.get_json(force=True)
    input_data = np.array(data['features']).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)

    # Return the result as JSON
    return jsonify({"prediction": int(prediction[0])})


if __name__ == "__main__":
    app.run(host='0.0.0.0')
