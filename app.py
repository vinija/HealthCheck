from flask import Flask, request, jsonify
import joblib
import logging
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')  # Ensure the model is correctly loaded
scaler = joblib.load('scaler.pkl')  # Ensure the scaler is correctly loaded

logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    logging.info(f"Received data: {data}")

    # Prepare input for prediction
    input_data = [[
        float(data['age']),
        float(data['bmi']),
        float(data['bloodPressure']),
        float(data['totalCholesterol']),
        float(data['ldl']),
        float(data['hdl']),
        float(data['tsh']),
        float(data['lamotrigine']),
        float(data['bloodSugar'])
    ]]

    logging.info(f"Input data for prediction: {input_data}")

    try:
        input_array = np.array(input_data)
        input_scaled = scaler.transform(input_array)  # Apply the same scaling
        prediction = model.predict(input_scaled)[0]

        logging.info(f"Prediction: {prediction}")
        return jsonify(prediction=int(prediction))
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
