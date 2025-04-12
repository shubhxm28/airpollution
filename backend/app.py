from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load('linear_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        pm25 = float(data['pm25'])

        input_array = np.array([[temperature, humidity, pm25]])
        prediction = model.predict(input_array)[0]
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
