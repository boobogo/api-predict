from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
try:
    model = joblib.load('best_knn_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Root endpoint (GET request)
@app.route('/predict', methods=['GET'])
def home():
    return jsonify({"message": "KNN Model API is running"}), 200

# Prediction endpoint (POST request)
@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Invalid input, expecting JSON"}), 400
    
    try:
        data = request.get_json()  # parse JSON
        features = np.array(data['features']).reshape(1, -1)  # convert features to numpy array
        prediction = model.predict(features)  # make prediction
        return jsonify({'prediction': int(prediction[0])}), 200
    except KeyError:
        return jsonify({"error": "Missing 'features' in request"}), 400
    except Exception as e:
        return jsonify({"error": f"Error making prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
