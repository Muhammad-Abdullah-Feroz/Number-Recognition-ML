from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

model = joblib.load("SVC_model.pkl")

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return render_template('./index.html')

@app.route("/predict", methods = ['POST'])
def predict():
    try:
        data = request.get_json()
        feature_list = data["pixels"]
        
        feature_array = np.array(feature_list).reshape(1, -1)
        # print(f"Feature list shape: {feature_array.shape}")
        # print(f"Feature list: {feature_array}")
        prediction = model.predict(feature_array)
        return jsonify(prediction.tolist())
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400
if __name__ == '__main__':
    app.run(debug=True)