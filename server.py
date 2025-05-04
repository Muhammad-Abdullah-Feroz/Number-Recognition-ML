from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

SVC = joblib.load("./Models/SVC_model.pkl")
KNN = joblib.load("./Models/KNN_model.pkl")
LR = joblib.load("./Models/LR_model.pkl")
NB = joblib.load("./Models/NaiveBayes_model.pkl")

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return render_template('./index.html')

@app.route("/predict-svc", methods = ['POST'])
def predictSVC():
    try:
        data = request.get_json()
        feature_list = data["pixels"]
        
        feature_array = np.array(feature_list).reshape(1, -1)
        # print(f"Feature list shape: {feature_array.shape}")
        # print(f"Feature list: {feature_array}")
        prediction = SVC.predict(feature_array)
        return jsonify(prediction.tolist())
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400
    

@app.route("/predict-lr", methods = ['POST'])
def predictLR():
    try:
        data = request.get_json()
        feature_list = data["pixels"]
        
        feature_array = np.array(feature_list).reshape(1, -1)
        # print(f"Feature list shape: {feature_array.shape}")
        # print(f"Feature list: {feature_array}")
        prediction = LR.predict(feature_array)
        return jsonify(prediction.tolist())
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400
    

@app.route("/predict-knn", methods = ['POST'])
def predictKNN():
    try:
        data = request.get_json()
        feature_list = data["pixels"]
        
        feature_array = np.array(feature_list).reshape(1, -1)
        # print(f"Feature list shape: {feature_array.shape}")
        # print(f"Feature list: {feature_array}")
        prediction = KNN.predict(feature_array)
        return jsonify(prediction.tolist())
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400
    
    
@app.route("/predict-nb", methods = ['POST'])
def predictNB():
    try:
        data = request.get_json()
        feature_list = data["pixels"]
        
        feature_array = np.array(feature_list).reshape(1, -1)
        # print(f"Feature list shape: {feature_array.shape}")
        # print(f"Feature list: {feature_array}")
        prediction = NB.predict(feature_array)
        return jsonify(prediction.tolist())
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400
    
    
if __name__ == '__main__':
    app.run(debug=True)