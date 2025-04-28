from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib

model = joblib.load("SVC_model.pkl")

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return render_template('./index.html')

@app.route("/predict", methods = ['POST'])
def predict():
    data = request.get_json(force=True)
    # Here you would typically load your model and make a prediction
    # For demonstration, we'll just return the received data
    prediction = data
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)