import joblib
from flask import Flask, request, jsonify
from config import CLASSIC_PIPELINE_PATH, W2V_MODEL_PATH
from tools import preprocess_text, preprocess_for_w2v

model = joblib.load(CLASSIC_PIPELINE_PATH)
model_w2v = joblib.load(W2V_MODEL_PATH)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    preprocessed_text = preprocess_for_w2v(text)
    prediction = model_w2v.predict_proba(preprocessed_text)

    positive_proba = prediction[:, 1]
    # print(positive_proba)
    threshold = 0.35

    return jsonify({'prediction': int(positive_proba[0] > threshold), 'prob': positive_proba[0]})

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)