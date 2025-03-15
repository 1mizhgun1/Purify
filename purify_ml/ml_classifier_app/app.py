import joblib
from flask import Flask, request, jsonify
from tools import preprocess_text
import pandas as pd

model = joblib.load('toxic_comment_classifier.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    preprocessed_text = preprocess_text(text)
    df = pd.DataFrame(data={'comment': [preprocessed_text]})
    prediction = model.predict(df)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)