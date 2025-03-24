from flask import Flask, request, jsonify
from config import *
from utils import *
from functools import lru_cache
from collections import defaultdict

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        negative_words = get_negative_words(text)
        
        return jsonify({
            "status": "success",
            "negative_words": negative_words,
            "count": len(negative_words)
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)