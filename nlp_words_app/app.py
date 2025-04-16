from flask import Flask, request, jsonify
from utils import *
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "PUT", "DELETE"],
    "allow_headers": ["Content-Type"],
    "supports_credentials": True,
    "max_age": 86400
}})

@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        blocks = data.get('blocks', '')

        negative_blocks = []
        for block in blocks:
            negative_words = get_negative_words(block)
            if len(list(negative_words)) > 0:
                negative_blocks.append({
                    "block": block,
                    "negative_words": list(negative_words)
                })

        return jsonify(negative_blocks)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)