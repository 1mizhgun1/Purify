from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
import io
from PIL import Image
from make_inference import get_image_results, parse_ocr_result  

app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
    
    base64_str = data['image']
    
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        ocr_results = get_image_results(img_cv)
        bboxes, combined_text = parse_ocr_result(ocr_results)
        return jsonify({"bboxes": bboxes if bboxes else [],
                        "text": combined_text})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)