from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
import io
from PIL import Image
from make_inference import get_image_results, parse_ocr_result, logger

app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    logger.info("Received request to /process_image")
    data = request.get_json()
    if not data or 'image' not in data:
        logger.error("No image provided in request")
        return jsonify({"error": "No image provided"}), 400
    
    base64_str = data['image']
    
    try:

        image_data = np.frombuffer(base64.b64decode(base64_str), dtype=np.uint8)
        img_cv = cv2.imdecode(image_data, cv2.IMREAD_COLOR)  

        original_path = "original_image.jpg"
        cv2.imwrite(original_path, img_cv)
        logger.info(f"Original image saved to {original_path}")

        logger.info("Prepared image...")
        ocr_results = get_image_results(img_cv)
        logger.info("OCR result collected...")
        bboxes, combined_text, blurred_base64 = parse_ocr_result(ocr_results, img_cv.copy())
        logger.info(f"Text extracted: {combined_text}")

        response_data = {
            "bboxes": bboxes if bboxes else [],
            "text": combined_text,
            "blurred_image": blurred_base64 if blurred_base64 else None
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)