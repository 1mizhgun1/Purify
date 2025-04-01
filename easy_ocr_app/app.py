from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
import io
from PIL import Image
from make_inference import get_image_results, parse_ocr_result 
import logging 
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Формат логов
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = RotatingFileHandler(
        'app.log', 
        maxBytes=1024*1024, 
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Консольный вывод
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

logger = setup_logger()

@app.route('/process_image', methods=['POST'])
def process_image():
    logger.info("Received request to /process_image")
    data = request.get_json()
    if not data or 'image' not in data:
        logger.error("No image provided in request")
        return jsonify({"error": "No image provided"}), 400
    
    base64_str = data['image']
    
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        logger.info("Prepared image...")
        ocr_results = get_image_results(img_cv)
        logger.info("OCR result collected...")
        bboxes, combined_text = parse_ocr_result(ocr_results)
        logger.info(f"Text extracted: {combined_text}")
        return jsonify({"bboxes": bboxes if bboxes else [],
                        "text": combined_text})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)