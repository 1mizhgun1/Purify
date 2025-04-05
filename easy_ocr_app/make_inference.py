import easyocr
import cv2
import torch
import numpy as np
import os
import requests
import base64
from PIL import Image
import io
import datetime
import logging 
from logging.handlers import RotatingFileHandler

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = RotatingFileHandler(
        'app.log', 
        maxBytes=1024*1024, 
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

logger = setup_logger()

def read_image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read())
        image_bytes = base64.b64decode(base64_image) 
    return image_bytes

def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

base_dir = os.path.dirname(os.path.abspath(__file__))  
weights_dir = os.path.join(base_dir, 'weights_config') 

device_gpu = True if torch.cuda.is_available() else False
print(device_gpu)
reader = easyocr.Reader(['ru', 'en'], 
                        # user_network_directory = weights_dir + "/user_network",
                        # model_storage_directory = weights_dir + "/model",
                        recog_network = 'custom_example',
                        detect_network="craft",
                        gpu=device_gpu, quantize=False)

def get_image_results(image, easyocr_reader = reader):
    result = easyocr_reader.readtext(image)
    return result

def blur_bboxes(image, bboxes, output_path=None, blur_strength=101):
    if image is None:
        raise ValueError("Не удалось загрузить изображение")
    
    blur_strength = blur_strength if blur_strength % 2 != 0 else blur_strength + 1
    
    for bbox in bboxes:
        pts = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        roi = image[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts - [x, y]], 255)
        image[y:y+h, x:x+w] = cv2.bitwise_and(blurred_roi, blurred_roi, mask=mask) + \
                              cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    
    if output_path:
         cv2.imwrite(output_path, image)
    
    return image

def parse_ocr_result(ocr_results, image):

    combined_text = " ".join([text for (_, text, _) in ocr_results])
    bboxes = [
        [[int(x), int(y)] for [x, y] in bbox]  
        for (bbox, _, _) in ocr_results
    ]

    payload = {
        "blocks": [combined_text]  
    }

    response = requests.post(
        "http://host.docker.internal:5001/analyze",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        
        api_response = response.json()
        if not api_response:  
            return None, "", None

        if len(combined_text) < 3 and combined_text.strip():
            blurred_image = image.copy()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            blurred_path = f"blurred_image_{timestamp}.jpg"
            blur_bboxes(blurred_image, bad_bboxes, 
                        # blurred_path
                        )
            logger.info(f"Blurred image saved to {blurred_path}")
            _, buffer = cv2.imencode('.jpg', blurred_image)
            blurred_base64 = base64.b64encode(buffer).decode('utf-8')
            return bboxes, combined_text, blurred_base64

        neg_words = api_response[0].get("negative_words", [])
        if not neg_words:  
            return None, "", None

        bad_bboxes = []
        for (bbox, text, _) in ocr_results:
            if any(bad_word.lower() in text.lower() for bad_word in neg_words):
                bad_bboxes.append([[int(x), int(y)] for [x, y] in bbox])

        if not bad_bboxes:  
            return None, "", None

        blurred_image = image.copy()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        blurred_path = f"blurred_image_{timestamp}.jpg"
        final_image = blur_bboxes(blurred_image, bad_bboxes, 
                                #   blurred_path
                                  )
        logger.info(f"Blurred image saved to {blurred_path}")
        _, buffer = cv2.imencode('.jpg', final_image)
        blurred_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return bad_bboxes, combined_text, blurred_base64
        
    else:
        print("Error:", response.status_code, response.text)
