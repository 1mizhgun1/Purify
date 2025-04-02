import easyocr
import cv2
import torch
import numpy as np
import os
import requests
import base64
from PIL import Image
import io

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

# print(reader)

# print(weights_dir + "/user_network")
# print(weights_dir + "/model")

def get_image_results(image, easyocr_reader = reader):
    result = easyocr_reader.readtext(image)
    return result

# def blur_bboxes(image_path, bboxes, output_path=None, blur_strength=25):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Не удалось загрузить изображение")
    
#     blur_strength = blur_strength if blur_strength % 2 != 0 else blur_strength + 1
    
#     for bbox in bboxes:
#         pts = np.array(bbox, dtype=np.int32)
#         x, y, w, h = cv2.boundingRect(pts)
#         roi = image[y:y+h, x:x+w]
#         blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
#         mask = np.zeros(roi.shape[:2], dtype=np.uint8)
#         cv2.fillPoly(mask, [pts - [x, y]], 255)
#         image[y:y+h, x:x+w] = cv2.bitwise_and(blurred_roi, blurred_roi, mask=mask) + \
#                               cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    
#     if output_path:
#         cv2.imwrite(output_path, image)
    
#     return image

def parse_ocr_result(ocr_results):

    combined_text = " ".join([text for (_, text, _) in ocr_results])
    bboxes = [
        [[int(x), int(y)] for [x, y] in bbox]  
        for (bbox, _, _) in ocr_results
    ]

    print(bboxes)
    print(combined_text)

    payload = {
        "blocks": [combined_text]  
    }

    response = requests.post(
        "http://host.docker.internal:5001/analyze",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        if len(combined_text) < 3 and len(combined_text) != 0:
               return bboxes, combined_text

        elif len(response.json()) != 0:
            neg_words = response.json()[0]['negative_words']
            if len(neg_words) != 0:
                return bboxes, combined_text
        else:
            return None, ""
    else:
        print("Error:", response.status_code, response.text)
