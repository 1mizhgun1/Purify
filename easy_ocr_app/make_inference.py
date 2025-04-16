import easyocr
import cv2
import torch
import numpy as np
import os
import requests
import base64
from PIL import Image
import io
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Tuple, Dict, Any
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
import contextlib

with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

logging.getLogger('yolov5').propagate = False
logging.getLogger('ultralytics').propagate = False

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        yield

model_path = config['model_path']
model_cigarette_detection_path = config['model_cigarette_detection_path']
num_classes = config['num_classes']
class_names = config['class_names']
probability_threshold = config['probabilty_threshold']
detection_threshold = config['detection_threshold']

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

device_gpu = torch.cuda.is_available()
reader = easyocr.Reader(['ru', 'en'], 
    recog_network = 'custom_example',
    detect_network="craft",
    gpu=device_gpu, quantize=False,
    cudnn_benchmark=True)

print(f"Recognition model is successfully loaded")

DEVICE = 'cuda' if device_gpu else 'cpu'
model_porn_classifer_resnet50 = resnet50()
model_porn_classifer_resnet50.fc = nn.Linear(model_porn_classifer_resnet50.fc.in_features, num_classes)
model_porn_classifer_resnet50.load_state_dict(torch.load(model_path, map_location=DEVICE))
model_porn_classifer_resnet50 = model_porn_classifer_resnet50.to(DEVICE)
print(f"Adult content classifier model is succesfully loaded!")

with suppress_output():
    model_cigarette_detection = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path=model_cigarette_detection_path,
        verbose=False,
        trust_repo=True  
    )
    model_cigarette_detection.eval()
print(f"Cigarette detetction model is successfully loaded!")

if device_gpu:
    warmup_batch = np.zeros([4, 600, 800, 3], dtype=np.uint8)
    reader.readtext_batched(warmup_batch)
    logger.info("GPU warmup completed")

def predict_image_adult_content(img_cv,
                                model = model_porn_classifer_resnet50,
                                device=DEVICE,
                                class_names=class_names,
                                probability_threshold = probability_threshold):
    """
   Функция для предсказания класса изображения из OpenCV/numpy массива.

   Параметры:
       img_cv (np.ndarray): Изображение в формате OpenCV (BGR) или numpy array.
       model (torch.nn.Module): Обученная модель.
       device (torch.device): Устройство (например, 'cuda', 'mps', 'cpu').
       class_names (list): Список имен классов.
       probability_threshold (float): Порог уверенности для классификации.

   Возвращает:
       tuple: (Имя предсказанного класса, вероятность)
             или ("neutral", 0.0) если вероятность ниже порога
   """
    model.eval()
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(img_rgb)


    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_prob, predicted = torch.max(probabilities, 1)
    

    predicted_probability = predicted_prob.item()
    predicted_class = class_names[predicted.item()]
    logger.info(f"Class: {predicted_class}, Prob: {predicted_probability}")
    if predicted_probability < probability_threshold:
        return "neutral", 0.0

    predicted_class = class_names[predicted.item()]
    return predicted_class, predicted_probability

def preprocess_batch_images(base64_images: List[str]) -> np.ndarray:
    """Конвертирует список base64 строк в numpy массив изображений одного размера"""

    def detect_and_blur_cigarettes(image: np.ndarray,
                               confidence_threshold: float = detection_threshold) -> np.ndarray:
        """Детектирует и размывает сигареты на изображении"""
        results = model_cigarette_detection(image)

        blurred_image = image.copy()

        for *xyxy, conf, cls in results.xyxy[0]:
            logger.info(f'Confidence: {conf}')
            if conf > confidence_threshold:
                x1, y1, x2, y2 = map(int, xyxy)
                roi = blurred_image[y1:y2, x1:x2]
                blurred_roi = cv2.GaussianBlur(roi, (451, 451), 0)
                blurred_image[y1:y2, x1:x2] = blurred_roi
        return blurred_image
    
    processed_images = []
    for base64_str in base64_images:
        try:
            image_data = np.frombuffer(base64.b64decode(base64_str), dtype=np.uint8)
            img_cv = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            img_resized = cv2.resize(img_cv, (1024, 1024))  
            img_no_cigarettes = detect_and_blur_cigarettes(img_resized)
            processed_images.append(img_no_cigarettes)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            processed_images.append(np.zeros((1024, 1024, 3), dtype=np.uint8))
    
    return np.array(processed_images)

def get_image_results_batch(images: np.ndarray) -> List[Any]:
    """Выполняет OCR для батча изображений"""
    try:
        return reader.readtext_batched(images)
    except Exception as e:
        logger.error(f"Batch OCR error: {str(e)}")
        raise

def blur_bboxes(image: np.ndarray, bboxes: List[List[List[int]]],
                blur_strength: int = 201,
                image_path: str = None) -> np.ndarray:
    """Размывает указанные области на изображении"""
    if image is None:
        raise ValueError("Image is None")
    
    blur_strength = blur_strength if blur_strength % 2 != 0 else blur_strength + 1
    image = image.copy()
    
    for bbox in bboxes:
        pts = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        roi = image[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts - [x, y]], 255)
        image[y:y+h, x:x+w] = cv2.bitwise_and(blurred_roi, blurred_roi, mask=mask) + \
                              cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    
    if image_path is not None:
        try:
            cv2.imwrite(image_path, image)
            logger.info(f"Blurred image saved to {image_path}")
        except Exception as e:
            logger.error(f"Failed to save blurred image to {image_path}: {str(e)}")
            raise
    
    return image

def analyze_text_with_api(texts: List[str]) -> List[Dict[str, Any]]:
    """Отправляет тексты на анализ во внешний API"""
    try:
        payload = {"blocks": texts}
        response = requests.post(
            "http://nlp_words:5001/analyze",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        return response.json() if response.status_code == 200 else []
    except Exception as e:
        logger.error(f"API analysis error: {str(e)}")
        return []

def process_single_result(ocr_result: List[Tuple], 
                          image: np.ndarray, 
                          image_idx:int, 
                          debug_dir: str = None) -> Dict[str, Any]:
    """Обрабатывает результаты OCR для одного изображения с возможностью сохранения отладочной информации
    
    Args:
        ocr_result: Результаты распознавания от EasyOCR
        image: Исходное изображение (numpy array)
        debug_dir: Директория для сохранения отладочных файлов (None - не сохранять)
    
    Returns:
        Словарь с результатами обработки:
        {
            "bboxes": список найденных bbox'ов,
            "text": объединённый текст,
            "blurred_image": base64 размытого изображения,
            "negative_words": список негативных слов,
            "debug_info": информация для отладки (только если debug_dir указан)
        }
    """
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_suffix = f"{timestamp}_{image_idx}" if image_idx is not None else timestamp

    adult_class, adult_prob = predict_image_adult_content(image)
    # logger.info(f"Class: {adult_class}, Prob: {adult_prob}")
    is_adult = adult_class in ["hentai", "porn", "sexy", 'dick'] and adult_prob >= probability_threshold

    if is_adult:
        logger.warning(f"Adult content detected: {adult_class} (prob: {adult_prob:.2f})")
        blurred_image = cv2.GaussianBlur(image, (551, 551), 0)

        _, buffer = cv2.imencode('.jpg', blurred_image)
        blurred_base64 = base64.b64encode(buffer).decode('utf-8')

        if debug_dir:
            debug_path = os.path.join(debug_dir, f"adult_blurred_{file_suffix}.jpg")
            cv2.imwrite(debug_path, blurred_image)

        return {
            "bboxes": [],
            "text": "",
            "blurred_image": blurred_base64,
            "negative_words": [],
            "adult_content": {
                "class": adult_class,
                "probability": adult_prob,
                "status": "blurred_full_image"
            },
            "debug_info": {
                "adult_debug_image": debug_path
            } if debug_dir else None
        }

    combined_text = " ".join([text for (_, text, _) in ocr_result])
    bboxes = [[[int(x), int(y)] for [x, y] in bbox] for (bbox, _, _) in ocr_result]
    
    if debug_dir:
        debug_orig_path = os.path.join(debug_dir, f"orig_with_boxes_{file_suffix}.jpg")
        debug_image = image.copy()
        for bbox in bboxes:
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(debug_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imwrite(debug_orig_path, debug_image)
    
    api_response = analyze_text_with_api([combined_text])
    neg_words = api_response[0].get("negative_words", []) if api_response else []
    
    bad_bboxes = []
    if len(combined_text) >= 3 or not combined_text.strip():
        for (bbox, text, _) in ocr_result:
            if any(bad_word.lower() in text.lower() for bad_word in neg_words):
                bad_bboxes.append([[int(x), int(y)] for [x, y] in bbox])
    
    blurred_base64 = None
    debug_info = {}
    
    if bad_bboxes:
        blurred_image = image.copy()
        
        if debug_dir:
            debug_bad_boxes_path = os.path.join(debug_dir, f"bad_boxes_{file_suffix}.jpg")
            debug_image = image.copy()
            for bbox in bad_bboxes:
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(debug_image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.imwrite(debug_bad_boxes_path, debug_image)
            debug_info["bad_boxes_image"] = debug_bad_boxes_path
        
        blurred_image = blur_bboxes(
            blurred_image, 
            bad_bboxes,
            image_path=os.path.join(debug_dir, f"blurred_{file_suffix}.jpg") if debug_dir else None
        )
        
        _, buffer = cv2.imencode('.jpg', blurred_image)
        blurred_base64 = base64.b64encode(buffer).decode('utf-8')
        
        if debug_dir:
            logger.debug(f"Timestamp: {timestamp}")
            debug_final_path = os.path.join(debug_dir, f"final_{file_suffix}.jpg")
            cv2.imwrite(debug_final_path, blurred_image)
            debug_info["final_image"] = debug_final_path

    result = {
        "bboxes": bad_bboxes if bad_bboxes else [],
        "text": combined_text,
        "blurred_image": blurred_base64,
        "negative_words": neg_words,
        "adult_content": {
            "class": adult_class,
            "probability": adult_prob,
            "status": "checked_no_adult"
        }
    }
    
    if debug_dir:
        logger.debug(f"Timestamp: {timestamp}")
        result["debug_info"] = debug_info
        result["debug_timestamp"] = file_suffix
    
    return result

def parse_ocr_result_batch(ocr_results: List[List[Tuple]], images: np.ndarray) -> List[Dict[str, Any]]:
    """Обрабатывает результаты батч-OCR"""
    batch_results = []
    
    for idx, (single_result, orig_image) in enumerate(zip(ocr_results, images)):
        try:
            result = process_single_result(single_result, orig_image, idx, debug_dir="debug")
            result["image_index"] = idx
            result["status"] = "success"
            batch_results.append(result)
        except Exception as e:
            logger.error(f"Error processing image {idx}: {str(e)}")
            batch_results.append({
                "image_index": idx,
                "error": str(e),
                "status": "error"
            })
    
    return batch_results

def get_image_results(image: np.ndarray) -> List[Tuple]:
    """Старая функция для обработки одного изображения"""
    return reader.readtext(image)

def parse_ocr_result(ocr_results: List[Tuple], image: np.ndarray) -> Tuple:
    """Старая функция для обработки одного результата"""
    batch_result = parse_ocr_result_batch([ocr_results], [image])
    result = batch_result[0]
    logger.debug(f"Result: {result}")
    return (result["bboxes"], result["text"], result["blurred_image"])



