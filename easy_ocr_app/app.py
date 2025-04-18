from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from make_inference import (
    get_image_results,
    parse_ocr_result,
    get_image_results_batch,
    parse_ocr_result_batch,
    preprocess_batch_images,
    process_single_result,
    logger,
    model_cigarette_detection,
    detection_threshold
)

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=20)

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

def process_single_image_worker(image_data: bytes, idx: int = None) -> dict:
    """Функция для обработки одного изображения в потоке"""
    try:
        start_time = time.time()
        
        img_cv = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        img_cv = cv2.resize(img_cv, (1024, 1024))

        img_no_cigarettes = detect_and_blur_cigarettes(img_cv)

        debug_path = f"debug_{idx}_{int(time.time())}.jpg" if idx is not None else f"debug_{int(time.time())}.jpg"
        cv2.imwrite(debug_path, img_no_cigarettes)
        logger.info(f"Image {idx} (no cigarettes) saved to {debug_path}")
        
        ocr_results = get_image_results(img_no_cigarettes)
        # bboxes, combined_text, blurred_base64 = parse_ocr_result(ocr_results, img_cv.copy())
        result = process_single_result(ocr_results, img_no_cigarettes.copy(), idx, debug_dir="debug")
        processing_time = time.time() - start_time
        logger.info(f"Image {idx} processed in {processing_time:.2f}s")
        
        return {
            **result,
            "processing_time": processing_time,
            "status": "success",
            "image_index": idx,
            "cigarette_detection": {
                "status": "processed",
                "debug_image": debug_path
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing image {idx}: {str(e)}")
        return {
            "error": str(e),
            "status": "error",
            "image_index": idx
        }

@app.route('/process_image', methods=['POST'])
def process_single_image():
    """Обработка одного изображения (старый эндпоинт для совместимости)"""
    logger.info("Received single image request")
    data = request.get_json()
    
    if not data or 'image' not in data:
        logger.error("No image provided")
        return jsonify({"error": "No image provided"}), 400
    
    try:
        start_time = time.time()
        
        image_data = np.frombuffer(base64.b64decode(data['image']), dtype=np.uint8)
        img_cv = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        img_cv = cv2.resize(img_cv, (1024, 1024))

        img_no_cigarettes = detect_and_blur_cigarettes(img_cv)
        
        debug_path = f"debug_original_{int(time.time())}.jpg"
        cv2.imwrite(debug_path, img_no_cigarettes)
        logger.info(f"Image (no cigarettes) saved to {debug_path}")
        
        ocr_results = get_image_results(img_cv)
        result = process_single_result(ocr_results, img_no_cigarettes.copy(), None, debug_dir="debug")
        # bboxes, combined_text, blurred_base64 = parse_ocr_result(ocr_results, img_cv.copy())
        logger.info(f"Processing time: {time.time()-start_time:.2f}s")
        
        return jsonify({
            **result,
            "processing_time": time.time() - start_time,
            "status": "success",
            "cigarette_detection": {
                "status": "processed",
                "debug_image": debug_path
            }
        })
        
        
    except Exception as e:
        logger.error(f"Single image processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/process_images_batch', methods=['POST'])
def process_images_batch():
    """Обработка батча изображений (новый эндпоинт)"""
    logger.info("Received batch processing request")
    data = request.get_json()
    
    if not data or 'images' not in data:
        logger.error("No images array provided")
        return jsonify({"error": "No images provided"}), 400
    
    if not isinstance(data['images'], list):
        logger.error("Images should be provided as array")
        return jsonify({"error": "Images should be provided as array"}), 400
    
    try:
        start_time = time.time()
        base64_images = data['images']
        
        MAX_BATCH_SIZE = 32
        if len(base64_images) > MAX_BATCH_SIZE:
            logger.warning(f"Batch size too large ({len(base64_images)}), truncating to {MAX_BATCH_SIZE}")
            base64_images = base64_images[:MAX_BATCH_SIZE]
        
        images_array = preprocess_batch_images(base64_images)
        logger.info(f"Preprocessed {len(images_array)} images in {time.time()-start_time:.2f}s")
        
        ocr_results = get_image_results_batch(images_array)
        logger.info(f"Batch OCR completed in {time.time()-start_time:.2f}s")
        
        batch_results = parse_ocr_result_batch(ocr_results, images_array)
        
        for idx, img in enumerate(images_array):
            debug_path = f"debug_batch_{idx}_{int(time.time())}.jpg"
            cv2.imwrite(debug_path, img)
        
        logger.info(f"Total processing time: {time.time()-start_time:.2f}s")
        
        return jsonify({
            "results": batch_results,
            "processed_count": len(batch_results),
            "success_count": sum(1 for r in batch_results if r.get("status") == "success"),
            "processing_time": time.time() - start_time
        })
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/process_images_parallel', methods=['POST'])
def process_images_parallel():
    """Обработка нескольких изображений параллельно"""
    logger.info("Received parallel processing request")
    data = request.get_json()
    
    if not data or 'images' not in data:
        logger.error("No images array provided")
        return jsonify({"error": "No images provided"}), 400
    
    if not isinstance(data['images'], list):
        logger.error("Images should be provided as array")
        return jsonify({"error": "Images should be provided as array"}), 400
    
    try:
        start_time = time.time()
        base64_images = data['images']
        
        MAX_IMAGES = 32
        if len(base64_images) > MAX_IMAGES:
            logger.warning(f"Too many images ({len(base64_images)}), truncating to {MAX_IMAGES}")
            base64_images = base64_images[:MAX_IMAGES]
        
        image_data_list = [base64.b64decode(img) for img in base64_images]
        
        futures = [executor.submit(process_single_image_worker, img_data, idx) 
                 for idx, img_data in enumerate(image_data_list)]
        
        results = [future.result() for future in futures]
        
        success_count = sum(1 for r in results if r.get("status") == "success")
        total_time = time.time() - start_time
        
        logger.info(f"Processed {len(results)} images in {total_time:.2f}s ({success_count} successes)")
        
        return jsonify({
            "results": results,
            "processed_count": len(results),
            "success_count": success_count,
            "processing_time": total_time
        })
        
    except Exception as e:
        logger.error(f"Parallel processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    """Проверка работоспособности сервиса"""
    try:
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        get_image_results(test_image)
        return jsonify({"status": "healthy", "gpu_available": torch.cuda.is_available()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    import torch  
    app.run(host='0.0.0.0', port=5002, debug=True)

# from flask import Flask, request, jsonify
# import base64
# import numpy as np
# import cv2
# import io
# from PIL import Image
# from make_inference import get_image_results, parse_ocr_result, logger

# app = Flask(__name__)

# @app.route('/process_image', methods=['POST'])
# def process_image():
#     logger.info("Received request to /process_image")
#     data = request.get_json()
#     if not data or 'image' not in data:
#         logger.error("No image provided in request")
#         return jsonify({"error": "No image provided"}), 400
    
#     base64_str = data['image']
    
#     try:

#         image_data = np.frombuffer(base64.b64decode(base64_str), dtype=np.uint8)
#         img_cv = cv2.imdecode(image_data, cv2.IMREAD_COLOR)  

#         original_path = "original_image.jpg"
#         cv2.imwrite(original_path, img_cv)
#         logger.info(f"Original image saved to {original_path}")

#         logger.info("Prepared image...")
#         ocr_results = get_image_results(img_cv)
#         logger.info("OCR result collected...")
#         bboxes, combined_text, blurred_base64 = parse_ocr_result(ocr_results, img_cv.copy())
#         logger.info(f"Text extracted: {combined_text}")

#         response_data = {
#             "bboxes": bboxes if bboxes else [],
#             "text": combined_text,
#             "blurred_image": blurred_base64 if blurred_base64 else None
#         }
        
#         return jsonify(response_data)
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5002, debug=True)