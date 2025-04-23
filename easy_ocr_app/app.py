from fastapi import FastAPI, HTTPException, Request
import base64
import numpy as np
import cv2
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from make_inference import (
    get_image_results,
    parse_ocr_result,
    detect_and_blur_cigarettes,
    predict_image_adult_content,
    probability_threshold,
    logger
)
import time
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Replace @app.on_event("startup") and @app.on_event("shutdown")"""
    asyncio.create_task(worker())  
    logger.info("Application startup complete")
    yield
    logger.info("Application shutting down")

app = FastAPI(lifespan=lifespan)
request_queue = asyncio.Queue(maxsize=100)  
executor = ThreadPoolExecutor(max_workers=20)  

async def worker():
    """Background task processor"""
    while True:
        try:
            image_data, response_future = await request_queue.get()
            result = await process_image_async(image_data)
            response_future.set_result(result)
        except Exception as e:
            response_future.set_exception(e)
        finally:
            request_queue.task_done()

async def process_image_async(image_data: bytes) -> dict:
    loop = asyncio.get_event_loop()
    
    try:
        result = await loop.run_in_executor(
            executor,
            lambda: process_image_sync(image_data)
        )
        return result
        
    except Exception as e:
        logger.error(f"Async processing error: {str(e)}")
        raise

def process_image_sync(image_data: bytes) -> dict:
    def unified_image_processing(img_data: bytes, target_size=(1024, 1024)):
        """Унифицированная обработка изображения с контролем цветового пространства"""
        try:
            img_cv = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_cv is None:
                raise ValueError("Failed to decode original image")

            original_h, original_w = img_cv.shape[:2]

            original_bgr = img_cv.copy()

            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

            _, buffer = cv2.imencode('.jpg', img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            processed_img = cv2.imdecode(np.frombuffer(buffer.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB) 

            if target_size:
                processed_img = cv2.resize(processed_img, target_size, interpolation=cv2.INTER_LINEAR)

            return processed_img, original_bgr, (original_w, original_h)

        except Exception as e:
            logger.error(f"Image processing error: {str(e)}", exc_info=True)
            raise

    try:
        processed_img, original_bgr, (orig_w, orig_h) = unified_image_processing(image_data)

        adult_class, adult_prob = predict_image_adult_content(
            cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)  
        )
        
        is_adult = adult_class in ["hentai", "porn", "sexy", 'dick'] and adult_prob >= probability_threshold
        
        if is_adult:
            logger.warning(f"Adult content detected: {adult_class} (prob: {adult_prob:.2f})")
            debug_path = f"debug_{int(time.time())}_initial_adult.jpg"
            cv2.imwrite(filename=debug_path, img=cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
            logger.info(f"Saved debug image to {debug_path}")
            blurred = cv2.GaussianBlur(original_bgr, (1025, 1025), 0)  # Работаем с BGR
            _, buffer = cv2.imencode('.jpg', blurred)
            debug_path = f"debug_{int(time.time())}_blurred_adult.jpg"
            cv2.imwrite(filename=debug_path, img=blurred)
            logger.info(f"Saved debug image to {debug_path}")
            return {
                "bboxes": [],
                "text": "",
                "blurred_image": base64.b64encode(buffer).decode('utf-8'),
                "status": "success"
            }

        img_with_cigarettes = detect_and_blur_cigarettes(processed_img.copy())

        ocr_results, final_img = get_image_results(img_with_cigarettes.copy())

        bboxes, text, blurred_b64 = parse_ocr_result(ocr_results, final_img)

        blurred_cv = cv2.imdecode(np.frombuffer(base64.b64decode(blurred_b64), dtype=np.uint8), cv2.IMREAD_COLOR)
        blurred_cv = cv2.resize(blurred_cv, (orig_w, orig_h))
        
        _, buffer = cv2.imencode('.jpg', blurred_cv)
        debug_path = f"debug_{int(time.time())}_blurred_final.jpg"
        cv2.imwrite(img=blurred_cv, filename=debug_path)

        return {
            "bboxes": bboxes,
            "text": text,
            "blurred_image": base64.b64encode(buffer).decode('utf-8'),
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Full processing pipeline error: {str(e)}", exc_info=True)
        return {"error": str(e), "status": "error"}

@app.post('/process_image')
async def process_image(request: Request):
    try:
        data = await request.json()
        if not data or 'image' not in data:
            raise HTTPException(status_code=400, detail="No image provided")

        if request_queue.full():
            raise HTTPException(status_code=429, detail="Server is busy. Try later.")

        loop = asyncio.get_event_loop()
        response_future = loop.create_future()  
        image_data = base64.b64decode(data['image'])
        
        await request_queue.put((image_data, response_future))
        result = await response_future  
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5002)