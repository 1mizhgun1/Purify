FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    cmake \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.EasyOCR/
RUN mkdir -p /root/.EasyOCR/user_network
RUN mkdir -p /root/.EasyOCR/model

COPY ./easy_ocr_app/weights_config/user_network/custom_example.py /root/.EasyOCR/user_network
COPY ./easy_ocr_app/weights_config/user_network/custom_example.yaml /root/.EasyOCR/user_network
COPY ./easy_ocr_app/weights_config/model/custom_example.pth /root/.EasyOCR/model
COPY ./easy_ocr_app/weights_config/model/craft_mlt_25k.pth /root/.EasyOCR/model

COPY ./easy_ocr_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install easyocr

COPY ./easy_ocr_app/weights_config/adults_content_recognition/final_model.pth .
COPY ./easy_ocr_app/weights_config/cigarettes_detection/weights.pt .

COPY ./easy_ocr_app/app.py .
COPY ./easy_ocr_app/make_inference.py .

CMD ["python", "app.py"]