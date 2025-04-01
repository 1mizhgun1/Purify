FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.EasyOCR/
RUN mkdir -p /root/.EasyOCR/user_network
RUN mkdir -p /root/.EasyOCR/model

COPY ./weights_config/user_network/custom_example.py /root/.EasyOCR/user_network
COPY ./weights_config/user_network/custom_example.yaml /root/.EasyOCR/user_network
COPY ./weights_config/model/custom_example.pth /root/.EasyOCR/model
COPY ./weights_config/model/craft_mlt_25k.pth /root/.EasyOCR/model

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install easyocr

COPY app.py .
COPY make_inference.py .

CMD ["python", "app.py"]