FROM python:3.12.9-alpine

WORKDIR /app

RUN apk update && apk add --no-cache \
    build-base \
    gcc \
    g++ \
    musl-dev \
    libffi-dev \
    openssl-dev \
    && rm -rf /var/cache/apk/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "import nltk; nltk.download('stopwords')"
EXPOSE 5000

CMD ["python", "app.py"]