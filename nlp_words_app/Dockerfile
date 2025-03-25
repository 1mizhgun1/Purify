FROM python:3.11-slim

WORKDIR /app

COPY /nlp_words_app/requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y gcc python3-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN python -c "import nltk; nltk.download('stopwords')"
RUN python -c "import pymorphy3; pymorphy3.MorphAnalyzer()"
EXPOSE 5000

CMD ["python", "app.py"]

