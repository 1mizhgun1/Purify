# API для проекта Purify команды Scratch Senior Devs

## Запуск

1. Создать в корне проекта файл `.env`. Пример содержимого файла:

    ```env
   LOG_DIRECTORY=/var/log/purify
   
   MAIN_LOG_FILE=${LOG_DIRECTORY}/main.log
   CONFIG_FILE=src/config/config.yaml
   
   FRONTEND_HOST_NAME=*
   
   MISTRAL_AI_API_KEY=some_key
   CHATGPT_API_KEY=some_key
   DEEPSEEK_API_KEY=some_key

   REDIS_PASSWORD=some_password
   REDIS_URL=redis://default:${REDIS_PASSWORD}@redis:6379/0?protocol=3
   
   MINIO_ROOT_USER=some_username
   MINIO_ROOT_PASSWORD=some_password
   MINIO_ENDPOINT=minio:9000
   MINIO_REGION=us-east-1

   POSTGRES_DB=some_db
   POSTGRES_USER=ssd
   POSTGRES_PASSWORD=some_password
   POSTGRES_URL=postgres://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
    ```

2. При первом запуске выполнить команды:

   ```shell
   sudo mkdir -p /var/log/purify
   sudo mkdir -p /var/purify_redis_data
   ```
   
   Директория должны совпадать с `LOG_DIRECTORY` и `REDIS_DATA_DIRECTORY` из `.env`.

3. При первом запуске создать в корне проекта redis.conf с таким содержимым:

   ```conf
   save 10 1
   appendonly yes
   appendfsync everysec
   dir /data
   dbfilename dump.rdb
   requirepass some_password
   ```
   
   Где `some_password` заменить на пароль из `env` переменной `REDIS_PASSWORD`

4. Собрать и запустить проект:

    ```shell
    docker-compose build
    docker-compose up
    ```
    
    или
    
    ```shell
    docker-compose build
    docker-compose up -d
    ```
    
    Во втором случае логи не будет видно в stdout, можно посмотреть с помощью
    
    ```shell
    docker logs main
    ```
   
5. Убедиться, что всё работает:

   ```shell
   curl http://127.0.0.1:8080/api/v1/ping
   ```
   
   В ответ должно прийти `pong`, также должен появиться лог с такой структурой (на самом деле будет неформатированный):

   ```json
   {
      "time": "2025-03-12T20:13:30.986935368+03:00",
      "level": "INFO",
      "msg": "finished request",
      "x-request-id": "5a951745-6717-47d9-bb0c-fb30e9fd3879",
      "method": "GET",
      "uri": "/api/v1/ping",
      "status": "200"
   }
   ```

6. NLP-сервис. ```(Baseline решение):``` [тык](nlp_words_app/)

```bash
# Запуск:
cd nlp_words_app
docker-compose -f docker-compose.dev.yml up --build # (добавлен в .gitignore)
curl -X POST http://localhost:5003/analyze \
-H "Content-Type: application/json" \
-d '{
  "blocks": [
    "Легко представить себе мир, описываемый [негативной матерной] лексикой … мир, в котором крадут и обманывают, бьют и боятся, в котором «все расхищено, предано, продано», в котором падают, но не поднимаются, берут, но не дают, в котором либо работают до изнеможения, либо халтурят — но в любом случае относятся к работе, как и ко всему окружающему и всем окружающим, с отвращением либо с глубоким безразличием, — и всё кончается тем, что приходит полный пиздец.",
    "Банк России будет поддерживать такую жесткость денежно-кредитных условий, которая необходима для возвращения инфляции к цели в 2026 году. Это означает продолжительный период проведения жесткой денежно-кредитной политики. Дальнейшие решения по ключевой ставке будут приниматься в зависимости от скорости и устойчивости снижения инфляции и инфляционных ожиданий. В рамках базового сценария это предполагает среднюю ключевую ставку в диапазоне 19,5–21,5% годовых в 2025 году и 13,0–14,0% годовых в 2026 году. По прогнозу Банка России, с учетом проводимой денежно-кредитной политики годовая инфляция снизится до 7,0–8,0% в 2025 году, вернется к 4,0% в 2026 году и будет находиться на цели в дальнейшем."
  ]
}'
```

7. OCR-сервис (easy_ocr_app/)[тык]

```bash
# Запуск:
cd easy_ocr_app
docker build -t ocr-app .
docker run -p 5002:5002 ocr-app

# Одно изображение
curl -X POST \
  http://localhost:5002/process_image \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 -i /Users/chervonikov_alexey/Desktop/mat.jpeg | tr -d '\n')\"}"

# Несколько изображений
curl -X POST \
  http://localhost:5002/process_images_batch \
  -H "Content-Type: application/json" \
  -w "\n" \
  -d "{\"images\": [\"$(base64 -i /Users/chervonikov_alexey/Desktop/projects/Technopark_Spring_2025/diploma_project/porn/dick.jpeg | tr -d '\n')\", \"$(base64 -i /Users/chervonikov_alexey/Desktop/projects/Technopark_Spring_2025/diploma_project/porn/mike.jpeg | tr -d '\n')\"]}"

# В несколько потоков
curl -X POST \
  http://localhost:5002/process_images_parallel \
  -H "Content-Type: application/json" \
  -w "\n" \
  -d "{\"images\": [\"$(base64 -i /Users/chervonikov_alexey/Desktop/projects/Technopark_Spring_2025/diploma_project/easyocr/invalid_images/IMG_8329.JPG | tr -d '\n')\", \"$(base64 -i /Users/chervonikov_alexey/Desktop/2025-04-05_16.18.59.jpg | tr -d '\n')\", \"$(base64 -i /Users/chervonikov_alexey/Desktop/invalid_images/IMG_8346.JPG | tr -d '\n')\"]}"
```

8. Video Service (video_processing_service/)[тык]

```bash
# Healthcheck
curl http://localhost:5003/health
```

```bash
# Send video link
curl -X POST "http://localhost:5003/transcribe" \
     -H "Content-Type: application/json" \
     -d '{"url":"https://ok.ru/video/42697558629", "max_duration":60}'
```



## Наш оберег

![kanev](images/kanev.png)