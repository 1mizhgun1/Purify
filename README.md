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
   
   REDIS_DATA_DIRECTORY=/var/purify_redis_data
   REDIS_PASSWORD=some_password
   REDIS_URL=redis://default:${REDIS_PASSWORD}@redis:6379/0?protocol=3
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

6. Первоначальный классификатор ```(Baseline решение):``` [тык](purify_ml/ml_classifier_app)
   
## Наш оберег

![kanev](images/kanev.png)