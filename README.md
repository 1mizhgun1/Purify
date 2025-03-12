# API для проекта Purify команды Scratch Senior Devs

## Запуск

1. Создать в корне проекта файл `.env`. Пример содержимого файла:

    ```env
   LOG_DIRECTORY=/var/log/purify
   
   MAIN_LOG_FILE=${LOG_DIRECTORY}/main.log
   CONFIG_FILE=src/config/config.yaml
   
   FRONTEND_HOST_NAME=http://localhost:3000
    ```

2. При первом запуске выполнить команду:

   ```bash
   sudo mkdir -p /var/log/purify
   ```
   
   Директория для лог файла должна совпадать с `LOG_DIRECTORY` из `.env`.

3. Собрать и запустить проект:

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
   
4. Убедиться, что всё работает:

   ```bash
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
   
## Наш амулет

![kanev](images/kanev.png)
