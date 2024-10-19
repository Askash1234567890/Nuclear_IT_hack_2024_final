# Анализатор Exit Interview

Этот проект предоставляет веб-сервис на основе FastAPI для анализа данных exit interview. Он включает функции для
загрузки данных, предварительной обработки, кластеризации, генерации статистики и создания визуальных графиков.

Подробнее про проделанные эксперименты:
[Отчет о проделанных экспериментах](research/Readme.md)

Подробнее про реализацию через Telegram бота:
[Telegram бот](tg_bot/README.md)

## Содержание

- [Установка](#установка)
- [Использование](#использование)
- [API Endpoints](#api-endpoints)
- [Структура проекта](#структура-проекта)
- [Лицензия](#лицензия)
- [Docker](#docker)
- [Запуск Telegram бота](#запуск-telegram-бота)

## Установка

1. Клонируйте репозиторий:
    ```sh
    git clone https://github.com/Askash1234567890/Nuclear_IT_hack_2024_final.git
    cd Nuclear_IT_hack_2024_final
    ```

2. Создайте виртуальное окружение и активируйте его:
    ```sh
    python -m venv venv
    source venv/bin/activate  # В Windows используйте `venv\Scripts\activate`
    ```

3. Установите необходимые зависимости:
    ```sh
    pip install -r requirements.txt
    ```

## Использование

1. Запустите сервер FastAPI:
    ```sh
    uvicorn scripts.api:app --reload
    ```

2. Откройте браузер и перейдите по адресу `http://127.0.0.1:8000/docs`, чтобы получить доступ к интерактивной
   документации API.

## API Endpoints

- **POST /load_data**: Загрузить данные в анализатор.
- **POST /get_personal_statistic**: Получить личную статистику по заданному ID.
- **POST /get_graphics**: Сгенерировать графики для загруженных данных в виде PDF.

### Примеры запросов

- **Загрузить данные**
    ```sh
    curl -X 'POST' \
      'http://127.0.0.1:8000/load_data' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "data": [{"column1": "value1", "column2": "value2"}]
    }'
    ```

- **Предварительная обработка данных**
    ```sh
    curl -X 'POST' \
      'http://127.0.0.1:8000/preprocess' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "words": ["word1", "word2"]
    }'
    ```

## Структура проекта

```plaintext
.
├── scripts/
│   ├── api.py                # Приложение FastAPI
│   ├── analyzer.py           # Класс анализатора с методами обработки данных
│   ├── grad_search.py        # Реализация градиентного поиска
├── requirements.txt          # Зависимости проекта
└── README.md                 # Документация проекта
```

## Лицензия

Этот проект лицензирован по лицензии MIT. См. файл [LICENSE](LICENSE) для получения подробной информации.

## Docker

Для запуска проекта с использованием Docker выполните следующие шаги:

1. Постройте Docker образ:
    ```sh
    docker build -t exit-interview-analyzer .
    ```

2. Запустите контейнер:
    ```sh
    docker run -p 8000:80 exit-interview-analyzer
    ```

## Запуск Telegram бота

Для запуска Telegram бота выполните следующие шаги:

1. Перейдите в директорию `tg_bot`:
    ```sh
    cd tg_bot
    ```

2. Постройте Docker образ для Telegram бота:
    ```sh
    docker build -t tg-bot .
    ```

3. Запустите контейнер:
    ```sh
    docker run -d tg-bot
    ```

```
