# Документация проекта: Zaure/Ұстаз RAG Чат-бот

## Обзор проекта

Мультиязычный чат-бот, позволяющий:

* генерировать ответы на основе данных приказов (в базе данных — 83 приказа)

## Структура проекта

### Ядро приложения (`src/rag_chatbot/`)

#### Точка входа

* **`main.py`**: Точка входа FastAPI приложения с полным управлением жизненным циклом

  * Настройка CORS middleware для междоменных запросов
  * Управление запуском/остановкой приложения с инициализацией аналитики
  * Встроенная панель аналитики по адресу `/analytics`
  * API-эндпоинты (проверка состояния, чат, сессии, метрики)
  * Эндпоинт метрик в стиле Prometheus для мониторинга

#### Конфигурация (`config/`)

* **`settings.py`**: Централизованное управление конфигурациями с помощью Pydantic

  * Настройки LLM (конфигурация OpenAI API)
  * Пути к векторным хранилищам и документам
  * Параметры памяти и ограничения по скорости
  * Таймауты сессий и настройки памяти диалога
  * Загрузка переменных окружения из `.env` файлов

#### API-слой (`api/`)

##### Маршруты (`api/routes/`)

* **`chat.py`**: Главный эндпоинт для чата

  * Управление памятью диалога по сессиям
  * Ограничение скорости (5 запросов в минуту на сессию)
  * Получение документов и работа с PDF
  * Отслеживание аналитики всех взаимодействий
  * Обработка ошибок и фоновые задачи
  * Контекстно-зависимая генерация ответов

* **`health.py`**: Эндпоинты проверки состояния системы

  * Проверка базового состояния
  * Проверка подключения к базе данных
  * Отчет о статусах компонентов

* **`sessions.py`**: Эндпоинты управления сессиями

  * Создание и получение сессий
  * Статистика и мониторинг активных сессий
  * Очистка и управление жизненным циклом сессий

##### Middleware (`api/middleware/`)

* **`auth.py`**: Middleware для аутентификации и контроля доступа к API

  * Проверка API-ключа
  * Аутентификация и авторизация запросов

#### Ядро компонентов (`core/`)

* **`database.py`**: Настройка базы данных через SQLAlchemy

  * Инициализация SQLite базы для аналитики
  * Проверка состояния базы и управление соединениями
  * Настройка фабрики сессий и внедрение зависимостей

* **`instances.py`**: Синглтоны для основных сервисов

  * Глобальный менеджер сессий
  * Создание ограничителя скорости
  * Управление экземпляром RAG пайплайна

* **`llm.py`**: Конфигурация и инициализация языковой модели

  * Настройка GPT-4 от OpenAI с контролем temperature
  * Отслеживание использования токенов и стоимости
  * Управление параметрами LLM

* **`rag_pipeline.py`**: Обертка над RAG пайплайном и интеграция агента

  * Инициализация UnifiedRAGAgent с конфигурацией
  * Интерфейс обработки запросов и генерации ответов
  * Связь между FastAPI и компонентами RAG

* **`rate_limiter.py`**: Реализация ограничения запросов

  * Ограничение на сессию по скользящему окну
  * Мониторинг статистики ограничений
  * Настраиваемые лимиты (по умолчанию 5 запросов/мин.)

* **`session_manager.py`**: Управление жизненным циклом сессий

  * Хранение сессий в памяти с таймаутом 30 мин.
  * Интеграция памяти диалога через LangChain
  * Очистка сессий и сборка мусора

#### Модели данных (`models/`)

* **`schemas.py`**: Pydantic-схемы для запросов и ответов API

  * `ChatMessage`: валидация входящего сообщения
  * `ChatResponse`: структурированный ответ с метаданными
  * `SessionCreateResponse`: формат ответа на создание сессии
  * Модели для ограничения скорости и проверки состояния

* **`analytics.py`**: SQLAlchemy ORM модели для аналитики

  * `Session`: отслеживание пользовательских сессий
  * `Conversation`: хранение и анализ диалогов
  * `AnalyticsEvent`: логирование системных событий
  * `DocumentUsage`: статистика использования документов
  * `QueryAnalytics`: метрики запросов и производительности
  * `SystemMetrics`: общая производительность системы

#### Сервисы (`services/`)

* **`analytics_service.py`**: Обработка и отчетность аналитики

  * Отслеживание и сохранение диалогов
  * Агрегация данных для панелей
  * Статистика использования документов
  * Подсчет метрик производительности
  * Отслеживание ошибок и лимитов

#### Фоновые задачи (`tasks/`)

* **`analytics_tasks.py`**: Управление фоновыми задачами аналитики

  * Асинхронная обработка аналитических данных
  * Периодическая очистка и обслуживание
  * Очередь задач и обработка ошибок

#### Утилиты (`utils/`)

* **`background_tasks.py`**: Утилиты фоновых задач

  * Очистка сессий и сборка мусора
  * Задачи по обслуживанию ограничителя скорости
  * Планирование обслуживания

* **`logger.py`**: Централизованная настройка логирования

  * Структурированное логирование с временными метками
  * Управление уровнями логов и форматами
  * Единые стандарты логирования по приложению

### RAG пайплайн (`rag_pipeline/`)

#### Основной агент

* **`agent.py`**: UnifiedRAGAgent — ядро системы RAG

  * **Принятие решений**: интеллектуальный выбор между поиском документов, базой знаний или прямым ответом
  * **Поиск документов**: семантический поиск PDF по embedding
  * **RAG Fusion**: генерация нескольких запросов для лучшего покрытия
  * **Память диалога**: ответы с учетом истории
  * **Мультиязычность**: автоопределение языка и соответствующие ответы
  * **Function Calling**: структурированные вызовы функций OpenAI

#### Вспомогательные компоненты

* **`prompts.py`**: Системные промпты и шаблоны запросов

  * Мультиязычные промпты (русский, казахский, английский)
  * Промпты для RAG fusion
  * Промпты для маршрутизации агентом

* **`rag_fusion_pipeline.py`**: Реализация RAG Fusion

  * Генерация нескольких запросов
  * Слияние ранговых позиций Reciprocal Rank Fusion
  * Контекстно-зависимая генерация ответов

* **`add_doc.py`**: Скрипт для добавления документов в базу faiss


### Управление документами

#### Хранилище документов (`documents/`)

* **`83/kk/`**: Приказы на казахском (14 PDF)
* **`83/rus/`**: Приказы на русском (14 PDF)
* **`documents.json`**: Связь имён документов с путями

#### Векторное хранилище

* **`faiss_base/`**: FAISS-векторное хранилище для поиска
* **`double_db_faiss-16-06-2025/`**: Дополнительное хранилище FAISS
* **`data/document_embeddings.npy`**: Кешированные embedding

### Веб-интерфейсы (`static/`)

#### Веб-страницы

* **`st_page.py`**: Интерфейс Streamlit

  * Взаимодействие с чат-ботом
  * Управление сессиями и историей диалогов
  * Поиск и скачивание документов

* **`streamlit_dashboard.py`**: Панель аналитики на Streamlit

  * Визуализация аналитики в реальном времени
  * Мониторинг производительности
  * Статистика взаимодействий

### Конфигурационные файлы

#### Окружение и зависимости

* **`requirements.txt`**: Зависимости Python

  * FastAPI, LangChain, OpenAI, SQLAlchemy
  * Библиотеки аналитики и мониторинга
  * Векторные базы и ML

* **`.env.example`**: Шаблон переменных окружения

  * API-ключ OpenAI
  * Подключение к БД
  * Параметры конфигурации

#### Метаданные проекта

* **`README.md`**: Обзор и инструкция по запуску проекта
