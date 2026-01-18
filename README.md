# Sentiment Classification (RuBERT) — Hack & Change (Changellenge x Правительство Москвы)

Модель классификации эмоциональной тональности русскоязычных отзывов горожан:
- **0 — negative**
- **1 — neutral**
- **2 — positive**

Модель обучена в рамках **Hack & Change by Changellenge (ML/Web трек)**.
Достигнуто качество: **macro-F1 = 0.80**.

## Почему это важно
Ручной разбор большого потока комментариев — дорого и медленно. Автоматическая тональная разметка помогает быстрее выявлять проблемные цифровые сервисы и точки роста.

## What’s inside
- **Streamlit web app**: загрузка CSV → разметка → скачивание размеченного файла → визуализация → расчёт macro-F1 на валидации
- **Model inference** через Hugging Face Transformers
- Notebook с экспериментами: `notebooks/sentiment-model-final.ipynb`

## Quickstart (out of the box)

### 1) Install
```
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run web app
```
streamlit run app/app.py
```

Открой браузер и:

загрузи CSV с колонкой text → получишь pred (0/1/2) + график распределения

загрузи валидационный CSV с text и истинной меткой → получишь macro-F1

### Model
Base: BertForSequenceClassification (3 labels)
Weights hosted on Hugging Face Hub: VikaFA/sentiment-ru-bert-hackchange

### Notes
Модель возвращает как класс, так и confidence (максимальная вероятность softmax).

Для воспроизводимости и удобства инференса веса вынесены в HF Hub, GitHub содержит только код и приложение.
