# Sentiment Classification (RuBERT, 3 classes)

Русскоязычная модель тональности на базе BERT.
Классы:
- 0 — negative
- 1 — neutral
- 2 — positive

## Quickstart
```bash
pip install -r requirements.txt
python -m src.predict --text "Очень понравилось, спасибо!"
