import io
import pandas as pd
import streamlit as st

from src.inference import load_model, predict_texts
from src.metrics import macro_f1

st.set_page_config(page_title="Sentiment (RuBERT)", layout="centered")

st.title("Классификация тональности (0/1/2)")
st.caption("0 — negative, 1 — neutral, 2 — positive")

@st.cache_resource
def _load():
    return load_model()

tokenizer, model = _load()

st.subheader("1) Разметка CSV")
st.write("Загрузи CSV с колонкой `text` (или выбери колонку вручную).")

file = st.file_uploader("CSV файл", type=["csv"])
if file is not None:
    df = pd.read_csv(file)

    st.write("Превью данных:")
    st.dataframe(df.head(10))

    text_col = st.selectbox("Колонка с текстом", options=list(df.columns), index=0)

    max_rows = st.slider("Сколько строк обработать (для демо)", 50, min(5000, len(df)), min(500, len(df)))
    batch_size = st.selectbox("Batch size", options=[8, 16, 32, 64], index=2)

    if st.button("Запустить разметку"):
        texts = df[text_col].astype(str).tolist()[:max_rows]

        preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            res = predict_texts(batch, tokenizer, model)
            preds.extend([r["pred_id"] for r in res])

        out = df.iloc[:max_rows].copy()
        out["pred"] = preds

        st.success("Готово! Ниже — результат.")
        st.dataframe(out.head(20))

        st.subheader("Визуализация распределения классов")
        st.bar_chart(out["pred"].value_counts().sort_index())

        # download
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        st.download_button(
            label="Скачать размеченный CSV",
            data=buf.getvalue().encode("utf-8"),
            file_name="labeled.csv",
            mime="text/csv",
        )

st.divider()

st.subheader("2) Оценка macro-F1 на валидации")
st.write("Загрузи CSV с колонками `text` и целевой колонкой (например `label` или `target`).")

val_file = st.file_uploader("CSV валидации (с разметкой эксперта)", type=["csv"], key="val")
if val_file is not None:
    vdf = pd.read_csv(val_file)
    st.dataframe(vdf.head(10))

    text_col_v = st.selectbox("Колонка с текстом (val)", options=list(vdf.columns))
    target_col = st.selectbox("Колонка с true-меткой (0/1/2)", options=list(vdf.columns))

    max_rows_v = st.slider("Сколько строк оценить", 50, min(5000, len(vdf)), min(500, len(vdf)), key="maxv")

    if st.button("Посчитать macro-F1"):
        texts = vdf[text_col_v].astype(str).tolist()[:max_rows_v]
        y_true = vdf[target_col].tolist()[:max_rows_v]

        res = predict_texts(texts, tokenizer, model)
        y_pred = [r["pred_id"] for r in res]

        score = macro_f1(y_true, y_pred)
        st.metric("macro-F1", f"{score:.3f}")
