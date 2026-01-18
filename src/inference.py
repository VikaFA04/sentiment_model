from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import HF_MODEL_ID, ID2LABEL


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def predict_texts(texts: List[str], tokenizer, model, max_length: int = 256) -> List[Dict]:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)

    pred_ids = torch.argmax(probs, dim=-1).cpu().tolist()
    confs = probs.max(dim=-1).values.cpu().tolist()

    out = []
    for t, pid, c in zip(texts, pred_ids, confs):
        out.append({
            "text": t,
            "pred_id": int(pid),
            "pred_label": ID2LABEL[int(pid)],
            "confidence": float(c),
        })
    return out
