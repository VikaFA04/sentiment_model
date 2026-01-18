import argparse
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import HF_MODEL_ID, ID2LABEL


def load():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def predict(texts: List[str], tokenizer, model, max_length: int = 256) -> List[Dict]:
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
        out.append({"text": t, "pred_id": pid, "pred_label": ID2LABEL[pid], "confidence": float(c)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, default=None, help="Single text")
    p.add_argument("--max_length", type=int, default=256)
    args = p.parse_args()

    if not args.text:
        raise SystemExit("Provide --text")

    tokenizer, model = load()
    res = predict([args.text], tokenizer, model, max_length=args.max_length)[0]
    print(res)


if __name__ == "__main__":
    main()
