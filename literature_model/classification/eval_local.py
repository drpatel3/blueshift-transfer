"""Local evaluation of trained DeBERTa model — computes F1, precision, recall, accuracy."""
import json
import os
import random
import sys

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_sm")
CORPUS = os.path.join(os.path.dirname(__file__), "corpus.json")

def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    model.eval()

    print("Loading corpus...", flush=True)
    with open(CORPUS, encoding="utf-8") as f:
        corpus = json.load(f)

    random.seed(42)
    pairs = random.sample(corpus, min(n_samples, len(corpus)))
    print(f"Evaluating on {len(pairs)} pairs...", flush=True)

    tp = fp = fn = tn = 0
    batch_size = 32

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        texts = [p["text"] for p in batch]
        tables = [p["table_str"] for p in batch]
        labels = [p["label"] for p in batch]

        inputs = tokenizer(
            tables,
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1).tolist()

        for pred, label in zip(preds, labels):
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1
            else:
                tn += 1

        done = min(i + batch_size, len(pairs))
        if done % 160 == 0 or done == len(pairs):
            print(f"  {done}/{len(pairs)}", flush=True)

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print()
    print(f"=== Evaluation Results ({len(pairs)} samples) ===")
    print(f"Accuracy:  {accuracy:.4f} ({tp+tn}/{total})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"  TP={tp}  FP={fp}")
    print(f"  FN={fn}  TN={tn}")
    print(f"  Positive samples: {tp+fn}, Negative samples: {tn+fp}")


if __name__ == "__main__":
    main()
