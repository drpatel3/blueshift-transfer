"""TF-IDF + XGBoost text-table relevance classifier.

Lightweight baseline for comparison with the DeBERTa model.
Uses the same corpus.json built by text_table_classification.py.

Usage:
    python xgb_classification.py --train
    python xgb_classification.py --predict --table-json '{"caption":"...","headers":[],"rows":[]}' --text "some passage"
"""

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from text_table_classification import serialize_table, DEFAULT_CORPUS

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "model_xgb"


def train(corpus: list[dict], model_dir: str):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    table_strs = [item["table_str"] for item in corpus]
    texts = [item["text"] for item in corpus]
    labels = np.array([item["label"] for item in corpus])

    print("Fitting TF-IDF on table strings...")
    tfidf_table = TfidfVectorizer(max_features=10_000, sublinear_tf=True)
    X_table = tfidf_table.fit_transform(table_strs)
    print(f"  Done ({time.time() - t0:.1f}s)")

    print("Fitting TF-IDF on text passages...")
    tfidf_text = TfidfVectorizer(max_features=10_000, sublinear_tf=True)
    X_text = tfidf_text.fit_transform(texts)
    print(f"  Done ({time.time() - t0:.1f}s)")

    X = hstack([X_table, X_text])
    print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")

    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    print("Training XGBoost...")
    clf = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    print(f"Training finished ({time.time() - t0:.1f}s)")

    val_preds = clf.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    print(f"\nValidation accuracy: {acc:.4f}")
    print(classification_report(y_val, val_preds, target_names=["irrelevant", "relevant"]))

    clf.save_model(str(model_dir / "xgb_model.json"))
    with open(model_dir / "tfidf_table.pkl", "wb") as f:
        pickle.dump(tfidf_table, f)
    with open(model_dir / "tfidf_text.pkl", "wb") as f:
        pickle.dump(tfidf_text, f)
    print(f"Model saved to {model_dir}")


def predict(model_dir: str, table: dict, text: str) -> float:
    model_dir = Path(model_dir)
    clf = XGBClassifier()
    clf.load_model(str(model_dir / "xgb_model.json"))
    with open(model_dir / "tfidf_table.pkl", "rb") as f:
        tfidf_table = pickle.load(f)
    with open(model_dir / "tfidf_text.pkl", "rb") as f:
        tfidf_text = pickle.load(f)

    table_str = serialize_table(table)
    X_table = tfidf_table.transform([table_str])
    X_text = tfidf_text.transform([text])
    X = hstack([X_table, X_text])

    proba = clf.predict_proba(X)
    return proba[0, 1]


def main():
    parser = argparse.ArgumentParser(description="TF-IDF + XGBoost text-table classifier")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--table-json", type=str)
    parser.add_argument("--text", type=str)
    args = parser.parse_args()

    if args.train:
        if DEFAULT_CORPUS.exists():
            with open(DEFAULT_CORPUS, encoding="utf-8") as f:
                corpus = json.load(f)
            print(f"Loaded corpus: {len(corpus)} pairs")
        else:
            raise FileNotFoundError(f"No corpus at {DEFAULT_CORPUS}. Run text_table_classification.py --build-corpus first.")
        sample_size = 100_000
        if len(corpus) > sample_size:
            random.seed(42)
            corpus = random.sample(corpus, sample_size)
            print(f"Sampled {sample_size} pairs for training")
        train(corpus, args.model_dir)

    elif args.predict:
        if not args.table_json or not args.text:
            parser.error("--predict requires --table-json and --text")
        table = json.loads(args.table_json)
        score = predict(args.model_dir, table, args.text)
        print(f"Relevance score: {score:.4f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
