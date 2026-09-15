"""DeBERTa-based text-table relevance classifier.

Learns which text passages are relevant to a given table extracted from
NI 43-101 mining technical reports. Structured for easy migration to
AWS SageMaker (hyperparameters via argparse, standard I/O dirs).

Usage:
    python text_table_classification.py --build-corpus --results-dir ../tmp/results
    python text_table_classification.py --train
    python text_table_classification.py --predict --table-json '{"caption":"...","headers":[],"rows":[]}' --text "some passage"
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

MODEL_NAME = "microsoft/deberta-v3-base"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = SCRIPT_DIR / "corpus.json"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "model"


# ---------------------------------------------------------------------------
# 1. clean_text
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lemmatize and remove stopwords, keeping numbers."""
    words = text.split()
    filtered = [_lemmatizer.lemmatize(w.lower()) for w in words
                if w.lower() not in _stop_words]
    return " ".join(filtered)


# ---------------------------------------------------------------------------
# 2. serialize_table
# ---------------------------------------------------------------------------

def serialize_table(table: dict) -> str:
    """Flatten a table dict into a single string for model input."""
    caption = table.get("caption", "") or ""
    headers = " ".join(table.get("headers", []) or [])
    rows = table.get("rows", []) or []
    flat_rows = " ".join(
        " ".join(cell for cell in row) for row in rows[:5]
    )
    return f"{caption} | {headers} | {flat_rows}".strip()


# ---------------------------------------------------------------------------
# 2. extract_table_number
# ---------------------------------------------------------------------------

_TABLE_NUM_RE = re.compile(
    r"(?:Table|Tabla|Tableau)\s+(\d+[\.\-]\d+(?:[\.\-]\d+)*)",
    re.IGNORECASE,
)


def extract_table_number(caption: str) -> str | None:
    """Pull a table number pattern (e.g. 'Table 1-1') from caption text."""
    if not caption:
        return None
    m = _TABLE_NUM_RE.search(caption)
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# 3. build_corpus
# ---------------------------------------------------------------------------

def build_corpus(results_dir: str) -> list[dict]:
    """Read all JSON results and build (table_str, text, label) pairs.

    Positives: text that references the same table number, plus text_before/after.
    Negatives: random text from a *different* document.  ~1:1 with positives.
    """
    results_path = Path(results_dir)
    json_files = sorted(results_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {results_path}")

    # Load all documents -------------------------------------------------------
    docs: list[dict] = []
    for fp in json_files:
        with open(fp, encoding="utf-8") as f:
            try:
                docs.append(json.load(f))
            except json.JSONDecodeError:
                continue

    # Collect all page_text per document index for negative sampling
    all_texts_by_doc: list[list[str]] = []
    for doc in docs:
        texts = [pt["text"] for pt in doc.get("page_text", []) if pt.get("text")]
        all_texts_by_doc.append(texts)

    positives: list[dict] = []
    tables_for_negatives: list[tuple[str, int]] = []  # (table_str, doc_idx)

    for doc_idx, doc in enumerate(docs):
        tables = doc.get("tables", []) or []
        page_texts = doc.get("page_text", []) or []

        for table in tables:
            table_str = serialize_table(table)
            if not table_str.strip():
                continue

            table_num = extract_table_number(table.get("caption", ""))
            found_positive = False

            # Match by table number in page text
            if table_num:
                pattern = re.compile(re.escape(table_num), re.IGNORECASE)
                for pt in page_texts:
                    text = pt.get("text", "")
                    if text and pattern.search(text):
                        positives.append({"table_str": table_str, "text": text, "label": 1})
                        found_positive = True

            # text_before / text_after as positives
            for field in ("text_before", "text_after"):
                val = table.get(field, "")
                if val and len(val.strip()) > 20:
                    positives.append({"table_str": table_str, "text": val, "label": 1})
                    found_positive = True

            if found_positive:
                tables_for_negatives.append((table_str, doc_idx))

    # Build negatives (1:1) ----------------------------------------------------
    negatives: list[dict] = []
    doc_indices = list(range(len(docs)))
    for table_str, doc_idx in tables_for_negatives:
        # pick a different document
        other_indices = [i for i in doc_indices if i != doc_idx and all_texts_by_doc[i]]
        if not other_indices:
            continue
        other_idx = random.choice(other_indices)
        neg_text = random.choice(all_texts_by_doc[other_idx])
        negatives.append({"table_str": table_str, "text": neg_text, "label": 0})

    # Balance to ~1:1
    if len(negatives) > len(positives):
        negatives = random.sample(negatives, len(positives))
    elif len(positives) > len(negatives) and negatives:
        extra = random.choices(negatives, k=len(positives) - len(negatives))
        negatives.extend(extra)

    # Apply lemmatization and stopword removal
    print("Cleaning text (lemmatization + stopword removal)...")
    for item in positives + negatives:
        item["table_str"] = clean_text(item["table_str"])
        item["text"] = clean_text(item["text"])

    corpus = positives + negatives
    random.shuffle(corpus)

    # Persist
    DEFAULT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_CORPUS, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    print(f"Corpus built: {len(positives)} positives, {len(negatives)} negatives, {len(corpus)} total")
    print(f"Saved to {DEFAULT_CORPUS}")
    return corpus


# ---------------------------------------------------------------------------
# 4. TableTextDataset
# ---------------------------------------------------------------------------

class TableTextDataset(Dataset):
    def __init__(self, corpus: list[dict], tokenizer, max_length: int = 128):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        item = self.corpus[idx]
        enc = self.tokenizer(
            item["table_str"],
            item["text"],
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": item["label"],
        }


def collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [0] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["label"])
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# 5. train
# ---------------------------------------------------------------------------

def train(
    corpus: list[dict],
    model_dir: str,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 2e-5,
):
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    print(f"  Model loaded ({time.time() - t0:.1f}s)")

    print("Tokenizing dataset...")
    dataset = TableTextDataset(corpus, tokenizer)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"  Train: {train_size}, Val: {val_size} ({time.time() - t0:.1f}s)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)
    num_batches = len(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_f1 = 0.0
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        log_interval = max(1, num_batches // 10)
        for i, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()
            if i % log_interval == 0:
                print(f"  Epoch {epoch} — batch {i}/{num_batches}, loss: {outputs.loss.item():.4f} ({time.time() - t0:.1f}s)")
        avg_loss = total_loss / num_batches

        # --- Validate ---
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                preds = outputs.logits.argmax(dim=-1).cpu().tolist()
                labels = batch["labels"].tolist()
                all_preds.extend(preds)
                all_labels.extend(labels)

        tp = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(all_preds, all_labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(all_preds, all_labels))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) if all_labels else 0.0

        print(f"Epoch {epoch}/{epochs} — loss: {avg_loss:.4f}, val_acc: {val_acc:.4f}, "
              f"f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")

        if f1 >= best_f1:
            best_f1 = f1
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"  -> Saved best model (f1={f1:.4f})")

    print(f"Training complete. Best f1: {best_f1:.4f}")


# ---------------------------------------------------------------------------
# 6. predict
# ---------------------------------------------------------------------------

def predict(model_dir: str, table: dict, text: str) -> float:
    """Return relevance probability for a (table, text) pair."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    table_str = serialize_table(table)
    enc = tokenizer(
        table_str,
        text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        )
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[0, 1].item()


# ---------------------------------------------------------------------------
# 7. main — argparse CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DeBERTa text-table relevance classifier")

    parser.add_argument("--build-corpus", action="store_true", help="Build training corpus from JSON results")
    parser.add_argument("--results-dir", type=str, default="../tmp/results", help="Directory with result JSON files")

    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--model-dir", type=str,
        default=os.environ.get("SM_MODEL_DIR", str(DEFAULT_MODEL_DIR)),
        help="Model output directory")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("SM_HP_EPOCHS", "3")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("SM_HP_BATCH_SIZE", "8")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("SM_HP_LR", "2e-5")))
    parser.add_argument("--sample-size", type=int, default=50_000,
        help="Corpus pairs to sample (0 = use all)")

    parser.add_argument("--predict", action="store_true", help="Run prediction on a single pair")
    parser.add_argument("--table-json", type=str, help="Table dict as JSON string")
    parser.add_argument("--text", type=str, help="Text passage for prediction")

    parser.add_argument("--score-batch", action="store_true",
        help="Score all pairs from input file (SageMaker batch mode)")
    parser.add_argument("--input-file", type=str, help="Path to pairs JSON for scoring")
    parser.add_argument("--output-file", type=str, default="scored_results.json",
        help="Output filename for scored results")

    args = parser.parse_args()

    # Auto-detect SageMaker: default to training when SM_CHANNEL_TRAINING is set
    if os.environ.get("SM_CHANNEL_TRAINING") and not (args.build_corpus or args.predict or args.score_batch):
        args.train = True

    if args.build_corpus:
        build_corpus(args.results_dir)

    elif args.train:
        # Resolve corpus: SageMaker channel > local default > build from results
        sm_train = os.environ.get("SM_CHANNEL_TRAINING")
        if sm_train:
            corpus_path = Path(sm_train) / "corpus.json"
            print(f"Loading corpus from SageMaker channel: {corpus_path}")
        else:
            corpus_path = DEFAULT_CORPUS

        if corpus_path.exists():
            with open(corpus_path, encoding="utf-8") as f:
                corpus = json.load(f)
            print(f"Loaded corpus: {len(corpus)} pairs")
        else:
            print(f"No corpus found at {corpus_path}, building from {args.results_dir}...")
            corpus = build_corpus(args.results_dir)

        if args.sample_size > 0 and len(corpus) > args.sample_size:
            random.seed(42)
            corpus = random.sample(corpus, args.sample_size)
            print(f"Sampled {args.sample_size} pairs for training")
        train(corpus, args.model_dir, args.epochs, args.batch_size, args.lr)

    elif args.score_batch:
        # Resolve input: SageMaker channel or explicit path
        sm_test = os.environ.get("SM_CHANNEL_TEST")
        input_path = Path(sm_test) / "pairs.json" if sm_test else Path(args.input_file)
        if not input_path.exists():
            parser.error(f"Input file not found: {input_path}")

        # Resolve model: SageMaker channel or explicit path
        sm_model = os.environ.get("SM_CHANNEL_MODEL")
        model_path = sm_model if sm_model else args.model_dir

        # Resolve output: SM_MODEL_DIR gets auto-uploaded to S3
        sm_output = os.environ.get("SM_MODEL_DIR", ".")
        output_path = Path(sm_output) / args.output_file

        print(f"Loading model from {model_path}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()

        print(f"Loading pairs from {input_path}...")
        with open(input_path, encoding="utf-8") as f:
            pairs = json.load(f)
        print(f"Scoring {len(pairs)} pairs on {device}...")

        scored = []
        batch_size = args.batch_size
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            encodings = tokenizer(
                [p["table_str"] for p in batch_pairs],
                [p["text"] for p in batch_pairs],
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = model(
                    input_ids=encodings["input_ids"].to(device),
                    attention_mask=encodings["attention_mask"].to(device),
                )
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().tolist()
            for pair, score in zip(batch_pairs, probs):
                scored.append({
                    "table_str": pair["table_str"],
                    "text": pair["text"],
                    "score": score,
                })
            if (i // batch_size) % 100 == 0:
                print(f"  Scored {min(i + batch_size, len(pairs))}/{len(pairs)}")

        Path(sm_output).mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scored, f, ensure_ascii=False)
        print(f"Wrote {len(scored)} scored pairs to {output_path}")

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
