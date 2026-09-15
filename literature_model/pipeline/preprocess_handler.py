"""Lambda handler: preprocess page text (stop words + lemmatization)."""

import io
import json
import logging
import os

import boto3
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger()
logger.setLevel(logging.INFO)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

S3_BUCKET = os.environ.get("S3_BUCKET", "mineral-pipeline-pipeline")
INPUT_KEY = "training/page_text.parquet"
OUTPUT_KEY = "training/page_text_clean.parquet"


def preprocess_text(text):
    """Remove stopwords and lemmatize (mirrors pdf_extraction.preprocess_text)."""
    if not isinstance(text, str) or not text.strip():
        return ""
    words = word_tokenize(text)
    filtered = [lemmatizer.lemmatize(w.lower()) for w in words
                if w.lower() not in stop_words and w.isalpha()]
    return ' '.join(filtered)


def handler(event, context):
    s3 = boto3.client("s3")

    logger.info("Reading %s from %s", INPUT_KEY, S3_BUCKET)
    resp = s3.get_object(Bucket=S3_BUCKET, Key=INPUT_KEY)
    df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    logger.info("Loaded %d rows", len(df))

    df["text_clean"] = df["text"].apply(preprocess_text)
    df["clean_char_count"] = df["text_clean"].str.len()

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)

    s3.put_object(Bucket=S3_BUCKET, Key=OUTPUT_KEY, Body=buf.getvalue())
    logger.info("Wrote %s (%d rows)", OUTPUT_KEY, len(df))

    sample = df.iloc[0].to_dict() if len(df) > 0 else {}
    # Convert non-serializable values to strings
    sample = {k: str(v)[:200] for k, v in sample.items()}

    return {
        "statusCode": 200,
        "body": json.dumps({
            "rows": len(df),
            "output_key": OUTPUT_KEY,
            "sample": sample
        })
    }
