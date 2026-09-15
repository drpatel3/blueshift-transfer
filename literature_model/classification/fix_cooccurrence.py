"""Fix co-occurrence weights for graphs where per-doc TF-IDF produced all-1.0 weights.

Recomputes section_cooccurrence edges using the stored global TF-IDF vectors
already in each node's features. Runs locally — no GPU needed.

Usage: python classification/fix_cooccurrence.py
"""

import json
import logging
import numpy as np
import boto3
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from config import SAGEMAKER_BUCKET

BUCKET = SAGEMAKER_BUCKET
PREFIX = "graphs/"
COOCCURRENCE_THRESHOLD = 0.15


def main():
    s3 = boto3.client("s3")
    
    paginator = s3.get_paginator("list_objects_v2")
    skip = ("stage_vocab.json", "cross_doc_edges.json", "graphs_summary.json",
            "checkpoint.json", "skip_list.json", "attempt_marker.json")

    fixed = 0
    checked = 0

    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json") or key.endswith(skip):
                continue

            resp = s3.get_object(Bucket=BUCKET, Key=key)
            g = json.loads(resp["Body"].read())
            if not isinstance(g, dict) or not g.get("document_id"):
                continue

            checked += 1

            # Check if this graph has all-1.0 co-occurrence weights
            cooc_edges = [e for e in g.get("edges", [])
                          if e.get("type") == "section_cooccurrence"]
            if not cooc_edges:
                continue
            if not all(e["weight"] == 1.0 for e in cooc_edges):
                continue

            # This graph needs fixing — recompute from stored global TF-IDF
            context_nodes = [n for n in g["nodes"] if n.get("type") == "context"]
            if len(context_nodes) < 2:
                continue

            # Extract stored TF-IDF vectors
            tfidf_vectors = []
            node_ids = []
            for n in context_nodes:
                tfidf = n.get("features", {}).get("tfidf", [])
                tfidf_vectors.append(tfidf)
                node_ids.append(n["id"])

            tfidf_matrix = np.array(tfidf_vectors, dtype=np.float32)
            sim_matrix = cosine_similarity(tfidf_matrix)

            # Remove old co-occurrence edges
            other_edges = [e for e in g["edges"]
                           if e.get("type") != "section_cooccurrence"]

            # Add new co-occurrence edges with correct weights
            new_cooc = []
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    sim = float(sim_matrix[i, j])
                    if sim > COOCCURRENCE_THRESHOLD:
                        new_cooc.append({
                            "source": node_ids[i],
                            "target": node_ids[j],
                            "type": "section_cooccurrence",
                            "weight": round(sim, 4),
                        })

            g["edges"] = other_edges + new_cooc

            # Upload fixed graph
            s3.put_object(
                Bucket=BUCKET, Key=key,
                Body=json.dumps(g, ensure_ascii=False).encode("utf-8"),
                ContentType="application/json",
            )
            fixed += 1
            if fixed % 50 == 0:
                logger.info(f"Fixed {fixed} graphs so far...")

    logger.info(f"Done. Checked {checked} graphs, fixed {fixed}.")


if __name__ == "__main__":
    main()
