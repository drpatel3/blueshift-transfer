"""Consolidate graph JSONs from SageMaker job outputs into a single S3 prefix.

Reads all graph-build job model.tar.gz archives, deduplicates by document_id,
uploads each unique graph as an individual JSON to s3://<bucket>/graphs/<doc_id>.json,
then deletes the old job folders.

Usage:
    python classification/consolidate_graphs.py
    python classification/consolidate_graphs.py --dry-run
"""

import argparse
import io
import json
import tarfile
import tempfile
import os

import boto3

from config import SAGEMAKER_BUCKET

BUCKET = SAGEMAKER_BUCKET
JOB_PREFIX = "graph-build/"
CONSOLIDATED_PREFIX = "graphs/"


def consolidate(dry_run=False):
    s3 = boto3.client("s3")

    # 1. Find all job model.tar.gz files
    paginator = s3.get_paginator("list_objects_v2")
    model_keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=JOB_PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("/output/model.tar.gz"):
                model_keys.append(obj["Key"])

    print(f"Found {len(model_keys)} job outputs")

    # 2. Extract unique graphs by document_id (keep latest version)
    seen_doc_ids = {}  # doc_id -> (graph_dict, source_key)
    stage_vocab = None

    for i, key in enumerate(model_keys):
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        body = resp["Body"].read()
        tf = tarfile.open(fileobj=io.BytesIO(body), mode="r:gz")

        for m in tf.getmembers():
            if not m.name.endswith(".json") or not m.name.startswith("graphs/"):
                continue
            if m.name == "graphs/":
                continue

            fname = m.name.split("/")[-1]

            # Grab stage_vocab from latest job
            if fname == "stage_vocab.json":
                f = tf.extractfile(m)
                stage_vocab = json.loads(f.read())
                continue

            # Skip metadata files
            if fname in ("checkpoint.json", "graphs_summary.json",
                         "tfidf_vectorizer.pkl"):
                continue

            f = tf.extractfile(m)
            if f is None:
                continue
            graph = json.loads(f.read())
            doc_id = graph.get("document_id", "")
            if not doc_id:
                continue

            # Always overwrite — later jobs may have better data
            seen_doc_ids[doc_id] = graph

        tf.close()
        del body

        if (i + 1) % 10 == 0:
            print(f"  Scanned {i + 1}/{len(model_keys)} jobs, "
                  f"{len(seen_doc_ids)} unique graphs")

    print(f"\nTotal unique graphs: {len(seen_doc_ids)}")
    if stage_vocab:
        print(f"Stage vocab: {len(stage_vocab)} stages")

    if dry_run:
        print("\n[DRY RUN] Would upload graphs and delete old jobs. Exiting.")
        return

    # 3. Upload each graph as individual JSON to consolidated prefix
    print(f"\nUploading {len(seen_doc_ids)} graphs to "
          f"s3://{BUCKET}/{CONSOLIDATED_PREFIX}")

    for i, (doc_id, graph) in enumerate(seen_doc_ids.items()):
        key = f"{CONSOLIDATED_PREFIX}{doc_id}.json"
        body = json.dumps(graph, ensure_ascii=False).encode("utf-8")
        s3.put_object(Bucket=BUCKET, Key=key, Body=body,
                      ContentType="application/json")
        if (i + 1) % 100 == 0:
            print(f"  Uploaded {i + 1}/{len(seen_doc_ids)}")

    # Upload stage_vocab
    if stage_vocab:
        s3.put_object(
            Bucket=BUCKET,
            Key=f"{CONSOLIDATED_PREFIX}stage_vocab.json",
            Body=json.dumps(stage_vocab, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        print("  Uploaded stage_vocab.json")

    print(f"Upload complete.")

    # 4. Delete all old job folders
    print(f"\nDeleting old job outputs under s3://{BUCKET}/{JOB_PREFIX}")
    delete_count = 0
    for page in paginator.paginate(Bucket=BUCKET, Prefix=JOB_PREFIX):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            s3.delete_objects(Bucket=BUCKET, Delete={"Objects": objects})
            delete_count += len(objects)
            if delete_count % 500 == 0:
                print(f"  Deleted {delete_count} objects")

    print(f"  Deleted {delete_count} objects total")
    print(f"\nDone. Graphs consolidated at s3://{BUCKET}/{CONSOLIDATED_PREFIX}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and count without uploading or deleting")
    args = parser.parse_args()
    consolidate(dry_run=args.dry_run)
