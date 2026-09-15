"""Cross-document similarity layer — runs on SageMaker CPU.

For each pair of documents, computes per-section cosine similarity using
TF-IDF vectors and keyword features. Documents with similar sections AND
similar flowsheet strategies get strong cross-document edges, enabling
the GNN to borrow signal from analogous projects.

Output: s3://<bucket>/graphs/cross_doc_edges.json
  {
    "<doc_id>": {
      "neighbors": [
        {
          "doc_id": "<neighbor_id>",
          "strategy_similarity": 0.85,    # stage set Jaccard
          "section_similarities": {       # per-section cosine
            "geology": 0.92,
            "climate": 0.78, ...
          },
          "overall_similarity": 0.83      # weighted combination
        }, ...
      ]
    }, ...
  }

Usage (SageMaker entry script):
    Hyperparameters: graphs-bucket, graphs-prefix, top-k
"""

import json
import logging
import os
import time
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TOP_K = 20  # Neighbors per document
MIN_SIMILARITY = 0.3  # Minimum overall similarity to keep an edge
MIN_CONTEXT_NODES = 4  # Minimum context sections to include a document
STRATEGY_WEIGHT = 0.4  # Weight of strategy similarity in overall score
SECTION_WEIGHT = 0.6   # Weight of section similarity in overall score

# Section groups ordered by expected influence on flowsheet design
SECTION_GROUPS = [
    "geology", "metallurgical_testing", "resource_estimate", "mining_method",
    "recovery_methods", "climate", "infrastructure", "economics",
    "environmental", "property", "history", "exploration", "drilling",
    "sample_analysis", "data_verification", "summary", "interpretation",
    "market", "references",
]

# Weights reflect each section's influence on flowsheet design decisions.
# Sections that directly determine process choices get higher weight.
SECTION_WEIGHTS = {
    "metallurgical_testing": 0.20,
    "geology": 0.15,
    "resource_estimate": 0.10,
    "mining_method": 0.10,
    "recovery_methods": 0.25,
    "climate": 0.08,
    "economics": 0.07,
    "infrastructure": 0.05,
    "environmental": 0.05,
    "sample_analysis": 0.02,
    "exploration": 0.02,
    "drilling": 0.02,
    "property": 0.01,
    "history": 0.01,
    "summary": 0.01,
    "interpretation": 0.005,
    "data_verification": 0.005,
    "market": 0.005,
    "references": 0.005,
}


SKIP_SUFFIXES = ("stage_vocab.json", "cross_doc_edges.json",
                  "graphs_summary.json", "checkpoint.json", "skip_list.json",
                  "attempt_marker.json")


def load_graphs_from_s3(s3, bucket, prefix):
    """Load all graph JSONs from S3 prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    graphs = {}
    stage_vocab = None
    skipped = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            if key.endswith("stage_vocab.json"):
                try:
                    resp = s3.get_object(Bucket=bucket, Key=key)
                    stage_vocab = json.loads(resp["Body"].read())
                except Exception as e:
                    logger.warning(f"Failed to load stage_vocab: {e}")
                continue
            if key.endswith(SKIP_SUFFIXES):
                continue

            try:
                resp = s3.get_object(Bucket=bucket, Key=key)
                g = json.loads(resp["Body"].read())
                if not isinstance(g, dict):
                    skipped += 1
                    continue
                doc_id = g.get("document_id", "")
                if not doc_id:
                    skipped += 1
                    continue
                # Skip docs with too few context sections to compare meaningfully
                n_ctx = sum(1 for n in g.get("nodes", [])
                            if n.get("type") == "context")
                if n_ctx < MIN_CONTEXT_NODES:
                    skipped += 1
                    continue
                graphs[doc_id] = g
            except Exception as e:
                logger.warning(f"Failed to load {key}: {e}")
                skipped += 1

    logger.info(f"Loaded {len(graphs)} graphs, skipped {skipped}, "
                f"stage vocab: {len(stage_vocab) if stage_vocab else 0}")
    return graphs, stage_vocab


def extract_section_vectors(graphs):
    """Build per-section feature matrices across all documents.

    Returns:
        doc_ids: list of document IDs (consistent ordering)
        section_vectors: {section_group: np.array of shape (n_docs, n_features)}
            where missing sections get zero vectors
        stage_vectors: np.array of shape (n_docs, vocab_size) — binary stage presence
    """
    doc_ids = sorted(graphs.keys())
    doc_idx = {did: i for i, did in enumerate(doc_ids)}
    n_docs = len(doc_ids)

    # First pass: determine feature keys (excluding tfidf) from any context node
    feature_keys = None
    tfidf_dim = 200
    for g in graphs.values():
        for n in g.get("nodes", []):
            if n.get("type") == "context" and n.get("features"):
                feats = n["features"]
                feature_keys = [k for k in feats if k != "tfidf"]
                tfidf_dim = len(feats.get("tfidf", []))
                break
        if feature_keys is not None:
            break

    if feature_keys is None:
        logger.error("No context nodes with features found in any graph")
        feature_keys = []
        tfidf_dim = 0

    feat_dim = len(feature_keys) + tfidf_dim
    logger.info(f"Feature dim per section: {feat_dim} "
                f"({len(feature_keys)} scalar + {tfidf_dim} tfidf)")

    # Build per-section matrices
    section_vectors = {}
    for sec in SECTION_GROUPS:
        section_vectors[sec] = np.zeros((n_docs, feat_dim), dtype=np.float32)

    # Collect all unique stage IDs for strategy vectors
    all_stages = set()
    for g in graphs.values():
        for n in g.get("nodes", []):
            if n.get("type") == "stage" and n.get("id"):
                all_stages.add(n["id"])
    stage_list = sorted(all_stages)
    stage_idx = {s: i for i, s in enumerate(stage_list)}
    stage_vectors = np.zeros((n_docs, max(len(stage_list), 1)), dtype=np.float32)

    # Fill matrices
    parse_errors = 0
    for doc_id, g in graphs.items():
        i = doc_idx[doc_id]

        for n in g.get("nodes", []):
            try:
                ntype = n.get("type", "")
                if ntype == "context":
                    sec = n.get("group", "")
                    if sec not in section_vectors:
                        continue
                    feats = n.get("features") or {}
                    vec = [feats.get(k, 0) for k in feature_keys]
                    vec.extend(feats.get("tfidf", [0.0] * tfidf_dim))
                    section_vectors[sec][i] = vec

                elif ntype == "stage":
                    sid = n.get("id", "")
                    if sid in stage_idx:
                        stage_vectors[i, stage_idx[sid]] = 1.0
            except Exception as e:
                parse_errors += 1
                if parse_errors <= 5:
                    logger.warning(f"Error parsing node in {doc_id}: {e}")

    if parse_errors:
        logger.warning(f"Total node parse errors: {parse_errors}")

    # Min-max normalize scalar columns so keyword counts (0-50+) don't
    # dominate TF-IDF features (0-1) in cosine similarity.
    # Use float64 to avoid overflow from large scalar values.
    n_scalar = len(feature_keys)
    for sec in SECTION_GROUPS:
        mat = section_vectors[sec]
        scalar_part = mat[:, :n_scalar].astype(np.float64)
        col_min = scalar_part.min(axis=0)
        col_max = scalar_part.max(axis=0)
        denom = col_max - col_min
        denom[denom == 0] = 1.0
        normalized = (scalar_part - col_min) / denom
        mat[:, :n_scalar] = normalized.astype(np.float32)
        np.nan_to_num(mat, copy=False)

    return doc_ids, section_vectors, stage_vectors


def compute_cross_doc_edges(doc_ids, section_vectors, stage_vectors,
                            top_k=TOP_K, min_similarity=MIN_SIMILARITY):
    """Compute per-section similarity and strategy similarity between all docs.

    Returns dict mapping each doc_id to its top-K neighbors with scores.
    """
    n_docs = len(doc_ids)
    logger.info(f"Computing similarities for {n_docs} documents...")

    # Strategy similarity: Jaccard via dot product on binary vectors
    # Jaccard = intersection / union = dot(a,b) / (|a| + |b| - dot(a,b))
    stage_norms = stage_vectors.sum(axis=1)  # (n_docs,)
    stage_dot = stage_vectors @ stage_vectors.T  # (n_docs, n_docs)
    union = stage_norms[:, None] + stage_norms[None, :] - stage_dot
    # Avoid divide by zero for docs with no stages
    strategy_sim = np.divide(stage_dot, union,
                             out=np.zeros_like(stage_dot),
                             where=union > 0)
    logger.info("  Strategy similarity computed")

    # Per-section cosine similarity
    section_sims = {}
    active_sections = []
    for sec in SECTION_GROUPS:
        mat = section_vectors[sec]
        # Only compute if section has non-zero vectors
        norms = np.linalg.norm(mat, axis=1)
        if norms.max() == 0:
            continue
        sim = cosine_similarity(mat)
        np.nan_to_num(sim, copy=False)
        # Zero out self-similarity and pairs where either doc is missing
        np.fill_diagonal(sim, 0)
        mask = np.outer(norms > 0, norms > 0)
        sim *= mask
        section_sims[sec] = sim
        active_sections.append(sec)

    logger.info(f"  Section similarities computed for {len(active_sections)} sections")

    # Weighted section similarity (only over sections both docs share)
    section_sum = np.zeros((n_docs, n_docs), dtype=np.float32)
    weight_sum = np.zeros((n_docs, n_docs), dtype=np.float32)
    for sec in active_sections:
        w = SECTION_WEIGHTS.get(sec, 0.01)
        sim = section_sims[sec]
        has_section = np.linalg.norm(section_vectors[sec], axis=1) > 0
        pair_mask = np.outer(has_section, has_section).astype(np.float32)
        section_sum += w * sim * pair_mask
        weight_sum += w * pair_mask

    mean_section_sim = np.divide(section_sum, weight_sum,
                                 out=np.zeros_like(section_sum),
                                 where=weight_sum > 0)

    overall = (STRATEGY_WEIGHT * strategy_sim +
               SECTION_WEIGHT * mean_section_sim)
    np.fill_diagonal(overall, 0)

    logger.info("  Overall similarity computed")

    # Build top-K neighbor lists
    result = {}
    edges_kept = 0
    for i, doc_id in enumerate(doc_ids):
        scores = overall[i]
        # Get top-K indices
        if top_k < n_docs:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
        else:
            top_indices = np.arange(n_docs)
        top_indices = top_indices[scores[top_indices] >= min_similarity]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        neighbors = []
        for j in top_indices:
            neighbor = {
                "doc_id": doc_ids[j],
                "strategy_similarity": round(float(strategy_sim[i, j]), 4),
                "overall_similarity": round(float(overall[i, j]), 4),
                "section_similarities": {},
            }
            for sec in active_sections:
                val = float(section_sims[sec][i, j])
                if val > 0:
                    neighbor["section_similarities"][sec] = round(val, 4)
            neighbors.append(neighbor)
            edges_kept += 1

        result[doc_id] = {"neighbors": neighbors}

    logger.info(f"  {edges_kept} cross-document edges kept "
                f"(avg {edges_kept / max(n_docs, 1):.1f} per doc)")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs-bucket", type=str, default="")
    parser.add_argument("--graphs-prefix", type=str, default="graphs/")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    cli_args, _ = parser.parse_known_args()

    t0 = time.time()

    output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    graphs_bucket = cli_args.graphs_bucket or os.environ.get("SM_HP_GRAPHS_BUCKET", "")
    graphs_prefix = cli_args.graphs_prefix or os.environ.get("SM_HP_GRAPHS_PREFIX", "graphs/")
    top_k = cli_args.top_k

    logger.info(f"Graphs: s3://{graphs_bucket}/{graphs_prefix}")
    logger.info(f"Top-K: {top_k}")

    import boto3
    s3 = boto3.client("s3")

    # 1. Load all graphs
    graphs, stage_vocab = load_graphs_from_s3(s3, graphs_bucket, graphs_prefix)

    # 2. Extract feature matrices
    doc_ids, section_vectors, stage_vectors = extract_section_vectors(graphs)
    del graphs  # Free memory

    logger.info(f"Stage vector dim: {stage_vectors.shape[1]} "
                f"({int(stage_vectors.sum())} total stage entries)")

    # 3. Compute cross-document edges
    result = compute_cross_doc_edges(doc_ids, section_vectors, stage_vectors,
                                     top_k=top_k)

    # 4. Write output locally (SageMaker will package as model.tar.gz)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cross_doc_edges.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    logger.info(f"Wrote {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)")

    # 5. Also upload directly to graphs/ prefix for easy access
    if graphs_bucket:
        s3.upload_file(output_path, graphs_bucket,
                       graphs_prefix + "cross_doc_edges.json")
        logger.info(f"Uploaded to s3://{graphs_bucket}/{graphs_prefix}cross_doc_edges.json")

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
