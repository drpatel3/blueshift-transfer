"""SageMaker entry script: evaluate LLM flowsheet prediction against ground truth.

Supports three modes:
  - standalone: LLM predicts from graph context only
  - hybrid: XGBoost predicts first, LLM validates and refines
  - xgboost: XGBoost only (baseline comparison)

Reports macro-F1, micro-F1, per-stage F1, and per-doc comparisons.

Usage (SageMaker entry script):
    Hyperparameters:
        graphs-bucket, graphs-prefix, provider, model-id, top-k, max-docs, mode
"""

import argparse
import json
import logging
import os
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import boto3
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LLM Flowsheet Eval")
    parser.add_argument("--graphs-bucket", type=str, required=True)
    parser.add_argument("--graphs-prefix", type=str, default="graphs/")
    parser.add_argument("--provider", type=str, default="bedrock")
    parser.add_argument("--model-id", type=str, default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["standalone", "hybrid", "xgboost"])
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Which split to evaluate: val (default) or test")
    args = parser.parse_args()

    output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(output_dir, exist_ok=True)

    s3 = boto3.client("s3")
    bucket = args.graphs_bucket

    # Setup LLM provider (skip for xgboost-only mode)
    if args.mode != "xgboost":
        os.environ["LLM_PREDICTOR_PROVIDER"] = args.provider
        if args.model_id:
            os.environ["LLM_PREDICTOR_MODEL"] = args.model_id
        _setup_llm_provider(args)

    # 1. Load all graphs
    logger.info("Loading graphs from S3...")
    graphs = _load_all_graphs(s3, bucket, args.graphs_prefix)
    logger.info(f"Loaded {len(graphs)} graphs")

    # 2. Load cross-doc edges
    cross_doc_data = {}
    try:
        resp = s3.get_object(Bucket=bucket,
                             Key=f"{args.graphs_prefix}cross_doc_edges.json")
        cross_doc_data = json.loads(resp["Body"].read())
        logger.info(f"Loaded {len(cross_doc_data)} cross-doc entries")
    except Exception:
        pass

    # 3. Build transition counts + stage vocab
    transition_counts = _build_transition_counts(graphs)
    from llm_predictor import CANONICAL_STAGES
    stage_vocab, raw_to_canonical = _build_stage_vocab(graphs, min_support=3)
    stage_to_idx = {s: i for i, s in enumerate(stage_vocab)}
    S = len(stage_vocab)
    logger.info(f"Stage vocab: {S} stages, {len(transition_counts)} transitions")

    # 4. Identify labeled docs
    labeled_graphs = []
    for graph in graphs:
        doc_stages = set()
        for node in graph.get("nodes", []):
            if node.get("type") == "stage":
                raw_id = node.get("stage_id", node.get("id", "").replace("stg_", ""))
                canonical = raw_to_canonical.get(raw_id)
                if canonical and canonical in stage_to_idx:
                    doc_stages.add(canonical)
        if doc_stages:
            labeled_graphs.append((graph, doc_stages))

    logger.info(f"Found {len(labeled_graphs)} labeled documents")

    # 5. Load XGBoost predictions (test + val + OOF from training checkpoint)
    xgb_preds_by_doc = {}
    test_doc_ids_from_split = set()
    val_doc_ids_from_split = set()
    if args.mode in ("hybrid", "xgboost"):
        xgb_preds_by_doc, test_doc_ids_from_split, val_doc_ids_from_split = \
            _load_xgb_predictions(s3, bucket)

    # Filter to the requested split FIRST, then apply max_docs
    if args.split == "val" and val_doc_ids_from_split:
        eval_doc_ids = val_doc_ids_from_split
        logger.info(f"Filtering to {len(eval_doc_ids)} val docs")
    elif args.split == "test" and test_doc_ids_from_split:
        eval_doc_ids = test_doc_ids_from_split
        logger.info(f"Filtering to {len(eval_doc_ids)} test docs")
    else:
        eval_doc_ids = test_doc_ids_from_split | val_doc_ids_from_split
        logger.info(f"No split filter — using all {len(eval_doc_ids)} held-out docs")

    if eval_doc_ids:
        labeled_graphs = [
            (g, stages) for g, stages in labeled_graphs
            if g.get("doc_id", g.get("document_id", "")) in eval_doc_ids
        ]
        logger.info(f"Eval set: {len(labeled_graphs)} docs ({args.split})")

    # Apply max_docs limit AFTER split filter
    if args.max_docs > 0 and len(labeled_graphs) > args.max_docs:
        labeled_graphs = labeled_graphs[:args.max_docs]
        logger.info(f"Limiting to {args.max_docs} docs")

    # 7. Run eval loop
    from llm_predictor import predict_from_graph, extract_params_from_graph

    # Build exclusion set: ALL val + test docs (never use as LLM examples)
    all_eval_doc_ids = test_doc_ids_from_split | val_doc_ids_from_split
    # Also add any docs being evaluated in this run
    for graph, _ in labeled_graphs:
        did = graph.get("doc_id", graph.get("document_id", ""))
        if did:
            all_eval_doc_ids.add(did)

    logger.info(f"Excluding {len(all_eval_doc_ids)} held-out docs from LLM neighbor retrieval "
                f"(test={len(test_doc_ids_from_split)}, val={len(val_doc_ids_from_split)})")

    # Stages to exclude from F1 scoring (inconsistent in ground truth)
    EXCLUDE_FROM_SCORING = {"input", "output", "concentrate_product"}
    scoring_indices = [i for i, s in enumerate(stage_vocab) if s not in EXCLUDE_FROM_SCORING]
    logger.info(f"Scoring on {len(scoring_indices)}/{S} stages "
                f"(excluding {EXCLUDE_FROM_SCORING & set(stage_vocab)})")

    N = len(labeled_graphs)
    y_true = np.zeros((N, S), dtype=int)
    y_pred_final = np.zeros((N, S), dtype=int)
    y_pred_xgb = np.zeros((N, S), dtype=int)
    per_doc_results = []

    t0 = time.time()
    for i, (graph, ground_truth) in enumerate(labeled_graphs):
        doc_id = graph.get("doc_id", graph.get("document_id", f"doc_{i}"))

        # Ground truth
        for stage in ground_truth:
            if stage in stage_to_idx:
                y_true[i, stage_to_idx[stage]] = 1

        # --- XGBoost prediction (from pre-computed test/val/OOF) ---
        xgb_result = None
        xgb_predicted = set()
        if doc_id in xgb_preds_by_doc:
            xgb_entry = xgb_preds_by_doc[doc_id]
            xgb_predicted = set(xgb_entry.get("xgb_stages", []))
            xgb_result = {
                "stages": xgb_entry.get("xgb_stages", []),
                "stage_probabilities": xgb_entry.get("xgb_predicted", {}),
                "connections": [],
            }

        # Record XGBoost-only prediction
        for stage in xgb_predicted:
            if stage in stage_to_idx:
                y_pred_xgb[i, stage_to_idx[stage]] = 1

        # --- Final prediction (depends on mode) ---
        result = {}
        predicted = set()

        if args.mode == "xgboost":
            predicted = xgb_predicted
            result = xgb_result or {}

        elif args.mode == "standalone":
            try:
                result = predict_from_graph(
                    graph=graph, all_graphs=graphs,
                    cross_doc_data=cross_doc_data,
                    transition_counts=transition_counts,
                    top_k=args.top_k, exclude_doc_id=doc_id,
                    exclude_doc_ids=all_eval_doc_ids)
                predicted = set(result.get("stages", []))
            except Exception as e:
                logger.error(f"LLM failed for {doc_id[:20]}: {e}")

        elif args.mode == "hybrid":
            # Pass XGBoost result to LLM for validation
            try:
                result = predict_from_graph(
                    graph=graph, all_graphs=graphs,
                    cross_doc_data=cross_doc_data,
                    transition_counts=transition_counts,
                    top_k=args.top_k, exclude_doc_id=doc_id,
                    exclude_doc_ids=all_eval_doc_ids,
                    xgb_result=xgb_result)
                predicted = set(result.get("stages", []))
            except Exception as e:
                logger.error(f"Hybrid failed for {doc_id[:20]}: {e}")
                predicted = xgb_predicted  # Fall back to XGBoost

        # Record final prediction
        for stage in predicted:
            if stage in stage_to_idx:
                y_pred_final[i, stage_to_idx[stage]] = 1

        # Per-doc comparison (excluding input/output from scoring)
        gt_scored = ground_truth - EXCLUDE_FROM_SCORING
        pred_scored = predicted - EXCLUDE_FROM_SCORING
        correct = gt_scored & pred_scored
        missed = gt_scored - pred_scored
        false_pos = pred_scored - gt_scored
        xgb_scored = xgb_predicted - EXCLUDE_FROM_SCORING
        xgb_correct = gt_scored & xgb_scored

        gt_sequence = graph.get("stage_sequence", sorted(ground_truth))
        gt_connections = [
            f"{e['source'].replace('stg_', '')} -> {e['target'].replace('stg_', '')}"
            for e in graph.get("edges", []) if e.get("type") == "stage_transition"
        ]
        doc_params = extract_params_from_graph(graph)
        n_gt = len(gt_scored)
        n_pred = len(pred_scored)
        n_correct = len(correct)
        doc_f1 = (2 * n_correct / (n_gt + n_pred)) if (n_gt + n_pred) > 0 else 0.0

        per_doc_results.append({
            "doc_id": doc_id,
            "source_params": doc_params,
            "ground_truth": sorted(ground_truth),
            "ground_truth_sequence": gt_sequence,
            "ground_truth_connections": gt_connections,
            "xgboost_predicted": sorted(xgb_predicted),
            "xgboost_correct": len(xgb_correct),
            "final_predicted": sorted(predicted),
            "final_connections": [
                f"{c['from']} -> {c['to']}" for c in result.get("connections", [])
            ],
            "correct": sorted(correct),
            "missed": sorted(missed),
            "false_positive": sorted(false_pos),
            "doc_f1": round(doc_f1, 4),
            "reasoning": result.get("reasoning", {}),
        })

        elapsed = time.time() - t0
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (N - i - 1)
        logger.info(
            f"[{i+1}/{N}] {doc_id[:20]}... "
            f"GT={n_gt} XGB={len(xgb_predicted)} Final={n_pred} "
            f"Correct={n_correct} Missed={len(missed)} FP={len(false_pos)} "
            f"F1={doc_f1:.3f}  ({elapsed:.0f}s, ~{remaining:.0f}s left)")

    total_time = time.time() - t0

    # 8. Compute metrics (excluding input/output stages from scoring)
    metrics = {}
    si = scoring_indices
    for label, y_p in [("final", y_pred_final), ("xgboost", y_pred_xgb)]:
        m_f1 = f1_score(y_true[:, si], y_p[:, si], average="macro", zero_division=0)
        i_f1 = f1_score(y_true[:, si], y_p[:, si], average="micro", zero_division=0)
        prec = precision_score(y_true[:, si], y_p[:, si], average="macro", zero_division=0)
        rec = recall_score(y_true[:, si], y_p[:, si], average="macro", zero_division=0)
        metrics[label] = {
            "macro_f1": round(float(m_f1), 4),
            "micro_f1": round(float(i_f1), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
        }

    # Per-stage for final (excluding input/output)
    per_stage = {}
    for j, stage in enumerate(stage_vocab):
        if stage in EXCLUDE_FROM_SCORING:
            continue
        support = int(y_true[:, j].sum())
        if support == 0:
            continue
        per_stage[stage] = {
            "f1": round(float(f1_score(y_true[:, j], y_pred_final[:, j], zero_division=0)), 4),
            "precision": round(float(precision_score(y_true[:, j], y_pred_final[:, j], zero_division=0)), 4),
            "recall": round(float(recall_score(y_true[:, j], y_pred_final[:, j], zero_division=0)), 4),
            "support": support,
        }

    results = {
        "mode": args.mode,
        "metrics": metrics,
        "per_stage": dict(sorted(per_stage.items(), key=lambda x: x[1]["f1"], reverse=True)),
        "per_doc": per_doc_results,
        "comparison": {
            "xgboost_only": metrics.get("xgboost", {}),
            "final": metrics.get("final", {}),
            "xgboost_baseline_macro_f1": 0.6261,
        },
        "config": {
            "mode": args.mode,
            "provider": os.environ.get("LLM_PREDICTOR_PROVIDER", "n/a"),
            "model": os.environ.get("LLM_PREDICTOR_MODEL", "n/a"),
            "top_k": args.top_k,
            "max_docs": args.max_docs,
            "num_docs_evaluated": N,
            "total_time_seconds": round(total_time, 1),
        },
    }

    result_path = os.path.join(output_dir, "llm_eval_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    final = metrics["final"]
    xgb = metrics.get("xgboost", {})
    logger.info(f"\n{'=' * 60}")
    logger.info(f"EVAL RESULTS — {args.mode} mode, {N} docs, {total_time:.0f}s")
    logger.info(f"{'=' * 60}")
    logger.info(f"  XGBoost-only:  macro-F1={xgb.get('macro_f1', 'n/a')}")
    logger.info(f"  {args.mode:13s}:  macro-F1={final['macro_f1']}  "
                f"P={final['precision']}  R={final['recall']}")
    logger.info(f"  XGBoost baseline (5-fold CV): 0.6261")


# ---------------------------------------------------------------------------
# XGBoost OOF prediction loader
# ---------------------------------------------------------------------------

def _load_xgb_predictions(s3, bucket: str) -> tuple:
    """Load pre-computed XGBoost predictions from training checkpoint.

    Returns:
        (predictions_by_doc_id, test_doc_ids, val_doc_ids)
        predictions_by_doc_id: dict mapping doc_id -> prediction entry
        test_doc_ids: set of doc IDs in the test split
        val_doc_ids: set of doc IDs in the validation split
    """
    paginator = s3.get_paginator("list_objects_v2")

    xgb_tars = []
    for page in paginator.paginate(Bucket=bucket, Prefix="xgb-stages/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("model.tar.gz"):
                xgb_tars.append(obj["Key"])

    if not xgb_tars:
        logger.warning("No XGBoost checkpoint found in S3")
        return {}, set(), set()

    model_key = sorted(xgb_tars)[-1]
    logger.info(f"Loading XGBoost predictions from s3://{bucket}/{model_key}")

    tmpdir = tempfile.mkdtemp(prefix="xgb_preds_")
    tar_path = os.path.join(tmpdir, "model.tar.gz")
    s3.download_file(bucket, model_key, tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(tmpdir)

    all_preds = {}
    test_doc_ids = set()
    val_doc_ids = set()

    # Load test predictions (held-out, never seen during training)
    test_path = os.path.join(tmpdir, "test_predictions.json")
    if os.path.exists(test_path):
        with open(test_path) as f:
            test_data = json.load(f)
        for entry in test_data.get("per_doc", []):
            all_preds[entry["doc_id"]] = entry
            test_doc_ids.add(entry["doc_id"])
        logger.info(f"Loaded {len(test_doc_ids)} test predictions "
                    f"(macro-F1={test_data['metrics']['macro_f1']})")

    # Load val predictions
    val_path = os.path.join(tmpdir, "val_predictions.json")
    if os.path.exists(val_path):
        with open(val_path) as f:
            val_data = json.load(f)
        for entry in val_data.get("per_doc", []):
            all_preds[entry["doc_id"]] = entry
            val_doc_ids.add(entry["doc_id"])
        logger.info(f"Loaded {len(val_doc_ids)} val predictions "
                    f"(macro-F1={val_data['metrics']['macro_f1']})")

    # Also load OOF (training) predictions as fallback
    oof_path = os.path.join(tmpdir, "oof_predictions.json")
    if os.path.exists(oof_path):
        with open(oof_path) as f:
            oof_list = json.load(f)
        for entry in oof_list:
            if entry["doc_id"] not in all_preds:
                all_preds[entry["doc_id"]] = entry
        logger.info(f"Total predictions: {len(all_preds)} docs")

    return all_preds, test_doc_ids, val_doc_ids


# ---------------------------------------------------------------------------
# Graph/data helpers
# ---------------------------------------------------------------------------

def _setup_llm_provider(args):
    """Verify Bedrock access up front; fail fast if not available."""
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    model_id = args.model_id or "us.anthropic.claude-sonnet-4-20250514-v1:0"
    try:
        client.invoke_model(
            modelId=model_id, contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            }))
        logger.info("Bedrock access verified")
    except Exception as e:
        raise RuntimeError(
            f"Bedrock access check failed ({e}). Ensure Anthropic models are "
            "enabled and the execution role has bedrock:InvokeModel."
        )


def _load_all_graphs(s3, bucket: str, prefix: str) -> list:
    graphs = []
    paginator = s3.get_paginator("list_objects_v2")
    skip = {"cross_doc_edges.json", "stage_vocab.json", "graphs_summary.json",
            "checkpoint.json", "inference_index.json"}
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            fname = key.split("/")[-1]
            if not key.endswith(".json") or fname in skip:
                continue
            try:
                resp = s3.get_object(Bucket=bucket, Key=key)
                graph = json.loads(resp["Body"].read())
                doc_id = graph.get("document_id", fname.replace(".json", ""))
                graph["doc_id"] = doc_id
                graphs.append(graph)
            except Exception:
                pass
    return graphs


def _build_transition_counts(graphs: list) -> dict:
    counts = defaultdict(int)
    for graph in graphs:
        for edge in graph.get("edges", []):
            if edge.get("type") != "stage_transition":
                continue
            src = edge.get("source", "").replace("stg_", "")
            dst = edge.get("target", "").replace("stg_", "")
            if src and dst:
                counts[f"{src}|{dst}"] += 1
    return dict(counts)


def _build_stage_vocab(graphs: list, min_support: int = 3) -> tuple:
    """Build V2 stage vocabulary using map_stage_v2 for ground truth labels."""
    from stage_predictor import map_stage_v2
    raw_to_canonical = {}
    stage_doc_counts = defaultdict(set)

    for doc_idx, graph in enumerate(graphs):
        # Build edge context for V2 disambiguation
        nodes = {}
        out_by = defaultdict(list)
        in_by = defaultdict(list)
        for node in graph.get("nodes", []):
            if node.get("type") == "stage":
                sid = node.get("stage_id", node.get("id", "").replace("stg_", ""))
                order = node.get("features", {}).get("order_normalized", 0.5)
                nodes[node["id"]] = {"raw": sid, "order": order}
        for edge in graph.get("edges", []):
            if edge.get("type") == "stage_transition":
                src_raw = nodes.get(edge["source"], {}).get("raw", "")
                dst_raw = nodes.get(edge["target"], {}).get("raw", "")
                out_by[edge["source"]].append(dst_raw)
                in_by[edge["target"]].append(src_raw)

        for node in graph.get("nodes", []):
            if node.get("type") != "stage":
                continue
            nid = node["id"]
            raw_id = node.get("stage_id", nid.replace("stg_", ""))
            info = nodes.get(nid, {"raw": raw_id, "order": 0.5})
            canonical = map_stage_v2(
                info["raw"], order=info["order"],
                edges_in=[{"type": r} for r in in_by.get(nid, [])],
                edges_out=[{"target": r} for r in out_by.get(nid, [])],
            )
            if canonical:
                raw_to_canonical[raw_id] = canonical
                stage_doc_counts[canonical].add(doc_idx)

    stage_vocab = sorted(s for s, docs in stage_doc_counts.items()
                         if len(docs) >= min_support)
    return stage_vocab, raw_to_canonical


if __name__ == "__main__":
    main()
