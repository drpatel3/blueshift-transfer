"""SageMaker entry script: run LLM flowsheet prediction.

Loads graph data + inference index from S3, calls the LLM predictor
via Bedrock, and saves results to the model output directory.

Can run in batch mode (predict for multiple parameter sets) or single mode.

Usage (SageMaker entry script):
    Hyperparameters:
        graphs-bucket: S3 bucket with graph data
        graphs-prefix: prefix for graph JSONs (default: graphs/)
        head-grade: target Cu head grade (default: 1.0)
        deposit-type: deposit type (default: "")
        ore-type: ore type (default: "")
        throughput-tpd: plant throughput in tonnes/day (default: 0)
        mode: "standalone" or "augment" (default: standalone)
        top-k: number of similar docs (default: 5)
        model-id: Bedrock inference-profile ID (default: provider-specific)
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LLM Flowsheet Prediction")
    parser.add_argument("--graphs-bucket", type=str, required=True)
    parser.add_argument("--graphs-prefix", type=str, default="graphs/")
    parser.add_argument("--head-grade", type=float, default=1.0)
    parser.add_argument("--deposit-type", type=str, default="")
    parser.add_argument("--ore-type", type=str, default="")
    parser.add_argument("--throughput-tpd", type=float, default=0)
    parser.add_argument("--mode", type=str, default="standalone",
                        choices=["standalone", "augment"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-id", type=str, default="")
    args = parser.parse_args()

    output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(output_dir, exist_ok=True)

    # Bedrock is the only supported provider.
    os.environ["LLM_PREDICTOR_PROVIDER"] = "bedrock"
    if args.model_id:
        os.environ["LLM_PREDICTOR_MODEL"] = args.model_id

    # Smoke-test Bedrock access up front so the job fails fast with a clear
    # message rather than erroring mid-prediction.
    try:
        test_client = boto3.client("bedrock-runtime", region_name="us-east-1")
        model_id = args.model_id or "us.anthropic.claude-sonnet-4-20250514-v1:0"
        test_client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            }),
        )
        logger.info("Bedrock access verified")
    except Exception as e:
        raise RuntimeError(
            f"Bedrock access check failed ({e}). Submit the Anthropic "
            "use-case form in the AWS Bedrock console and grant the "
            "SageMaker execution role bedrock:InvokeModel on the target model."
        )

    s3 = boto3.client("s3")
    bucket = args.graphs_bucket
    prefix = args.graphs_prefix

    # 1. Load inference index
    logger.info("Loading inference index...")
    index_key = f"{prefix}inference_index.json"
    # Try XGBoost checkpoint location first, then graphs prefix
    index = None
    # Search multiple possible locations for the inference index
    index_locations = [
        index_key,
        "graphs/inference_index.json",
        "graphs_v2/inference_index.json",
    ]
    # Also search in xgb-stages job outputs
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix="xgb-stages/"):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("inference_index.json"):
                    index_locations.append(obj["Key"])
    except Exception:
        pass
    # Search in gat-build-index outputs
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix="gat-build-index/"):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("inference_index.json"):
                    index_locations.append(obj["Key"])
    except Exception:
        pass

    for try_key in index_locations:
        try:
            resp = s3.get_object(Bucket=bucket, Key=try_key)
            index = json.loads(resp["Body"].read())
            logger.info(f"Loaded index from s3://{bucket}/{try_key}: "
                        f"{len(index['docs'])} docs")
            break
        except Exception:
            pass

    if index is None:
        # Build a minimal index from graph data
        logger.info("No inference index found, building from graphs...")
        index = _build_index_from_graphs(s3, bucket, prefix)

    # 2. Load ALL graphs from S3 (for full graph traversal)
    logger.info("Loading document graphs...")
    graphs = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            if "cross_doc" in key or "stage_vocab" in key or "summary" in key:
                continue
            if "inference_index" in key:
                continue
            try:
                resp = s3.get_object(Bucket=bucket, Key=key)
                graph = json.loads(resp["Body"].read())
                doc_id = graph.get("document_id", key.split("/")[-1].replace(".json", ""))
                graph["doc_id"] = doc_id
                graphs.append(graph)
            except Exception as e:
                logger.debug(f"Skip {key}: {e}")

    logger.info(f"Loaded {len(graphs)} graphs")

    # 3. Load cross-doc edges
    logger.info("Loading cross-document edges...")
    cross_doc_data = {}
    try:
        resp = s3.get_object(Bucket=bucket, Key=f"{prefix}cross_doc_edges.json")
        cross_doc_data = json.loads(resp["Body"].read())
        logger.info(f"Loaded {len(cross_doc_data)} cross-doc entries")
    except Exception as e:
        logger.warning(f"No cross-doc edges: {e}")

    # 4. Load XGBoost predictions if in augment mode
    xgb_result = None
    if args.mode == "augment":
        logger.info("Running XGBoost prediction for augment mode...")
        # Find the best-matching graph and use it as feature source
        try:
            xgb_result = _run_xgboost_from_graph(s3, bucket, graphs, index, args)
            logger.info(f"XGBoost predicted {len(xgb_result.get('stages', []))} stages")
        except Exception as e:
            logger.warning(f"XGBoost failed, running standalone LLM: {e}")

    # 5. Run LLM prediction
    logger.info(f"Running LLM prediction: mode={args.mode}, "
                f"grade={args.head_grade}, deposit={args.deposit_type}, "
                f"ore={args.ore_type}, throughput={args.throughput_tpd}")

    from llm_predictor import predict_flowsheet_llm
    t0 = time.time()
    result = predict_flowsheet_llm(
        index=index,
        head_grade=args.head_grade,
        deposit_type=args.deposit_type,
        ore_type=args.ore_type,
        mode="augment" if xgb_result else "standalone",
        xgb_result=xgb_result,
        graphs=graphs,
        cross_doc_data=cross_doc_data,
        top_k=args.top_k,
        throughput_tpd=args.throughput_tpd,
    )
    elapsed = time.time() - t0
    logger.info(f"LLM prediction complete in {elapsed:.1f}s")

    # 6. Save result
    result_path = os.path.join(output_dir, "llm_prediction.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved prediction to {result_path}")

    # Also save a human-readable summary
    summary_path = os.path.join(output_dir, "prediction_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"LLM Flowsheet Prediction Summary\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Input: {args.head_grade}% Cu")
        if args.deposit_type:
            f.write(f", {args.deposit_type}")
        if args.ore_type:
            f.write(f", {args.ore_type}")
        if args.throughput_tpd:
            f.write(f", {args.throughput_tpd:,.0f} tpd")
        f.write(f"\nMode: {result.get('model', 'unknown')}\n")
        f.write(f"Time: {elapsed:.1f}s\n\n")

        f.write(f"Stages ({len(result['stages'])}):\n")
        f.write(f"  {' -> '.join(result['stages'])}\n\n")

        f.write(f"Connections:\n")
        for c in result.get("connections", []):
            f.write(f"  {c['from']} -> {c['to']}\n")
            if c.get("rationale"):
                f.write(f"    {c['rationale']}\n")
        f.write("\n")

        f.write(f"Decision Tree:\n")
        for d in result.get("decision_tree", []):
            f.write(f"  Q: {d.get('decision', '')}\n")
            f.write(f"  => {d.get('choice', '')}\n")
            for e in d.get("evidence", [])[:2]:
                f.write(f"     {e}\n")
            f.write("\n")

        equipment = result.get("equipment", {})
        if equipment:
            f.write(f"Equipment Specifications:\n")
            for stage, spec in equipment.items():
                f.write(f"  {stage}:\n")
                for unit in spec.get("units", []):
                    f.write(f"    {unit.get('count', '?')}x {unit.get('label', '')} "
                            f"({unit.get('size', '')})\n")
                    if unit.get("rationale"):
                        f.write(f"      {unit['rationale']}\n")
            f.write("\n")

        f.write(f"Reasoning:\n")
        for stage, reason in result.get("reasoning", {}).items():
            f.write(f"  {stage}: {reason}\n")

    logger.info(f"Saved summary to {summary_path}")

    # Print summary to stdout
    with open(summary_path) as f:
        print(f.read())


def _build_index_from_graphs(s3, bucket: str, prefix: str) -> dict:
    """Build a minimal inference index from graph JSONs when no index exists."""
    # Load transition counts from XGBoost checkpoint
    transition_counts = {}
    try:
        resp = s3.get_object(Bucket=bucket, Key=f"{prefix}transition_counts.json")
        transition_counts = json.loads(resp["Body"].read())
    except Exception:
        # Try xgb checkpoint location
        try:
            tc_key = "xgb-stages/output/transition_counts.json"
            resp = s3.get_object(Bucket=bucket, Key=tc_key)
            transition_counts = json.loads(resp["Body"].read())
        except Exception:
            pass

    return {
        "docs": [],
        "transition_counts": transition_counts,
        "stage_vocab": {},
        "threshold": 0.5,
        "num_docs": 0,
        "num_labeled": 0,
    }


def _run_xgboost_from_graph(s3, bucket: str, graphs: list,
                            index: dict, args) -> dict:
    """Run XGBoost prediction using a real graph's features for augment mode.

    Builds the full 1470-feature vector: flatten(735) + neighbor_diff(735).
    """
    import pickle
    import tempfile
    import tarfile
    import numpy as np
    from stage_predictor import (
        predict_stages, flatten_document, build_feature_names,
        _extract_tfidf_vectors, _compute_neighbor_features,
    )
    from xgboost import XGBClassifier

    # Find the best-matching labeled graph
    best_graph = None
    best_score = -1.0
    for graph in graphs:
        score = 0
        has_stages = any(n.get("type") == "stage" for n in graph.get("nodes", []))
        for node in graph.get("nodes", []):
            if node.get("type") != "context":
                continue
            feats = node.get("features", {})
            if node.get("group") == "geology":
                if args.deposit_type and feats.get(f"kw_{args.deposit_type.lower()}", 0) > 0:
                    score += 1.5
                if args.ore_type and feats.get(f"kw_{args.ore_type.lower()}", 0) > 0:
                    score += 1.0
        if has_stages:
            score += 0.5
        if score > best_score:
            best_score = score
            best_graph = graph

    if not best_graph:
        raise ValueError("No graphs available for XGBoost feature extraction")

    best_doc_id = best_graph.get("doc_id", "")
    logger.info(f"Using graph {best_doc_id[:30]} (score={best_score:.2f}) for XGBoost features")

    # Flatten ALL graphs to build the feature matrix for neighbor computation
    doc_ids = []
    feature_vecs = []
    target_idx = -1
    for i, graph in enumerate(graphs):
        vec = flatten_document(graph)
        feature_vecs.append(vec)
        did = graph.get("doc_id", f"doc_{i}")
        doc_ids.append(did)
        if did == best_doc_id:
            target_idx = i

    if target_idx < 0:
        target_idx = 0

    X = np.array(feature_vecs, dtype=np.float32)
    logger.info(f"Feature matrix: {X.shape} ({len(graphs)} docs)")

    # Download cross-doc edges for neighbor features
    cross_doc_edges = {}
    try:
        resp = s3.get_object(Bucket=bucket, Key="graphs/cross_doc_edges.json")
        cross_doc_edges = json.loads(resp["Body"].read())
    except Exception:
        pass

    # Compute neighbor-diff features (same as training)
    if cross_doc_edges:
        neighbor_feats = _compute_neighbor_features(X, doc_ids, cross_doc_edges, k=10)
        X_full = np.hstack([X, neighbor_feats])
        logger.info(f"With neighbor features: {X_full.shape}")
    else:
        # Pad with zeros if no cross-doc edges
        X_full = np.hstack([X, np.zeros_like(X)])
        logger.info(f"No cross-doc edges, zero-padded: {X_full.shape}")

    feature_vec = X_full[target_idx]

    # Download XGBoost checkpoint
    tmpdir = tempfile.mkdtemp(prefix="xgb_ckpt_")
    paginator = s3.get_paginator("list_objects_v2")
    xgb_files = []
    for page in paginator.paginate(Bucket=bucket, Prefix="xgb-stages/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("model.tar.gz"):
                xgb_files.append(obj["Key"])

    if not xgb_files:
        raise FileNotFoundError("No XGBoost model found in S3")

    model_key = sorted(xgb_files)[-1]
    logger.info(f"Loading XGBoost from s3://{bucket}/{model_key}")

    tar_path = os.path.join(tmpdir, "model.tar.gz")
    s3.download_file(bucket, model_key, tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(tmpdir)

    model_dir = Path(tmpdir)
    with open(model_dir / "stage_vocab.json") as f:
        stage_vocab = json.load(f)
    with open(model_dir / "thresholds.json") as f:
        thresholds = json.load(f)
    with open(model_dir / "transition_counts.json") as f:
        tc = json.load(f)

    chain_order = None
    if (model_dir / "chain_order.json").exists():
        with open(model_dir / "chain_order.json") as f:
            chain_order = json.load(f)

    var_selector = None
    if (model_dir / "var_selector.pkl").exists():
        with open(model_dir / "var_selector.pkl", "rb") as f:
            var_selector = pickle.load(f)

    svd_models = None
    if (model_dir / "svd_models.pkl").exists():
        with open(model_dir / "svd_models.pkl", "rb") as f:
            svd_models = pickle.load(f)

    models = {}
    for stage in stage_vocab:
        mf = model_dir / f"xgb_{stage}.json"
        if mf.exists():
            clf = XGBClassifier()
            clf.load_model(str(mf))
            models[stage] = clf

    # Ensure feature vector matches expected dimensions
    expected_dim = var_selector.n_features_in_ if var_selector else len(feature_vec)
    actual_dim = len(feature_vec)
    logger.info(f"XGBoost feature vec: {actual_dim}, expected: {expected_dim}")

    if actual_dim != expected_dim:
        # Trim or pad to match
        if actual_dim > expected_dim:
            feature_vec = feature_vec[:expected_dim]
            logger.info(f"Trimmed feature vec to {expected_dim}")
        else:
            feature_vec = np.concatenate([
                feature_vec, np.zeros(expected_dim - actual_dim, dtype=np.float32)
            ])
            logger.info(f"Padded feature vec to {expected_dim}")

    # Extract TF-IDF for SVD features
    doc_tfidf = _extract_tfidf_vectors(best_graph)

    return predict_stages(models, thresholds, feature_vec, tc, stage_vocab,
                          chain_order=chain_order, var_selector=var_selector,
                          svd_models=svd_models, doc_tfidf=doc_tfidf)


if __name__ == "__main__":
    main()
