"""Run the KNN synthesizer on a real project and show the decisions it made.

Loads the real NI 43-101 graph corpus from S3, runs the exact KNN prediction
path used in `predictor.run_experiment` (find_neighbors -> knn_vote_stages ->
edge scoring), then decomposes one query document's prediction with
`similarity.build_knn_trace` and prints it as an auditable decision trace:
which analog projects drove each stage/edge, and which features made them
analogous.

With --narrate, the trace is handed to the same Bedrock Claude model the rest
of the codebase uses (`llm_predictor._call_bedrock`) to explain the decisions
in plain language — grounded ONLY in the trace (every claim must cite a
neighbor project). This is the read-only "explainer over a trace" step: the
physics/KNN still decides, the LLM narrates.

Usage:
    python -m quantitative.trace_demo --query-index 0
    python -m quantitative.trace_demo --query-doc <doc_id> --narrate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow both `python -m quantitative.trace_demo` and direct execution.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "classification"))

from quantitative import predictor
from quantitative.similarity import (
    find_neighbors, knn_vote_stages, build_knn_trace,
    score_connection_candidates, corpus_prior_scores,
    blend_connection_scores, threshold_connections,
)
from collections import defaultdict


def _fetch_graphs_parallel(graphs_bucket: str, graphs_prefix: str = "graphs/",
                           workers: int = 32) -> list:
    """Parallel S3 fetch of the graph corpus — the production loader in
    stage_predictor pulls ~960 objects one at a time (~20 min); threading the
    GETs cuts that to a minute or two. Same skip-list and filtering as
    stage_predictor._load_graphs_from_s3."""
    import boto3
    from concurrent.futures import ThreadPoolExecutor

    skip = {"cross_doc_edges.json", "stage_vocab.json", "graphs_summary.json",
            "checkpoint.json", "allowlist.json", "attempt_marker.json",
            "skip_list.json"}
    s3 = boto3.client("s3")
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=graphs_bucket, Prefix=graphs_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json") and key.split("/")[-1] not in skip:
                keys.append(key)

    def _get(key):
        body = s3.get_object(Bucket=graphs_bucket, Key=key)["Body"].read()
        return json.loads(body)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        graphs = [g for g in pool.map(_get, keys) if g is not None]
    print(f"  (fetched {len(graphs)} graphs in parallel)", file=sys.stderr)
    return graphs


def _load_graphs_cached(graphs_bucket: str):
    """Load the S3 graph corpus, caching it to the scratchpad so the S3 fetch
    happens at most once per machine. The cache lives outside the project tree
    (scratchpad), so no corpus data lands in the repo. Set TRACE_DEMO_CACHE to
    override the location; delete the file to refresh."""
    import os
    import pickle
    from stage_predictor import reduce_stage_vocab

    default_cache = (Path(os.environ.get("TEMP", "/tmp")) / "trace_demo_cache"
                     / f"graphs_{graphs_bucket}.pkl")
    cache = Path(os.environ.get("TRACE_DEMO_CACHE", default_cache))
    if cache.exists():
        print(f"  (using cached corpus at {cache})", file=sys.stderr)
        with open(cache, "rb") as f:
            return pickle.load(f)

    graphs = _fetch_graphs_parallel(graphs_bucket)
    stage_vocab, raw_to_canonical = reduce_stage_vocab(graphs, 3, "v1")
    graph_data = {"graphs": graphs, "stage_vocab": stage_vocab,
                  "raw_to_canonical": raw_to_canonical}
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(graph_data, f)
    print(f"  (cached corpus to {cache} for future runs)", file=sys.stderr)
    return graph_data


def _prepare(graphs_bucket: str, k: int, edge_threshold: float,
             edge_alpha: float):
    """Reproduce run_experiment's KNN setup and return everything the trace
    builder needs, for every validation document."""
    graph_data = _load_graphs_cached(graphs_bucket)
    data = predictor.build_features(graph_data, predictor.CORE_FEATURES)
    splits = predictor.split_data(data["X"], data["y_binary"], data["y_counts"],
                                  data["doc_ids"], data["doc_connections"])
    train, val = splits["train"], splits["val"]
    stage_vocab = data["stage_vocab"]

    indices, distances = find_neighbors(train["X"], val["X"],
                                        data["feature_names"], k=k)
    knn_probs = knn_vote_stages(indices, distances, train["y_binary"])
    val_pred = predictor.postprocess_predictions(
        knn_probs, stage_vocab, "xgboost", threshold=0.5)

    # Train-only corpus stats for the edge prior (mirrors run_experiment).
    train_transition_counts = defaultdict(int)
    for doc_edges in train["doc_connections"]:
        for src, dst in doc_edges:
            train_transition_counts[f"{src}|{dst}"] += 1
    y_train = train["y_binary"]
    joint_support = {}
    for j1, s1 in enumerate(stage_vocab):
        m1 = y_train[:, j1] > 0
        for j2, s2 in enumerate(stage_vocab):
            joint_support[(s1, s2)] = int((m1 & (y_train[:, j2] > 0)).sum())

    return {
        "data": data, "train": train, "val": val, "stage_vocab": stage_vocab,
        "indices": indices, "distances": distances, "val_pred": val_pred,
        "train_transition_counts": train_transition_counts,
        "joint_support": joint_support,
        "edge_threshold": edge_threshold, "edge_alpha": edge_alpha,
    }


def _trace_for(ctx: dict, qi: int) -> dict:
    """Build the decision trace for validation document index qi."""
    stage_vocab = ctx["stage_vocab"]
    val_pred, indices, distances = ctx["val_pred"], ctx["indices"], ctx["distances"]
    predicted = {stage_vocab[j] for j in range(len(stage_vocab))
                 if val_pred[qi, j] > 0}

    knn_scores = score_connection_candidates(
        indices[qi], distances[qi], ctx["train"]["doc_connections"], predicted)
    if ctx["edge_alpha"] < 1.0:
        prior = corpus_prior_scores(predicted, ctx["train_transition_counts"],
                                    ctx["joint_support"])
        scores = blend_connection_scores(knn_scores, prior, ctx["edge_alpha"])
    else:
        scores = knn_scores
    predicted_edges = threshold_connections(scores, ctx["edge_threshold"])

    return build_knn_trace(
        query_doc_id=ctx["val"]["doc_ids"][qi],
        neighbor_indices=indices[qi], neighbor_distances=distances[qi],
        X_query_row=ctx["val"]["X"][qi], X_train=ctx["train"]["X"],
        feature_names=ctx["data"]["feature_names"],
        train_doc_ids=ctx["train"]["doc_ids"], y_train=ctx["train"]["y_binary"],
        stage_vocab=stage_vocab,
        train_connections=ctx["train"]["doc_connections"],
        predicted_stages=predicted, predicted_edges=predicted_edges,
    )


def _short(doc_id: str, n: int = 42) -> str:
    return doc_id if len(doc_id) <= n else doc_id[:n - 3] + "..."


def _vintage(doc_id: str, meta: dict) -> str:
    """`[2019 FS]` — how old the disclosure is, plus its study grade where the
    report states one (FS/PFS only). Blank when doc_metadata.json hasn't been
    built."""
    from quantitative.doc_metadata import DISPLAYED_STUDY_TYPES

    m = meta.get(doc_id)
    if not m:
        return ""
    study = m.get("study_type")
    grade = f" {study}" if study in DISPLAYED_STUDY_TYPES else ""
    return f"  [{m.get('year') or 'year n/a'}{grade}]"


def render(trace: dict) -> None:
    from quantitative import doc_metadata
    meta = doc_metadata.load()

    print("=" * 78)
    print(f"DECISION TRACE  --  {_short(trace['query_doc_id'], 60)}"
          f"{_vintage(trace['query_doc_id'], meta)}")
    print(f"k = {trace['k']} nearest analog projects")
    print("=" * 78)

    print("\nANALOG PROJECTS (what made each a match):")
    for nb in trace["neighbors"]:
        print(f"\n  #{nb['rank']+1}  {_short(nb['doc_id'], 58)}"
              f"{_vintage(nb['doc_id'], meta)}")
        print(f"      distance={nb['distance']:.3f}  "
              f"vote weight={nb['similarity_weight']:.2f}")
        # Show the features that agree (low contribution) and any that diverge.
        agree = [fm for fm in nb["feature_match"] if fm["contribution"] < 1e-9]
        diverge = [fm for fm in nb["feature_match"] if fm["contribution"] >= 1e-9]
        if agree:
            shared = ", ".join(f"{fm['feature']}={fm['query_value']:g}"
                               for fm in agree[:6])
            print(f"      agrees on: {shared}")
        for fm in sorted(diverge, key=lambda f: -f["contribution"])[:2]:
            print(f"      differs on {fm['feature']}: "
                  f"query={fm['query_value']:g} vs neighbor="
                  f"{fm['neighbor_value']:g} (contrib {fm['contribution']:.3f})")

    print("\nSTAGE DECISIONS (vote share -> supporting analogs):")
    for sv in trace["stage_votes"]:
        mark = "[x] selected" if sv["selected"] else "    (below cut)"
        backers = ", ".join(_short(s["doc_id"], 24) for s in
                            sv["supporting_neighbors"][:3])
        print(f"  {mark}  {sv['stage']:<22} p={sv['probability']:.2f}  "
              f"from: {backers}")

    if trace["edge_votes"]:
        print("\nCONNECTION DECISIONS (score -> supporting analogs):")
        for e in trace["edge_votes"][:12]:
            mark = "[x]" if e["selected"] else "[ ]"
            print(f"  {mark} {e['src']:>18} -> {e['dst']:<18} "
                  f"score={e['score']:.2f}  "
                  f"({len(e['supporting_neighbors'])} analogs)")


NARRATE_SYSTEM = (
    "You are a process-engineering assistant explaining why a data-driven "
    "flowsheet synthesizer chose the stages it did. You are given a KNN "
    "decision trace: the query project, its nearest analog projects (with the "
    "features that made them analogous), and per-stage votes showing which "
    "analogs supported each stage. Explain the decisions GROUNDED ONLY IN THE "
    "TRACE. Every rationale must cite specific analog project ids from the "
    "trace and/or the matched features. Do not invent equipment counts, sizes, "
    "or facts not present in the trace. Return JSON with this shape: "
    '{"summary": str, '
    '"stage_decisions": [{"stage": str, "rationale": str, '
    '"grounded_in": [str]}], '
    '"caveats": [str]}'
)


def narrate(trace: dict) -> dict:
    from llm_predictor import _call_bedrock
    # Compact the trace so the prompt stays focused on the decision signal.
    compact = {
        "query_doc_id": trace["query_doc_id"],
        "neighbors": [
            {"doc_id": nb["doc_id"], "vote_weight": round(nb["similarity_weight"], 3),
             "agrees_on": [fm["feature"] for fm in nb["feature_match"]
                           if fm["contribution"] < 1e-9]}
            for nb in trace["neighbors"]
        ],
        "stage_votes": [
            {"stage": sv["stage"], "probability": round(sv["probability"], 3),
             "selected": sv["selected"],
             "supporting": [s["doc_id"] for s in sv["supporting_neighbors"]]}
            for sv in trace["stage_votes"] if sv["selected"]
        ],
    }
    user_prompt = (
        "Explain the flowsheet decisions in this KNN decision trace:\n\n"
        + json.dumps(compact, indent=2)
    )
    return _call_bedrock(NARRATE_SYSTEM, user_prompt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs-bucket",
                    default="sagemaker-us-east-1-666109694894")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--edge-threshold", type=float, default=0.3)
    ap.add_argument("--edge-alpha", type=float, default=1.0)
    ap.add_argument("--query-index", type=int, default=0,
                    help="Validation-set index to explain.")
    ap.add_argument("--query-doc", default=None,
                    help="Validation doc id to explain (overrides --query-index).")
    ap.add_argument("--narrate", action="store_true",
                    help="Also narrate the decisions with Bedrock Claude.")
    args = ap.parse_args()

    print(f"Loading corpus from s3://{args.graphs_bucket}/graphs/ ...",
          file=sys.stderr)
    ctx = _prepare(args.graphs_bucket, args.k, args.edge_threshold, args.edge_alpha)
    print(f"  {len(ctx['train']['doc_ids'])} train / "
          f"{len(ctx['val']['doc_ids'])} val docs, "
          f"{len(ctx['stage_vocab'])} stages", file=sys.stderr)

    qi = args.query_index
    if args.query_doc is not None:
        try:
            qi = ctx["val"]["doc_ids"].index(args.query_doc)
        except ValueError:
            print(f"doc {args.query_doc} not in validation set", file=sys.stderr)
            sys.exit(1)

    trace = _trace_for(ctx, qi)
    render(trace)

    if args.narrate:
        print("\n" + "=" * 78)
        print("LLM NARRATION (Bedrock Claude, grounded in the trace above)")
        print("=" * 78)
        out = narrate(trace)
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
