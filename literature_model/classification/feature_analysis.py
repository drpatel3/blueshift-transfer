"""Feature sensitivity analysis for V1 XGBoost stage predictor.

Loads saved models from xgb_checkpoint/, runs ablation studies by
zeroing out feature groups, and ranks features by impact on macro F1.

Uses Binary Relevance evaluation (no chain augmentation) since
chain_order and var_selector aren't in the checkpoint. This isolates
the impact of base features cleanly.

Usage:
    python classification/feature_analysis.py --graphs-bucket sagemaker-us-east-1-666109694894
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from stage_predictor import (load_training_data, _load_graphs_from_s3, _map_stage,
                             flatten_document, build_feature_names)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).parent / "xgb_checkpoint"


def load_checkpoint():
    """Load saved models, thresholds, and feature/stage names."""
    with open(CHECKPOINT_DIR / "stage_vocab.json") as f:
        stage_vocab = json.load(f)
    with open(CHECKPOINT_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(CHECKPOINT_DIR / "thresholds.json") as f:
        thresholds = json.load(f)

    models = {}
    for stage in stage_vocab:
        model_path = CHECKPOINT_DIR / f"xgb_{stage}.json"
        if model_path.exists():
            clf = XGBClassifier()
            clf.load_model(str(model_path))
            models[stage] = clf

    logger.info(f"Loaded {len(models)} models, {len(feature_names)} features")
    return {
        "models": models,
        "stage_vocab": stage_vocab,
        "feature_names": feature_names,
        "thresholds": thresholds,
    }


def predict_binary_relevance(X, checkpoint):
    """Predict each stage independently (no chain). Returns (N, S) binary."""
    models = checkpoint["models"]
    stage_vocab = checkpoint["stage_vocab"]
    thresholds = checkpoint["thresholds"]
    S = len(stage_vocab)
    N = X.shape[0]

    y_pred = np.zeros((N, S), dtype=np.int32)
    for j, stage in enumerate(stage_vocab):
        if stage not in models:
            continue
        clf = models[stage]
        n_expected = clf.n_features_in_
        # Pad or truncate X to match model's expected features
        if X.shape[1] < n_expected:
            X_padded = np.zeros((N, n_expected), dtype=np.float64)
            X_padded[:, :X.shape[1]] = X
            probs = clf.predict_proba(X_padded)[:, 1]
        elif X.shape[1] > n_expected:
            probs = clf.predict_proba(X[:, :n_expected])[:, 1]
        else:
            probs = clf.predict_proba(X)[:, 1]
        thresh = thresholds.get(stage, 0.5)
        y_pred[:, j] = (probs >= thresh).astype(np.int32)

    return y_pred


def compute_macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def build_feature_groups(feature_names):
    """Build dict of group_name -> list of feature indices."""
    groups = defaultdict(list)
    for i, name in enumerate(feature_names):
        if "__" in name:
            section = name.split("__")[0]
            groups[f"section:{section}"].append(i)

            if "__kw_" in name:
                groups["type:keywords"].append(i)
                kw = name.split("__kw_")[1]
                if kw in ["crusher", "mill", "flotation", "leach", "thickener", "filter",
                          "electrowinning", "gravity", "elution", "solvent_extraction",
                          "cyclone", "screen", "kiln", "reactor", "adsorption", "precipitation",
                          "stockpile", "hopper", "feeder", "conveyor", "tank", "agglomeration",
                          "magnetic_separation", "merrill_crowe", "ion_exchange", "drying",
                          "water_treatment", "regrind", "cell", "ore_sorting"]:
                    groups["kw:equipment"].append(i)
                elif kw in ["chalcopyrite", "bornite", "pyrite", "chalcocite", "covellite",
                            "enargite", "arsenopyrite", "molybdenite", "galena", "sphalerite",
                            "magnetite", "hematite", "goethite", "malachite", "azurite", "chrysocolla"]:
                    groups["kw:mineralogy"].append(i)
                elif kw in ["porphyry", "skarn", "vms", "iocg", "sedimentary", "epithermal",
                            "orogenic", "breccia", "intrusive", "volcanic"]:
                    groups["kw:deposit"].append(i)
                elif kw in ["oxide", "sulfide", "sulphide", "supergene", "hypogene",
                            "transition", "refractory", "free_milling"]:
                    groups["kw:ore_type"].append(i)
                elif kw in ["tonnage", "grade", "recovery", "npv", "irr", "capex", "opex",
                            "payback", "cut_off", "measured", "indicated", "inferred"]:
                    groups["kw:economics"].append(i)
                elif kw in ["open_pit", "underground", "block_cave", "stoping", "strip_ratio"]:
                    groups["kw:mining"].append(i)
                elif kw in ["rainfall", "arid", "water_availability", "tailings_dam", "closure"]:
                    groups["kw:climate"].append(i)
            elif any(name.endswith(f"__{x}") for x in
                     ["num_count", "num_mean", "num_median", "num_max",
                      "grade", "recovery", "tonnage", "cost", "npv", "irr"]):
                groups["type:numerics"].append(i)
            elif name.endswith("__char_count") or name.endswith("__num_pages") or \
                    name.endswith("__num_tables") or name.endswith("__present"):
                groups["type:text_stats"].append(i)
        elif name.startswith("meta__"):
            groups["type:meta"].append(i)

    groups = {k: sorted(set(v)) for k, v in groups.items()}
    return groups


def ablation_study(X_val, y_val, checkpoint, actual_feature_names):
    """Zero out each feature group and measure macro F1 impact."""
    baseline_pred = predict_binary_relevance(X_val, checkpoint)
    baseline_f1 = compute_macro_f1(y_val, baseline_pred)

    groups = build_feature_groups(actual_feature_names)

    results = []
    for group_name, indices in sorted(groups.items()):
        X_ablated = X_val.copy()
        X_ablated[:, indices] = 0.0
        pred = predict_binary_relevance(X_ablated, checkpoint)
        ablated_f1 = compute_macro_f1(y_val, pred)
        impact = baseline_f1 - ablated_f1
        results.append({
            "group": group_name,
            "n_features": len(indices),
            "ablated_f1": ablated_f1,
            "impact": impact,
        })

    results.sort(key=lambda x: -x["impact"])
    return results, baseline_f1


def per_stage_importance(checkpoint):
    """Extract per-feature importance from each model, rank by stage."""
    models = checkpoint["models"]
    feature_names = checkpoint["feature_names"]

    all_importances = {}
    for stage, clf in models.items():
        imp = clf.feature_importances_
        # Only take base feature portion (before chain augmentation)
        n_base = min(len(imp), len(feature_names))
        base_imp = imp[:n_base]
        all_importances[stage] = base_imp

    # Average importance across all stages
    n_feat = len(feature_names)
    avg_imp = np.zeros(n_feat)
    for stage, imp in all_importances.items():
        padded = np.zeros(n_feat)
        padded[:len(imp)] = imp
        avg_imp += padded
    avg_imp /= len(all_importances)

    return avg_imp, all_importances


def main():
    parser = argparse.ArgumentParser(description="Feature sensitivity analysis")
    parser.add_argument("--graphs-bucket", required=True)
    parser.add_argument("--graphs-prefix", default="graphs/")
    args = parser.parse_args()

    checkpoint = load_checkpoint()
    ckpt_stages = checkpoint["stage_vocab"]  # 28 stages from checkpoint
    ckpt_stage_to_idx = {s: i for i, s in enumerate(ckpt_stages)}

    # Load graphs and build features aligned to checkpoint's stage vocab
    graphs = _load_graphs_from_s3(args.graphs_bucket, args.graphs_prefix)

    X_list, y_list, doc_ids = [], [], []
    for graph in graphs:
        doc_stages = set()
        for node in graph.get("nodes", []):
            if node.get("type") != "stage":
                continue
            raw_id = node.get("stage_id", node["id"].replace("stg_", ""))
            canonical = _map_stage(raw_id)
            if canonical and canonical in ckpt_stage_to_idx:
                doc_stages.add(canonical)

        if not doc_stages:
            continue

        vec = flatten_document(graph)
        X_list.append(vec)
        label = np.zeros(len(ckpt_stages), dtype=np.float32)
        for s in doc_stages:
            label[ckpt_stage_to_idx[s]] = 1.0
        y_list.append(label)
        doc_ids.append(graph.get("document_id", "unknown"))

    X_all = np.array(X_list, dtype=np.float64)
    y_all = np.array(y_list, dtype=np.float32)
    X_all = np.sign(X_all) * np.log1p(np.abs(X_all))
    np.nan_to_num(X_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info(f"Loaded {len(X_all)} labeled docs, {X_all.shape[1]} features, "
                f"{len(ckpt_stages)} stages (checkpoint vocab)")

    # Split same as training (70/20/10, random_state=42)
    n = len(X_all)
    indices = np.arange(n)
    train_val_idx, _ = train_test_split(indices, test_size=0.10, random_state=42)
    val_frac = 0.20 / 0.90
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_frac,
                                           random_state=42)

    X_val, y_val = X_all[val_idx], y_all[val_idx]
    logger.info(f"Val set: {len(val_idx)} docs")

    # 1. Per-feature importance
    print("\n" + "=" * 70)
    print("  FEATURE IMPORTANCE BY SECTION (avg XGBoost gain)")
    print("=" * 70)
    avg_imp, stage_imps = per_stage_importance(checkpoint)
    feature_names = checkpoint["feature_names"]

    section_imp = defaultdict(float)
    for i, name in enumerate(feature_names):
        section = name.split("__")[0] if "__" in name else "meta"
        if i < len(avg_imp):
            section_imp[section] += avg_imp[i]

    for section, imp in sorted(section_imp.items(), key=lambda x: -x[1]):
        print(f"  {section:<30} {imp:.4f}")

    print(f"\n  Top 30 individual features:")
    top_idx = np.argsort(avg_imp)[::-1][:30]
    for rank, idx in enumerate(top_idx):
        if idx < len(feature_names):
            print(f"  {rank+1:3}. {feature_names[idx]:<55} {avg_imp[idx]:.5f}")

    # 2. Ablation study
    print("\n" + "=" * 70)
    print("  FEATURE ABLATION STUDY (macro F1 drop when zeroed)")
    print("=" * 70)
    actual_names = build_feature_names()
    logger.info(f"Actual feature vector: {len(actual_names)} features (from flatten_document)")
    results, baseline = ablation_study(X_val, y_val, checkpoint, actual_names)

    print(f"\n  Baseline macro F1: {baseline:.4f}\n")
    print(f"  {'Group':<35} {'N feat':>7} {'Ablated F1':>11} {'Impact':>8}")
    print("  " + "-" * 65)
    for r in results:
        direction = "DROP" if r["impact"] > 0.005 else ("HELP" if r["impact"] < -0.005 else "    ")
        print(f"  {r['group']:<35} {r['n_features']:>7} {r['ablated_f1']:>11.4f} "
              f"{r['impact']:>+8.4f}  {direction}")

    # 3. Summary
    print("\n" + "=" * 70)
    print("  RECOMMENDATIONS")
    print("=" * 70)
    helpful = [r for r in results if r["impact"] > 0.005]
    harmful = [r for r in results if r["impact"] < -0.005]
    neutral = [r for r in results if -0.005 <= r["impact"] <= 0.005]

    if helpful:
        print("\n  Keep (removing hurts F1):")
        for r in helpful:
            print(f"    {r['group']}: {r['impact']:+.4f} ({r['n_features']} features)")
    if harmful:
        print("\n  Consider removing (removing helps F1):")
        for r in harmful:
            print(f"    {r['group']}: {r['impact']:+.4f} ({r['n_features']} features)")
    if neutral:
        print("\n  Neutral (no significant impact):")
        for r in neutral:
            print(f"    {r['group']}: {r['impact']:+.4f} ({r['n_features']} features)")


if __name__ == "__main__":
    main()
