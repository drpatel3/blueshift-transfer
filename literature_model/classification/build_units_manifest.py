"""One-off: enrich static_site/manifest.json with per-stage equipment units.

For every entry in the manifest, call the LLM annotation layer to turn each
`stages: ["mill", "flotation", ...]` into
`stages: [{"name": "mill", "units": [{"type": "ball_mill", "count": 2}, ...]}, ...]`.

Then also rewrite the inline MANIFEST in static_site/index.html so the web app
can render hover tooltips without any server work.

Invariants preserved from the LLM annotation layer:
- Stage set is never mutated (XGBoost V2 chose them; LLM only annotates)
- Edge set is never mutated
- Entry count and order are preserved

Usage:
    python -m classification.build_units_manifest
    python -m classification.build_units_manifest --dry-run     # just print plan
    python -m classification.build_units_manifest --entries 3   # limit (smoke test)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

CLASSIFICATION_DIR = Path(__file__).resolve().parent
STANDALONE_DIR = CLASSIFICATION_DIR / "standalone"
STATIC_SITE_DIR = CLASSIFICATION_DIR / "static_site"
MANIFEST_JSON = STATIC_SITE_DIR / "manifest.json"
INDEX_HTML = STATIC_SITE_DIR / "index.html"

for _p in [str(STANDALONE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def enrich_entry(entry: dict, data: dict, k: int, llm_model: str) -> dict:
    """Produce a new entry with stages annotated with equipment units.

    Reuses the standalone predictor's logic so the LLM sees the same KNN
    neighbor tables it would at inference time.
    """
    from llm_units import predict_units

    # Build a minimal prediction dict that predict_units() can consume.
    # Normalize stages to plain strings — a previous partial run may have
    # left entries in {name, units} shape (fallback safe defaults). The LLM
    # layer expects a list of stage names.
    raw_stages = entry.get("stages") or []
    stage_names = [s["name"] if isinstance(s, dict) else s for s in raw_stages]
    base = {
        "cu_grade": float(entry["grade"]),
        "matched_doc_id": None,
        "matched_grade": entry["grade"],
        "stages": stage_names,
        "edges": list(entry.get("edges") or []),
        "ground_truth": None,
        "metrics": entry.get("metrics"),
    }

    annotated = predict_units(base, data, k=k, llm_model=llm_model)
    new_entry = dict(entry)
    new_entry["stages"] = annotated["stages"]  # list[{name, units}]
    return new_entry


def rewrite_inline_manifest(new_manifest: list[dict]) -> None:
    """Replace `const MANIFEST = [...];` in static_site/index.html in-place."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    # Single-line MANIFEST assignment, followed by newline (the current format).
    # Match everything between `const MANIFEST =` and the terminating `];`.
    pattern = re.compile(r"const MANIFEST = \[.*?\];", re.DOTALL)
    serialized = json.dumps(new_manifest, separators=(", ", ": "),
                            ensure_ascii=False)
    replacement = f"const MANIFEST = {serialized};"
    new_html, n = pattern.subn(replacement, html, count=1)
    if n == 0:
        raise RuntimeError("Could not locate `const MANIFEST = [...]` in index.html")
    INDEX_HTML.write_text(new_html, encoding="utf-8")
    logger.info("Rewrote inline MANIFEST in %s", INDEX_HTML)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5,
                        help="KNN neighbors for unit annotation")
    parser.add_argument("--llm-model", default=None,
                        help="Override provider default model. Bedrock default: "
                             "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    parser.add_argument("--entries", type=int, default=0,
                        help="Limit to first N entries (0 = all)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-html", action="store_true",
                        help="Only rewrite manifest.json, leave index.html alone")
    parser.add_argument("--resume", action="store_true",
                        help="Skip entries whose stages already carry diverse "
                             "LLM outputs (count>1 or >1 unit types). Use after "
                             "a partial failure (e.g. Bedrock throttling) to enrich "
                             "only what's left.")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(CLASSIFICATION_DIR.parent / ".ENV")
    except ImportError:
        pass

    if not MANIFEST_JSON.exists():
        parser.error(f"Manifest not found: {MANIFEST_JSON}")

    with open(MANIFEST_JSON, encoding="utf-8") as f:
        manifest = json.load(f)
    logger.info("Loaded %d manifest entries", len(manifest))

    full_manifest = list(manifest)
    if args.entries > 0:
        manifest = manifest[: args.entries]
        logger.info("Limited to first %d entries (partial run — "
                    "manifest.json / index.html will NOT be rewritten)",
                    len(manifest))

    # Load standalone bundle data once (doc_tables, predictions, etc.)
    import predict as standalone_predict
    data = standalone_predict._get_data()
    if not data.get("doc_tables"):
        parser.error(
            "standalone/xgb_checkpoint_v2/doc_tables.json is missing. "
            "Run: python -m classification.predict --build-tables-cache")

    if args.dry_run:
        total_llm_calls = sum(len(e["stages"]) for e in manifest)
        logger.info(
            "DRY RUN — would enrich %d entries, %d total LLM calls",
            len(manifest), total_llm_calls,
        )
        return

    def _has_llm_signal(stages) -> bool:
        for s in stages:
            if not isinstance(s, dict):
                return False
            units = s.get("units") or []
            if len(units) > 1 or any(u.get("count", 1) > 1 for u in units):
                return True
        return False

    enriched: list[dict] = []
    total_calls = 0
    skipped_resume = 0
    for i, entry in enumerate(manifest):
        if args.resume and _has_llm_signal(entry.get("stages", [])):
            enriched.append(entry)
            skipped_resume += 1
            logger.info("[%d/%d] grade=%.3f — skip (already enriched)",
                        i + 1, len(manifest), entry["grade"])
            continue
        logger.info("[%d/%d] grade=%.3f stages=%d",
                    i + 1, len(manifest), entry["grade"], len(entry["stages"]))
        new_entry = enrich_entry(entry, data, k=args.k, llm_model=args.llm_model)
        enriched.append(new_entry)
        total_calls += len(entry["stages"])
    if skipped_resume:
        logger.info("Resume: skipped %d already-enriched entries", skipped_resume)

    if args.entries > 0 and len(enriched) < len(full_manifest):
        logger.info("Partial run (--entries %d) — skipping manifest / HTML "
                    "rewrite to avoid truncating the other %d entries.",
                    args.entries, len(full_manifest) - len(enriched))
        return

    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    logger.info("Wrote enriched manifest -> %s (%d entries, ~%d LLM calls)",
                MANIFEST_JSON, len(enriched), total_calls)

    if not args.skip_html:
        rewrite_inline_manifest(enriched)


if __name__ == "__main__":
    main()
