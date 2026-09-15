"""Out-of-Lambda LLM patcher for gold PFS extractions.

The deployed Lambda runs with SKIP_LLM=true, so gold PDFs come out of S3
with PFS images extracted but no stages/units/connections. This script
fills that gap by:

  1. Listing per-PDF result JSONs in s3://mineral-pipeline-pipeline/results/
     where the doc is gold (target_mineral=="gold" or in grade_map_au.json)
     AND at least one flowsheet entry has an s3_image_key but no `data` block.
  2. For each flowsheet image, downloading the PNG, calling OpenAI's vision
     chat-completions endpoint with the same TERMS.md-driven prompt the
     Bedrock copper extractor uses, and merging the returned
     stages/units/connections JSON into the flowsheet entry.
  3. Re-uploading the patched result JSON to its existing S3 key.

Resumable: docs that already have `data` are skipped. Per-flowsheet
patching is tolerant of partial progress (only missing flowsheets are
re-extracted on a re-run).

Usage:
    python dev/gold/extract_pfs_llm.py                          # dry-run
    python dev/gold/extract_pfs_llm.py --execute --limit 5      # smoke test
    python dev/gold/extract_pfs_llm.py --execute                # full run
    python dev/gold/extract_pfs_llm.py --execute --source au_grade_map

OPENAI_API_KEY must be set. Default model is `gpt-5.4`; pass --model to
override. The model must support vision (image_url content blocks).
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEV_ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
AU_MAP_PATH = DEV_ROOT / "quantitative" / "grade_map_au.json"
TERMS_PATH = DEV_ROOT / "TERMS.md"
DOCID_INDEX_PATH = DEV_ROOT / "tmp" / "inspect" / "_docid_to_result_key.json"
ENV_PATH = DEV_ROOT / ".env"


def _load_env_file() -> None:
    """Populate os.environ from dev/.env if keys aren't already set.

    Recognizes `OPENAI_API` as an alias for `OPENAI_API_KEY` so the SDK
    (which always reads `OPENAI_API_KEY`) picks the value up either way.
    """
    if not ENV_PATH.exists():
        return
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if not k:
            continue
        if k not in os.environ:
            os.environ[k] = v
        # Alias mapping
        if k == "OPENAI_API" and "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = v

PIPELINE_BUCKET = "mineral-pipeline-pipeline"
RESULTS_PREFIX = "results/"

# OpenAI per-1M-token pricing. Used for cost reporting only — fall back to
# zero if the model isn't listed (the run will still log token totals).
OPENAI_PRICING = {
    "gpt-5.4":      {"input": 10.00, "output": 30.00},
    "gpt-4o":       {"input":  2.50, "output": 10.00},
    "gpt-4o-mini":  {"input":  0.15, "output":  0.60},
    "gpt-4-turbo":  {"input": 10.00, "output": 30.00},
    "gpt-4-vision-preview": {"input": 10.00, "output": 30.00},
}


# ---------------------------------------------------------------------------
# Prompt — kept in lock-step with pipeline/flowsheet_extraction.py so the
# OpenAI output JSON has the same shape as the Bedrock copper output.
# ---------------------------------------------------------------------------
def _load_terms() -> str:
    return TERMS_PATH.read_text(encoding="utf-8")


def _build_prompt(terms_reference: str) -> str:
    return f"""You are an expert process engineer analyzing an ore-to-metal process flowsheet.

    Use this reference to identify and categorize stages. Match flowsheet terms to standard IDs using the synonyms:

    <stage_reference>
    {terms_reference}
    </stage_reference>

    Extract ALL information from this flowsheet and return JSON with:
    1. "stages": array of process stages, each with:
       - id: lowercase identifier (e.g., "primary_crushing")
       - name: descriptive stage name
       - order: sequence number from input (1 = first stage)

    2. "units": array of equipment, each with:
       - id: lowercase identifier
       - label: exact text from diagram
       - type: equipment type
       - stage_id: which stage this belongs to
       - cells: number of cells if shown, otherwise 1

    3. "connections": array of flows, each with:
       - parent_id: source stage id (use "INPUT" for ore feed)
       - child_id: destination stage id (use "OUTPUT" for final product)

    Return only valid JSON."""


def _clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = OPENAI_PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens / 1e6) * pricing["input"] + (output_tokens / 1e6) * pricing["output"]


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
_openai_lock = threading.Lock()
_openai_client = None


def _get_openai():
    global _openai_client
    with _openai_lock:
        if _openai_client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError(
                    "openai SDK not installed. pip install 'openai>=1.0'"
                ) from e
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY env var not set")
            _openai_client = OpenAI(api_key=api_key)
        return _openai_client


def extract_with_openai(image_path: str, model: str, terms_reference: str,
                         max_retries: int = 3) -> dict:
    """Call OpenAI vision-chat with the flowsheet PNG. Returns the same shape
    as pipeline/flowsheet_extraction.py::extract_with_terms()."""
    client = _get_openai()
    prompt = _build_prompt(terms_reference)

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    # Newer OpenAI models (gpt-5.x, o-series) require `max_completion_tokens`
    # in place of `max_tokens` and reject custom temperature. Build kwargs
    # accordingly and fall back automatically if the API rejects them.
    use_legacy = model.startswith("gpt-4") or model.startswith("gpt-3")
    base_kwargs = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    if use_legacy:
        base_kwargs["temperature"] = 0.0
        base_kwargs["max_tokens"] = 4096
    else:
        base_kwargs["max_completion_tokens"] = 4096

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**base_kwargs)
            break
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            # Surface the API error body once so we can diagnose request-shape issues
            if attempt == 0:
                logger.warning(f"OpenAI call failed (model={model}): {e!r}")
            transient = (
                "rate limit" in err_str or "429" in err_str
                or "timeout" in err_str or "503" in err_str or "502" in err_str
            )
            # If a non-transient 400 mentioned an unsupported parameter, retry once
            # with the alternate token-budget key before giving up.
            if attempt == 0 and not transient and "max_tokens" in err_str:
                logger.warning("Retrying with max_completion_tokens swap")
                if "max_tokens" in base_kwargs:
                    base_kwargs["max_completion_tokens"] = base_kwargs.pop("max_tokens")
                else:
                    base_kwargs["max_tokens"] = base_kwargs.pop("max_completion_tokens", 4096)
                continue
            if attempt < max_retries - 1 and transient:
                wait = 2 ** attempt
                logger.warning(f"OpenAI transient error, retry {attempt+1} in {wait}s: {e}")
                time.sleep(wait)
                continue
            return {"error": str(e), "raw": "", "total_tokens": 0,
                    "total_cost": 0.0, "costs_by_model": {}}
    else:
        return {"error": str(last_err), "raw": "", "total_tokens": 0,
                "total_cost": 0.0, "costs_by_model": {}}

    raw = resp.choices[0].message.content or ""
    usage = resp.usage
    in_tok = getattr(usage, "prompt_tokens", 0) or 0
    out_tok = getattr(usage, "completion_tokens", 0) or 0
    total_tok = in_tok + out_tok
    cost = _calculate_cost(model, in_tok, out_tok)

    cleaned = _clean_json(raw)
    try:
        data = json.loads(cleaned)
        return {"data": data, "total_tokens": total_tok, "total_cost": cost,
                "costs_by_model": {model: cost}}
    except json.JSONDecodeError as e:
        return {"error": str(e), "raw": cleaned, "total_tokens": total_tok,
                "total_cost": cost, "costs_by_model": {model: cost}}


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------
def _s3() -> "boto3.client":
    return boto3.client("s3")


def _list_results(s3) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=PIPELINE_BUCKET, Prefix=RESULTS_PREFIX):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(".json"):
                keys.append(k)
    return keys


def _read_result_head(s3, key: str) -> dict | None:
    """Read first ~512B to peek at document_id and target_mineral
    cheaply (avoid downloading 10MB result JSONs we'll throw away)."""
    try:
        resp = s3.get_object(Bucket=PIPELINE_BUCKET, Key=key, Range="bytes=0-512")
        text = resp["Body"].read().decode("utf-8", errors="replace")
        m_id = re.search(r'"document_id"\s*:\s*"([^"]+)"', text)
        m_target = re.search(r'"target_mineral"\s*:\s*"([^"]*)"', text)
        return {"document_id": m_id.group(1) if m_id else None,
                "target_mineral": m_target.group(1) if m_target else None}
    except Exception as e:
        logger.warning(f"head read failed {key}: {e}")
        return None


def _read_full_result(s3, key: str) -> dict:
    resp = s3.get_object(Bucket=PIPELINE_BUCKET, Key=key)
    return json.loads(resp["Body"].read().decode("utf-8"))


def _write_full_result(s3, key: str, body: dict) -> None:
    s3.put_object(
        Bucket=PIPELINE_BUCKET,
        Key=key,
        Body=json.dumps(body).encode("utf-8"),
        ContentType="application/json",
    )


def _doc_needs_patching(key: str) -> bool:
    """Cheap check: does this result JSON have any flowsheet missing LLM data?"""
    try:
        result = _read_full_result(_s3(), key)
    except Exception:
        return False
    flowsheets = result.get("flowsheets") or {}
    return any(_flowsheet_needs_llm(fs) for fs in flowsheets.values())


def _flowsheet_needs_llm(fs: dict) -> bool:
    """A flowsheet entry needs LLM if it has an image but no parsed `data`.
    The Bedrock path stores extracted fields at top level (stages/units/
    connections) via dict-spread, so consider the entry done if any of
    those structural keys are present."""
    if "data" in fs and isinstance(fs.get("data"), dict):
        return False
    if any(k in fs for k in ("stages", "units", "connections")):
        return False
    return bool(fs.get("s3_image_key"))


# ---------------------------------------------------------------------------
# Doc selection
# ---------------------------------------------------------------------------
def _au_doc_ids() -> set[str]:
    if not AU_MAP_PATH.exists():
        return set()
    return set(json.loads(AU_MAP_PATH.read_text(encoding="utf-8")).keys())


def select_candidate_keys(s3, source: str, limit: int | None = None) -> list[str]:
    """Return result S3 keys that should be examined.

    source="target_mineral": every result whose head says target_mineral=="gold".
    source="au_grade_map":   every result whose document_id is in grade_map_au.json.
                              Uses the cached docid->result_key index when available.
    """
    if source == "au_grade_map":
        au_ids = _au_doc_ids()
        if not au_ids:
            logger.warning(f"{AU_MAP_PATH} not found or empty")
            return []
        if DOCID_INDEX_PATH.exists():
            idx = json.loads(DOCID_INDEX_PATH.read_text(encoding="utf-8"))
            keys = [idx[d] for d in au_ids if d in idx]
            logger.info(f"au_grade_map matches via cached index: {len(keys)}")
            return keys[:limit] if limit else keys
        # No cached index — fall back to a head-scan of every result JSON
        logger.info("No cached docid index — scanning S3 result heads (slower)")

    all_keys = _list_results(s3)
    logger.info(f"Found {len(all_keys)} result JSONs in s3://{PIPELINE_BUCKET}/{RESULTS_PREFIX}")

    if source == "target_mineral":
        candidates: list[str] = []
        for k in all_keys:
            head = _read_result_head(s3, k)
            if head and (head.get("target_mineral") or "").lower() == "gold":
                candidates.append(k)
                if limit and len(candidates) >= limit:
                    logger.info(f"Hit --limit {limit}, stopping scan early")
                    break
        logger.info(f"target_mineral=gold matches: {len(candidates)}")
        return candidates

    if source == "au_grade_map":
        au_ids = _au_doc_ids()
        candidates = []
        for k in all_keys:
            head = _read_result_head(s3, k)
            if head and head.get("document_id") in au_ids:
                candidates.append(k)
                if limit and len(candidates) >= limit:
                    break
        logger.info(f"au_grade_map matches: {len(candidates)}")
        return candidates

    raise ValueError(f"unknown source: {source}")


# ---------------------------------------------------------------------------
# Per-doc patcher
# ---------------------------------------------------------------------------
def _process_one_flowsheet(s3, name: str, fs: dict, model: str,
                            terms_reference: str, tmp_dir: Path) -> dict:
    """Download + LLM-extract one flowsheet. Mutates `fs` in place. Returns
    a small status dict so the caller can tally totals."""
    s3_image_key = fs.get("s3_image_key")
    if not s3_image_key:
        return {"name": name, "ok": False, "error": "no s3_image_key",
                "tokens": 0, "cost": 0.0}

    local_png = tmp_dir / f"{name}.png"
    try:
        resp = s3.get_object(Bucket=PIPELINE_BUCKET, Key=s3_image_key)
        local_png.write_bytes(resp["Body"].read())
    except Exception as e:
        return {"name": name, "ok": False, "error": f"download: {e}",
                "tokens": 0, "cost": 0.0}

    llm = extract_with_openai(str(local_png), model, terms_reference)
    tokens = llm.get("total_tokens", 0)
    cost = llm.get("total_cost", 0.0)

    if "data" in llm:
        fs.update(llm["data"])
        fs["llm_tokens"] = tokens
        fs["llm_cost"] = cost
        fs["llm_provider"] = "openai"
        fs["llm_model"] = model
        return {"name": name, "ok": True, "tokens": tokens, "cost": cost}
    else:
        fs["error"] = llm.get("error")
        fs["raw"] = llm.get("raw", "")
        fs["llm_provider"] = "openai"
        fs["llm_model"] = model
        return {"name": name, "ok": False, "error": llm.get("error"),
                "tokens": tokens, "cost": cost}


def patch_one_result(key: str, model: str, terms_reference: str,
                      dry_run: bool, max_flowsheets_per_doc: int = 3,
                      flowsheet_workers: int = 3) -> dict:
    """Pull, patch, push one result JSON. Up to max_flowsheets_per_doc
    flowsheets are LLM-extracted (in parallel across flowsheet_workers)."""
    s3 = _s3()
    try:
        result = _read_full_result(s3, key)
    except Exception as e:
        return {"key": key, "status": "read_error", "error": str(e)}

    flowsheets = result.get("flowsheets") or {}
    todo_all = [(name, fs) for name, fs in flowsheets.items()
                if _flowsheet_needs_llm(fs)]
    if not todo_all:
        return {"key": key, "status": "already_done",
                "n_flowsheets": len(flowsheets)}

    # Cap at max_flowsheets_per_doc. Iteration order is dict order =
    # PFS-page order from the extractor (earlier flowsheets are typically
    # the master / overall flowsheets, so taking the first N is sensible).
    skipped = max(0, len(todo_all) - max_flowsheets_per_doc)
    todo = todo_all[:max_flowsheets_per_doc]

    if dry_run:
        return {"key": key, "status": "dry_run",
                "to_patch": len(todo), "skipped_over_cap": skipped}

    patched = 0
    failures = 0
    total_tokens = 0
    total_cost = 0.0
    errors: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with ThreadPoolExecutor(max_workers=flowsheet_workers) as inner:
            inner_futs = [
                inner.submit(_process_one_flowsheet, s3, name, fs, model,
                              terms_reference, tmp_dir)
                for name, fs in todo
            ]
            for f in as_completed(inner_futs):
                r = f.result()
                total_tokens += r.get("tokens", 0)
                total_cost += r.get("cost", 0.0)
                if r["ok"]:
                    patched += 1
                else:
                    failures += 1
                    errors.append({"flowsheet": r["name"],
                                    "error": r.get("error")})

    try:
        _write_full_result(s3, key, result)
    except Exception as e:
        return {"key": key, "status": "write_error", "error": str(e),
                "patched_in_memory": patched, "failures": failures}

    return {"key": key, "status": "patched",
            "patched": patched, "failures": failures,
            "skipped_over_cap": skipped,
            "tokens": total_tokens, "cost": total_cost,
            "errors": errors}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    _load_env_file()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true",
                    help="Actually call OpenAI and write results. Default is dry-run.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N candidate docs.")
    ap.add_argument("--workers", type=int, default=16,
                    help="Across-doc concurrency (number of result JSONs in flight).")
    ap.add_argument("--flowsheet-workers", type=int, default=3,
                    help="Within-doc concurrency (parallel OpenAI calls per doc). "
                         "Bounded by --max-flowsheets-per-doc.")
    ap.add_argument("--max-flowsheets-per-doc", type=int, default=3,
                    help="Hard cap on how many flowsheets get LLM-extracted per "
                         "doc. Extras (4th onward) are left unpatched.")
    ap.add_argument("--model", default="gpt-5.4",
                    help="OpenAI vision-capable model id.")
    ap.add_argument("--source", choices=["target_mineral", "au_grade_map"],
                    default="target_mineral",
                    help="How to pick docs: by target_mineral=gold or by membership "
                         "in grade_map_au.json.")
    args = ap.parse_args()

    dry_run = not args.execute

    s3 = _s3()
    # When --limit is set on --execute, scan ALL candidates and pre-filter to
    # those that actually need patching; otherwise the limit can land entirely
    # on already-done docs and waste the smoke test. For pure --dry-run we
    # don't pre-filter (the dry-run itself reports counts).
    candidates = select_candidate_keys(s3, args.source,
                                        limit=None if args.execute else args.limit)
    if args.execute and args.limit:
        logger.info(f"Pre-filtering {len(candidates)} candidates for needs-patching "
                     f"(parallel, {args.workers} workers)...")
        needing: list[str] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_doc_needs_patching, k): k for k in candidates}
            for f in as_completed(futs):
                k = futs[f]
                try:
                    if f.result():
                        needing.append(k)
                        if len(needing) >= args.limit:
                            for ff in futs:
                                ff.cancel()
                            break
                except Exception:
                    continue
        candidates = needing[: args.limit]
        logger.info(f"Found {len(candidates)} docs needing patching for smoke run")
    elif args.limit:
        candidates = candidates[: args.limit]
        logger.info(f"Limiting to first {args.limit} candidates")

    if not candidates:
        logger.info("No candidate docs found.")
        return

    if dry_run:
        # Cheap pass: just count which need patching, no result downloads.
        # We still need to read each result JSON to count missing flowsheets.
        logger.info(f"DRY RUN: examining {len(candidates)} docs")
        terms_reference = ""
    else:
        terms_reference = _load_terms()
        logger.info(f"EXECUTE: will patch up to {len(candidates)} docs with model={args.model}")

    summary = {"already_done": 0, "patched": 0, "failures": 0,
               "to_patch": 0, "read_errors": 0, "write_errors": 0,
               "skipped_over_cap": 0, "tokens": 0, "cost": 0.0}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(patch_one_result, k, args.model, terms_reference,
                             dry_run, args.max_flowsheets_per_doc,
                             args.flowsheet_workers)
                for k in candidates]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            status = r.get("status")
            if status == "already_done":
                summary["already_done"] += 1
            elif status == "dry_run":
                summary["to_patch"] += r.get("to_patch", 0)
                summary["skipped_over_cap"] += r.get("skipped_over_cap", 0)
            elif status == "patched":
                summary["patched"] += r.get("patched", 0)
                summary["failures"] += r.get("failures", 0)
                summary["skipped_over_cap"] += r.get("skipped_over_cap", 0)
                summary["tokens"] += r.get("tokens", 0)
                summary["cost"] += r.get("cost", 0.0)
                stem = Path(r["key"]).stem
                cap_note = f" (skipped {r['skipped_over_cap']} over cap)" if r.get("skipped_over_cap") else ""
                logger.info(
                    f"  [{i}/{len(candidates)}] {stem}: "
                    f"patched={r['patched']} fail={r['failures']} "
                    f"tok={r['tokens']} cost=${r['cost']:.4f}{cap_note}"
                )
            elif status == "read_error":
                summary["read_errors"] += 1
                logger.warning(f"  [{i}/{len(candidates)}] read_error {r['key']}: {r['error']}")
            elif status == "write_error":
                summary["write_errors"] += 1
                logger.warning(f"  [{i}/{len(candidates)}] write_error {r['key']}: {r['error']}")

    print()
    print("=" * 60)
    if dry_run:
        print(f"DRY RUN summary over {len(candidates)} docs:")
        print(f"  already done:           {summary['already_done']}")
        print(f"  flowsheets to patch:    {summary['to_patch']}")
        print(f"  flowsheets over cap:    {summary['skipped_over_cap']}")
        print(f"  read errors:            {summary['read_errors']}")
        print()
        print("Re-run with --execute to actually call OpenAI and patch.")
    else:
        print(f"RUN summary over {len(candidates)} docs (model={args.model}):")
        print(f"  already done:           {summary['already_done']}")
        print(f"  flowsheets patched:     {summary['patched']}")
        print(f"  flowsheet failures:     {summary['failures']}")
        print(f"  flowsheets over cap:    {summary['skipped_over_cap']}")
        print(f"  doc read errors:        {summary['read_errors']}")
        print(f"  doc write errors:       {summary['write_errors']}")
        print(f"  total tokens:           {summary['tokens']:,}")
        print(f"  total cost:             ${summary['cost']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
