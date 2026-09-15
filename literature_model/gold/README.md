# Gold pipeline

Orchestration scripts and the OpenAI LLM patcher for gold (Au) NI 43-101
documents. Mirrors the copper extraction work but routes the LLM step
through OpenAI instead of the Lambda's Bedrock path.

## Files

| File | Purpose |
|------|---------|
| `fire_gold_triggers.py` | Upload + S3-copy-in-place triggers to push the gold backlog through Lambda (images/tables/text only — no LLM, since `SKIP_LLM=true`). |
| `watch_gold.py` | Polls `s3://.../results/` and the AWS quota approval endpoint. Emits only on state changes. |
| `gold_need_processing.txt` | 574 PDF stems already in `s3://.../pdfs/` that need a fresh PutObject event to re-fire Lambda. |
| `gold_missing_upload.txt` | 4 PDF stems present locally in `data/sedar_downloads/` but missing from S3. |
| `extract_pfs_llm.py` | The LLM patcher — pulls each gold result JSON, downloads its PFS PNG from S3, calls OpenAI vision, merges `stages` / `units` / `connections` back into the result, re-uploads. |

## Run order

1. **Stage the backlog through Lambda (images + tables + text only):**
   ```
   python dev/gold/fire_gold_triggers.py
   python dev/gold/watch_gold.py
   ```
   Wait until the result count stops growing.

2. **Run the LLM patcher (OpenAI):**
   ```
   export OPENAI_API_KEY=...
   python dev/gold/extract_pfs_llm.py --dry-run            # preview
   python dev/gold/extract_pfs_llm.py --execute --limit 5  # smoke test
   python dev/gold/extract_pfs_llm.py --execute            # full run
   ```

   By default the patcher processes every result with
   `target_mineral == "gold"`. Pass `--source au_grade_map` to instead
   iterate the docs in `quantitative/grade_map_au.json` (the
   PFS-worthy 0.3–30 g/t Au window).

3. **Refresh the Au grade map and audit:**
   ```
   python dev/quantitative/build_au_grade_map.py
   python dev/quantitative/audit_unlabeled_gold.py
   ```
   The `no_stages` bucket should shrink by roughly the count of
   successfully-patched docs.

## Notes

- Lambda env stays at `SKIP_LLM=true`. All gold LLM extraction happens
  out-of-band via `extract_pfs_llm.py`. Copper docs continue to run
  through the Lambda's Bedrock path untouched.
- The OpenAI prompt is kept in lock-step with
  `pipeline/flowsheet_extraction.py::extract_with_terms()` so the patched
  flowsheet entries are schema-compatible with copper extractions.
- `OPENAI_PRICING` in `extract_pfs_llm.py` is for cost reporting only —
  if a model isn't listed, the run still proceeds with `cost=$0.00` in
  logs.
