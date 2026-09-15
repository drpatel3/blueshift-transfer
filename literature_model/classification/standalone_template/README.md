# Standalone V2 Flowsheet Predictor

Self-contained bundle of the highest-performing prediction model
(V2 XGBoost stage predictor, val macro-F1 = 0.86). Copy this folder
anywhere, install deps, and run predictions.

## Install

```
cd classification/standalone
python -m venv .venv
.venv\Scripts\activate          # Windows
# or:  source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

## Run

Minimal (stages + edges + comparison visualization):

```
python predict.py --grade 1.25
```

Multiple grades in one run:

```
python predict.py --grade 0.50 1.00 1.25 1.50
```

Add LLM-inferred equipment per stage (requires AWS Bedrock access, see below):

```
python predict.py --grade 1.25 --with-units
```

Dump full prediction JSON alongside the PNG:

```
python predict.py --grade 1.25 --with-units --json prediction.json
```

## LLM Equipment Layer (`--with-units`)

The LLM is an **annotation layer only** — it never decides which stages
exist. The XGBoost model's stage predictions are authoritative and immutable.
The LLM is scoped to one stage at a time and its only job is to return
`{type, count}` for that stage based on equipment tables from the 5 nearest
neighbor projects (by Cu head grade).

Invariants enforced at runtime:
- Stage set before/after the LLM layer is identical
- Edge set before/after is identical
- The LLM prompt never exposes the full stage set — only one stage name per call
- Malformed LLM output falls back to a safe default for that stage only

**LLM provider: Amazon Bedrock (Claude Sonnet).** Uses whatever AWS
credentials are configured in your environment (`~/.aws/credentials`,
`AWS_PROFILE`, or instance role).

```
# Requires AWS creds + Bedrock access to the model
aws configure            # if not already done
python predict.py --grade 1.25 --with-units
```

Model and region overrides (env vars):

```
LLM_UNITS_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0  # default
LLM_UNITS_REGION=us-east-1                                    # default
```

## Files

- `predict.py` — entry point
- `llm_units.py` — LLM annotation layer (strict non-mutating)
- `similarity.py`, `eval.py`, `pfs_visual.py` — helpers
- `xgb_checkpoint_v2/` — cached V2 model predictions, vocab, edges, tables
- `grade_map.json` — doc → Cu head grade lookup for KNN

## Notes

- This bundle does **not** retrain the model. All predictions are served
  from the cached V2 outputs in `xgb_checkpoint_v2/`. Retraining lives in
  the main repo (`classification/stage_predictor.py`).
- `doc_tables.json` (used by `--with-units`) is populated from the main
  repo by:

  ```
  python -m classification.predict --build-tables-cache
  ```

  This pulls per-doc tables from `s3://mineral-pipeline-pipeline/results/`,
  filters to V2-covered docs, trims rows/columns to bound file size, and
  mirrors the result into `classification/standalone/xgb_checkpoint_v2/`.
  Without the cache, `--with-units` still runs but equipment counts default
  to `1×` of the canonical type per stage (invariants still hold — stages
  and edges are never mutated).

- No AWS credentials or S3 access needed at inference time — only at
  cache-build time, which is a dev task.
