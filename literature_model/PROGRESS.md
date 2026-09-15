# Progress Log — Gold Training Corpus Build
_Last updated: 2026-04-21_

## Snapshot

| metric | value |
|---|---|
| **Gold training corpus (PFS image + Au head grade in 0.3–30 g/t)** | **342 docs** |
| PFS images across those 342 docs | ≥1,000 (477 from straggler retrigger + 1,660 from new-batch retrigger, but many docs count multiple) |
| Stems list of the 342 | `C:\Users\RYANMU~1\AppData\Local\Temp\gold_training_corpus_stems.txt` |
| LLM extraction status on the 342 | **not yet run** — `SKIP_LLM=true` on Lambda |
| Bedrock spend on this batch | $0 |

---

## What happened today

### 1. Uploaded 758 new gold PDFs to S3
Ran `pipeline/batch_upload.py mineral-pipeline-pipeline data/sedar_downloads/gold --skip-existing` after adding a `--skip-existing` flag that dedupes against the existing `pdfs/` prefix by basename. 1,395 of the 2,153 local gold PDFs were already in S3 and got skipped; 758 were new.

### 2. Discovered 0% PFS-extraction yield on the new batch
First sweep: **0 / 593** PDFs yielded a PFS image. Also `target_mineral: "copper"` on docs clearly named with `_gold_`. Two bugs surfaced:

- **Stale Lambda container** — deployed 2026-04-17, but `pipeline/pdf_extraction.py` had improvements that never made it into the image. Verified by running `extract_all_figures` locally on 3 sample "zero-PFS" docs → got 6, 20, 2 flowsheets. Same code, same bytes, different result on Lambda.
- **Filename-regex bug** — `\bgold\b` against `1911_gold_corporation-…`. Python's `\b` treats `_` as a word character, so `_gold_` never matched. All `_gold_`-style filenames fell through to the `TARGET_MINERAL=copper` env default.

### 3. Fixes (commit `cc4e84d`, pushed)
- `pipeline/lambda_handler.py`: replaced `\b{mineral}\b` with `(?<![a-z]){mineral}(?![a-z])` — handles underscore/digit/hyphen boundaries, still rejects `goldcorp`/`marigold` false positives.
- `template.yaml`: `SKIP_LLM` flipped from `"false"` to `"true"` for this yield-measurement phase.
- `sam build` + `sam deploy` — new container live at `2026-04-21T20:48:51Z`.

### 4. Retrigger passes
Two cohorts reprocessed via S3 self-copy (`copy_object` with `MetadataDirective=REPLACE`) + `steps_completed` reset:

| cohort | retriggered | yielded ≥1 PFS | rate |
|---|---|---|---|
| New-batch zero-flowsheet (583 + earlier 10 sample) | 593 | 354 | 59.7% |
| Au-qualified "stragglers" (docs in existing `grade_map_au.json` with no PFS) | 281 | 139* | 53.1% |

\* = of 262 that completed within the 25-min poll; 19 stragglers still in-flight at last check.

### 5. Head-grade filtering on the new batch
Ran the **existing** `quantitative/enrich_graphs.py` + `quantitative/build_au_grade_map.py` logic directly on Lambda results (no graph rebuild needed) to compute per-doc Au head grade via the section fallback chain `metallurgical_testing → recovery_methods → resource_estimate → geology → summary`. Of the 354 new PFS-yielding docs:

- **119** in the 0.3–30 g/t PFS window ← qualified
- 69 too high (>30 g/t; drill intercepts/concentrate leaking through)
- 5 too low (<0.3 g/t; exploration-stage)
- 161 no resolvable head grade

74 of the 119 pulled their head grade from `metallurgical_testing` — exactly what you want (actual mill-feed head grades from met testwork, not resource-block averages).

### 6. Corpus totals after today

| source | docs | notes |
|---|---|---|
| Existing `grade_map_au.json` Au-qualified | 361 | pre-existing |
| … of which have PFS (before straggler retrigger) | 80 | |
| … of which have PFS (after straggler retrigger) | **223** | +143 from retrigger |
| New-batch qualified | **119** | |
| Overlap | 0 | genuinely new docs |
| **Total usable gold training corpus** | **342** | |

---

## Key artifacts & paths

**Committed code**
- `pipeline/lambda_handler.py` — filename detection fix at line 37-46
- `pipeline/batch_upload.py` — `--skip-existing` flag with S3 dedup
- `template.yaml` — `SKIP_LLM: "true"` (remember to flip before LLM pass)
- `quantitative/build_au_grade_map.py` — grade window constants and fallback order (unchanged)
- `quantitative/enrich_graphs.py` — section grouping and grade regex (unchanged, reused)

**Local working artifacts (temp; rebuild if needed)**
- `C:\Users\RYANMU~1\AppData\Local\Temp\gold_training_corpus_stems.txt` — 342 stems, one per line (the full training set)
- `C:\Users\RYANMU~1\AppData\Local\Temp\grade_map_au_new_batch.json` — 119 new entries, same shape as `quantitative/grade_map_au.json`
- `C:\Users\RYANMU~1\AppData\Local\Temp\au_headgrade_detail.json` — full per-doc analysis of the 354 new PFS docs (including 161 missing + 69/5 out-of-window)
- `C:\Users\RYANMU~1\AppData\Local\Temp\pfs_yield_report.json` — per-doc PFS counts for the full 593 new-batch sweep
- `C:\Users\RYANMU~1\AppData\Local\Temp\retriggered_stragglers.txt` — 281 straggler stems

**S3**
- `s3://mineral-pipeline-pipeline/pdfs/` — ~3,059 PDFs (was 2,301, +758 new)
- `s3://mineral-pipeline-pipeline/results/` — ~2,008 per-PDF JSONs (tables, page_text, flowsheet metadata)
- `s3://mineral-pipeline-pipeline/images/<doc_hash>/` — PFS PNG images, uploaded as part of image extraction step

**Lambda**
- Function: `mineral-pipeline-ExtractFunction-2VAIiis05hS6`
- Container last deployed: `2026-04-21T20:48:51Z`
- `SKIP_LLM=true` currently

---

## Open items / next steps

1. **Wait ~5–10 min, re-sweep the last 19 stragglers** — likely adds ~10 more docs to the corpus.
2. **Merge `grade_map_au_new_batch.json` (119) into `quantitative/grade_map_au.json`** — bumps it from 361 to 480 entries so downstream scripts see the new docs.
3. **Flip `SKIP_LLM` back to `"false"` and redeploy**, then retrigger LLM extraction on the 342 (or a doc-hash-filtered subset). This is where Bedrock spend starts — 342 PDFs × several flowsheets × few iterations each.
4. **`TARGET_MINERAL=copper` default is wrong** for a gold-heavy batch. Options: change the template default to `"gold"`, or return empty string when filename has no match and let the downstream treat target_mineral as optional.
5. **165 of the original 758 uploads never produced a results JSON** — separate bug, worth chasing via CloudWatch logs before the LLM pass.
6. **Copper audit equivalent** — `grade_map.json` has 963 Cu-qualified docs; some fraction lack PFS and would likely be recoverable via the same straggler-retrigger flow. User expressed interest but we haven't run it yet.

---

## Notable gotchas for next session

- `pipeline/retrigger_image_extraction.py` (committed) targets *unlabeled-in-V2* docs, not *PFS-missing* docs — similar-sounding but different set. Our straggler retrigger was a separate ad-hoc script that filtered on `flowsheets == 0` directly.
- Any time a Lambda/Docker redeploy is needed: **Docker Desktop must be running** (not auto-started on Windows). `sam build` fails silently with a "Running AWS SAM projects locally requires a container runtime" error otherwise.
- The section-aware head-grade resolution is canonical via `quantitative/enrich_graphs.group_tables_by_section` + `quantitative/build_au_grade_map._AU_GRADE_FALLBACK_SECTIONS`. Don't reimplement it — import it.
- The new batch's doc_ids (SHA-256 of PDF bytes) are not yet in `grade_map_au.json`; that file is still keyed to the Feb-corpus hashes. Until the merge happens (open item #2), any tool that cross-references `grade_map_au.json` won't see the new 119.
