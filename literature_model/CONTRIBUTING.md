# Contributing

## Pipeline Stages
1. **pdf_extraction.py** — PDF extraction: images, surrounding text context, CLIP embeddings for non-flowsheet images
2. **main.py / flowsheet_extraction.py** — LLM extraction of flowsheet structure (stages, units, connections) using TERMS.md as reference
3. **normalize_ids.py** — ID normalization: maps equipment IDs to canonical forms across all flowsheets
4. **stage_network.py** — Builds directed network graph of process stage transitions

## Data Stores
- `results.json` — Flowsheet data only: stages, units, connections, text context per image
- `extracted_images/embeddings.json` — Non-flowsheet images: CLIP embeddings + text context (text_before, text_after, caption, page)
- `extracted_images/pfs/` — Flowsheet PNG files (input to LLM extraction)
- `TERMS.md` — Standard equipment terminology reference (English + Spanish synonyms)

## Coding Conventions
- Priority is writing specific algorithms, not changing the codebase at large.
- Return the simplest method that accomplishes the given task.
- Unless explicitly asked, do not provide changes to code in more than one location.
- main.py should first try to use functions from other locations before writing new functions to complete specific tasks in the main run.

## PLAN.md — Project Roadmap
- PLAN.md is the authoritative roadmap for architectural and coding decisions.
- Reference PLAN.md when writing code, making recommendations, or prioritizing work.
- After completing any step or milestone, update PLAN.md: mark items `[x]` and update the phase status tag (`[COMPLETE]`, `[IN PROGRESS]`, `[PENDING]`).

## Git / Commits
- After every file change, immediately add, commit, and push it with a one-sentence message describing the update.
