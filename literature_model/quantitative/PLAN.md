# Quantitative Flowsheet Predictor

## Why

The graph-based XGBoost model (1,470 features) scores macro-F1=0.66 on V2 stages but is outperformed by the mini model which uses a single feature (Cu grade) with polynomial regression. Most of the 1,470 features are noise (TF-IDF vectors, keyword counts for 60+ terms, neighbor diffs). The actual engineering decisions are driven by ~10-15 quantitative inputs.

## Architecture

Simple model inspired by the mini model but with more features and the full 181-doc corpus:

### Input Features (~15-20 clean numerics)
- **Cu grade** (cu_grade_avg from pipeline results) — 664 docs with clean values
- **Ore type** flags (sulfide, oxide, mixed) — from geology keywords, 69% coverage
- **Deposit type** flags (porphyry, skarn, VMS, etc.) — 57-82% coverage
- **Mining method** flags (open_pit, underground) — 69% coverage
- Additional targeted extractions as available: tonnage, recovery, grind size

### Document Similarity
Match documents by distance in the quantitative feature space (euclidean/weighted), NOT by DeBERTa or TF-IDF. Two projects with similar grades and same ore type should have similar flowsheets.

### Output
Predict stage COUNTS + connection COUNTS directly (like mini model), not just presence/absence. This naturally handles repeated stages and ordering.

### Model Options
- Polynomial regression (like mini model) — simple, interpretable
- XGBoost on 15-20 clean features — more flexible, handles nonlinear interactions
- Neighbor-weighted voting — find K most similar docs by features, vote on stages/connections

### LLM Layer
Takes the predicted stages + connections and expands into the full process graph with equipment sizing, named instances, and recycle loops.

## Data Available

| Feature | Source | Coverage |
|---------|--------|----------|
| Cu grade (clean) | pipeline results `cu_grade_avg` | 664/973 docs (68%) |
| Ore type (sulfide/oxide) | geology `kw_sulphide`, `kw_oxide` | ~69% |
| Deposit type (porphyry, etc.) | geology keywords | ~57-82% |
| Mining method (open pit) | mining_method `kw_open_pit` | 69% |
| Flotation tested | met_testing `kw_flotation` | 69% |
| Leach tested | met_testing `kw_leach` | 68% |
| Recovery mentions | met_testing `kw_recovery` | 90% |
| Tonnage mentions | resource_estimate `kw_tonnage` | 87% |
| Target mineral | pipeline results `target_mineral` | most docs |

Clean tonnage, recovery %, grind size, CAPEX/OPEX are NOT currently extracted as clean values. They exist as raw numbers mixed in table averages. Could use LLM to extract clean values in a future pass.

## Key Files

| File | Purpose |
|------|---------|
| `quantitative/predictor.py` | Feature extraction, model training, prediction |
| `quantitative/similarity.py` | Feature-based document matching |
| `quantitative/eval.py` | Validation evaluation |
| `mini/digital_twin.py` | Reference — the mini model to match/beat |

## Validation Plan

- 70/20/10 split (same as current)
- Evaluate on validation set only until model is finalized
- Compare macro-F1 against: mini model (0.7), V1 XGBoost (0.76), V2 XGBoost (0.66)
- Evaluate both stage F1 and connection accuracy

## Previous Model (kept in `classification/`)

The graph-based XGBoost model is preserved in `classification/stage_predictor.py`, `edge_predictor.py`, `connection_predictor.py`, `llm_predictor.py`, etc. All commits are in git history.
