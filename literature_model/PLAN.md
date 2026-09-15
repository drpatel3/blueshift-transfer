# Project Plan

## Derisking Timeline — Process Synthesizer [IN PROGRESS]

**This is the working basis for model direction as of Jul 7 2026.** Source
documents (outside this repo):
`product_roadmap/optimization/word/blueshift_risk_register_20260701.docx`
(authoritative register) and `blueshift_derisk_timeline_20260625.docx`
(Gantt; all four Process Synthesizer risks active Q3 '26 → Q4 '28, all High
severity). The physics/validation groundwork below refers to
`process_model/PROGRESS.md` (M1 9/10, M2 1.77pp, bound-desaturation campaign
closed, 114/114 CI-gated tests).

**Scope (decided Jul 7 2026): this track covers the physics-informed model
(`process_model/`) only.** The ML flowsheet predictor (GNN/XGBoost phases
below) is NOT part of the derisking deliverables — interpretability and
output-resolution work target the optimizer/simulator/TEA chain.

### Critical-path decision (July 2026)

- [x] **Define the v0 designer-facing surface** (Jul 7 2026): a generated,
  self-contained HTML recommendation report per optimizer run —
  `python -m process_model.report --grade --tph` → `process_model/reports/`.
  Unblocks PS-4 (Aug), PS-2 (Sep), and gives PS-1 training something to
  demonstrate.
- [x] **Recommendation-output spec** (Jul 7 2026) — embodied as the report
  sections in `process_model/report.py`: recommended flowsheet (diagram +
  label) · alternatives leaderboard · operating setpoints (optimized +
  fixed-with-reasons) · equipment & sizing · stream table · economics
  (headline + capex/opex lines) · live M1/M2 validation block · limitations.
  Sensitivity trace joins in the Oct PS-3 item.

### PS-4 — Lack of resolution in output (likelihood 3, High)

- [ ] **Aug 2026 — equipment-level output visibility** (number of
  tanks/cells, residence time). Groundwork ~70%: simulator diagnostics
  already compute rougher/cleaner cell counts + volumes, residence time,
  thickener diameter, filter area, per-stage kW — and post-desaturation
  these are meaningful interior optima. Work = surface them in the
  recommendation artifact, readable by a process designer.
- [ ] **Oct 2026 — particle size + reagent volumes.** P80 (grind and
  regrind) already first-class; frother ppm exists but needs conversion to
  volumetric dosing; collector dosing not yet modeled (smallest new-model
  item in the register).
- [ ] **Dec 2026 — designer readability review** and incorporate feedback.

### PS-2 — Over-reliance on model (likelihood 3, High)

- [ ] **Sep 2026 — confidence scores + limitations attached to every
  recommendation.** Groundwork done: `benchmark_scorecard.py` computes the
  live validation KPIs (mean |ΔR| 1.96pp, M1 9/10, worst-plant callout).
  Work = render that block on each recommendation, plus explicit scope
  caveats (TEA relative-comparison-only, concentrator-only capex).
- [ ] **Oct 2026 — model limitations guide for designers.** Assemble from
  PROGRESS.md known-gaps, ASSUMPTIONS.md, TEA scope notes, El Abra
  structural miss, M2 error distribution. ~1-page first draft.
- [ ] **Nov 2026 — structured review session on model boundaries** with the
  design team.

### PS-3 — No interpretability plan (likelihood 3, High)

- [ ] **Oct 2026 — sensitivity output per flowsheet recommendation.**
  Physics-model equivalent of SHAP: per-variable NPV/recovery sweeps (the
  bound-desaturation probe machinery, plus visualize's tornado /
  param-sensitivity-grid plots) packaged as a per-recommendation
  sensitivity trace. (ML Tree SHAP is out of scope per the Jul 7 decision.)
- [ ] **Nov 2026 — validate the interpretability layer with two process
  engineers.**
- [ ] **Dec 2026 — decision-trace view to production.**
  `DECISION_MODEL_DATAFLOW.md` is the trace skeleton (step-by-step
  input→operation→output packets); production view renders that per run.

### PS-1 — Unfamiliar platform for process designers (likelihood 5, High)

- [ ] **Aug 2026 — interactive training model.** Gated on the v0 surface
  decision above.
- [ ] **Sep 2026 — first designer onboarding session.**
- [ ] **Oct 2026 — collect feedback and iterate training materials.**

### Out of scope for this track

Hardware/Software Controls risks (HW-1..5: sensors, fail-safe hierarchy,
cybersecurity, cloud outage, RL model errors) are the controls-integration
track (first items due Sep–Oct 2026) and are not part of the process-model
work plan.

## Process Model M1/M2 Cleaner Setpoint [COMPLETE]

Goal: replace the Cu-sulfide cleaner's open-loop gangue-recovery slope with
a merchant concentrate-grade setpoint so low- and high-grade chalcopyrite
feeds do not produce unrealistic concentrate-grade divergence.

- [x] Add explicit `target_cleaner_conc_grade_pct` assumption on
  `CuSulfideParams` (default 26% Cu).
- [x] Solve cleaner bank gangue recovery algebraically from rougher
  concentrate grade, cleaner Cu recovery, and target concentrate grade;
  invert the cleaner recycle equation and clamp bank `R_g` to [0.04, 0.20].
- [x] Add regression coverage for grade setpoint tracking across 0.4%,
  0.8%, and 0.9% chalcopyrite feeds.
- [x] Re-run M1/M2 and record sensitivity: M2 sulfide concentrate-grade
  median passes at 26%; M1 remains 7/10 because Spence and Lomas Bayas
  become sulfide picks, while a 22% target gives 9/10 but weakens M2.

## Project Objective

Given a copper head grade, predict an optimized process flowsheet (ordered stage sequence). Use the
full document context — not just grade — to make predictions that reflect the real engineering
decisions captured in NI 43-101 reports.

## Context

The mini-model predicts flowsheets from Cu head grade alone (0.7 F1). NI 43-101 reports contain 20+
sections of contextual data that explain WHY a flowsheet was designed a certain way. Section 17
tells us WHAT was designed; all other sections tell us WHY. We need a graph-based context layer that
captures this full decision landscape to improve predictions.

---

## ML Phase 1 — Text-Table Relevance Classifier [COMPLETE]

- [x] XGBoost TF-IDF baseline — F1: 0.90
  - Hyperparameter search logged in classification/hyper_parameter_combo.txt
  - Files: classification/xgb_classification.py, classification/model_xgb/
- [x] DeBERTa-v3-base — F1: 0.95, Precision: 0.92, Recall: 1.00
  - Trained on SageMaker (classification/sagemaker_train.py)
  - Files: classification/text_table_classification.py, classification/model_sm/
  - Local eval: classification/eval_local.py
- [x] Data: classification/corpus.json (579K pairs, lemmatized + stopwords removed)
- [x] DeBERTa selected as primary model (0.95 F1 vs XGBoost 0.90)

## ML Phase 2 — Document Context Graph [COMPLETE]

Goal: Build a graph that represents the full decision context from each NI 43-101 report.

Graph structure:
- **Context nodes** (up to 19 per doc): section groups from NI 43-101 (summary, property, climate, geology, metallurgical_testing, resource_estimate, mining_method, recovery_methods, infrastructure, economics, etc.)
  - Features: numeric summaries (count/mean/median/max), domain keyword counts (~60), TF-IDF vector (200), section_index, char_count, num_tables, num_pages
- **Stage nodes** (from flowsheets): normalized stage IDs with order, in/out degree, is_terminal
- **Edge types**:
  - `section_cooccurrence` — every pair of context nodes
  - `context_influences_stage` — context→stage weighted by max DeBERTa relevance score
  - `stage_transition` — directed parent→child from flowsheet connections

Steps:
- [x] 1. Updated TERMS.md with missing equipment (electrowinning, gravity, elution, solvent_extraction, agglomeration, magnetic_separation, merrill_crowe, ion_exchange, drying, water_treatment, flash_flotation)
- [x] 2. Created graph_pipeline.py — self-contained SageMaker entry script (section grouping, feature extraction, DeBERTa GPU scoring, TF-IDF fitting, graph building)
- [x] 3. Created sagemaker_graph.py — local orchestration CLI (build-graphs, status, download)
- [x] 4. Launch on SageMaker — 967 graphs produced, stored in S3 (model.tar.gz in graph-build job output)

Files:
- [x] classification/graph_pipeline.py — SageMaker entry script (reads results, scores pairs with DeBERTa, builds graphs)
- [x] classification/sagemaker_graph.py — local orchestration CLI (same pattern as sagemaker_train.py)

## ML Phase 2.5 — Cross-Document Similarity Layer [COMPLETE]

Goal: Compute pairwise document similarity so the GNN can borrow signal from analogous projects.

- [x] 1. cross_doc_similarity.py — per-section cosine similarity + strategy Jaccard, outputs cross_doc_edges.json
- [x] 2. Feature normalization — min-max on scalar columns so keyword counts don't dominate TF-IDF
- [x] 3. Section weighting — recovery_methods (0.25), metallurgical_testing (0.20), geology (0.15) weighted highest
- [x] 4. sagemaker_graph.py build-similarity command — sklearn image, ml.m5.large, 30min max
- [x] 5. Run on SageMaker — 963 docs, 19182 edges (avg 19.9/doc), output at s3://<SAGEMAKER_BUCKET>/graphs/cross_doc_edges.json

Files:
- classification/cross_doc_similarity.py — SageMaker entry script (reads graphs from S3, computes similarity, writes edges)
- classification/sagemaker_graph.py — build-similarity subcommand

## ML Phase 3 — GNN Prediction Model [IN PROGRESS]

Goal: Train a model that uses the document context graph to predict flowsheet stage sequences.

Architecture: Two-level GAT (intra-doc section cooccurrence → cross-doc similarity) with semi-supervised learning (consistency + pseudo-label losses).

- [x] Input: document context graph features (300-dim: ~100 scalar + 200 TF-IDF per section node)
- [x] Context layer: GAT message passing over section_cooccurrence edges (Level 1) + cross-doc similarity edges (Level 2)
- [x] Output: Multi-label prediction over 229 flowsheet stages (sigmoid)
- [x] Semi-supervised: ~200 labeled docs, 857 total in graph structure
- [ ] Training targets: LLM-extracted flowsheet outputs from Section 17
- [ ] Run on SageMaker, evaluate macro-F1

The GNN learns patterns like:
- High rainfall + heap leach mineralogy → heap leaching stages
- Arsenic-bearing ore → additional treatment/separation stages
- Multi-metal deposit (Cu + Au) → separate gold recovery circuit
- Low grade + high tonnage → different grinding/flotation configuration

Files:
- [x] classification/flowsheet_predictor.py — Two-level GAT model, data loading, 5-fold CV training, evaluation
- [x] classification/sagemaker_graph.py — train-gat subcommand (HuggingFace PyTorch image, ml.g4dn.xlarge)
- [x] classification/requirements_gat.txt — PyG dependencies for SageMaker

## ML Phase 3b — XGBoost Stage Prediction [IN PROGRESS]

Goal: Replace GAT classifier (macro-F1=0.05) with per-stage XGBoost binary classifiers on flattened document features. Gradient-boosted trees handle tabular data with <1K samples better than neural nets, and feature importance provides section-level interpretability.

Architecture: Flatten 6 high-signal sections × 81 features + 21 metadata + 6×20 TF-IDF SVD = ~627 features per doc → VarianceThreshold → reduce 229 stages to 23 canonical (min 3 support) → classifier chain (frequency-ordered) with per-stage hyperparameter tuning → semi-supervised pseudo-labeling → Tree SHAP section influence matrix.

Results progression:
- GAT baseline: macro-F1 = 0.05
- Binary Relevance (18 sections, with recovery_methods): macro-F1 = 0.565, micro-F1 = 0.694
- Chains + tuning (17 sections, no recovery_methods): macro-F1 = 0.5468, micro-F1 = 0.6808
- **6 high-signal sections + stage consolidation: macro-F1 = 0.6261, micro-F1 = 0.7058** (+14.5% macro-F1)
- [ ] Rare-stage improvements (target: macro-F1 0.68-0.72) — awaiting SageMaker run

Feature reduction: 6 sections (geology, metallurgical_testing, economics, mining_method, infrastructure, climate) — removed 12 noisy sections. Features: ~2,163 → ~627 pre-VT. Docs-per-feature: 0.09 → 0.29.

Stage consolidation (28 → 23): merrill_crowe/ion_exchange/solvent_extraction → solution_recovery (F1=0.62); dore/precipitation → product_recovery (F1=0.73); cell → electrowinning (F1=0.71); pond → tailing (F1=0.82). No zero-F1 stages remain.

Steps:
- [x] 1. Drop recovery_methods (Section 17) from features — circular prediction
- [x] 2. Classifier chains — exploit stage co-occurrence (flotation → thickener/filter)
- [x] 3. TF-IDF SVD features — 20-dim per section from TruncatedSVD on existing 200-dim TF-IDF vectors
- [x] 4. Per-stage hyperparameter tuning — small grid search (max_depth, colsample_bytree, reg_alpha)
- [x] 5. Semi-supervised pseudo-labeling — high-confidence predictions on ~700 unlabeled docs
- [x] 6. VarianceThreshold — remove near-constant features
- [x] 7. Strip to 6 high-signal sections + consolidate rare stages
- [x] 8. Train on SageMaker — macro-F1=0.6261, micro-F1=0.7058 (23 stages)
- [x] 9. Recall-biased threshold tuning — F-beta(1.5) for stages with <25 positives
- [x] 10. Co-occurrence prior features — P(stage_j|stage_i) conditional probability matrix
- [x] 11. SMOTE oversampling — synthetic minority examples for stages with <25 positives
- [x] 12. Early stopping + reg_alpha tuning — auto-select tree count per stage, L1 regularization
- [x] 13. Neighbor-averaged features — cross-doc similarity weighted feature differences
- [ ] 14. Run on SageMaker — validate macro-F1 improvement and no Tier 1 regression

Files:
- [x] classification/stage_predictor.py — classifier chain, SVD features, hyperparam tuning, pseudo-labeling
- [x] classification/sagemaker_graph.py — train-stages subcommand
- [x] classification/flowsheet_predictor.py — XGBoost integration with chain_order, var_selector, SVD
- [x] classification/xgb_checkpoint/ — trained models from SageMaker

## ML Phase 3c — KNN Edge Prediction [IN PROGRESS]

Goal: Mirror the simple-and-winning KNN stage predictor pattern for edge/connection prediction. Acts as a context layer on top of the stage predictor — takes the predicted stage set as input and decides which pairs are directly connected by voting over the same k nearest neighbors used for stages.

Approach:
- `score_connection_candidates()` — distance-weighted vote over each neighbor's ground-truth flowsheet edges, returns per-edge scores in [0, 1]. Direction-faithful: recycle edges (e.g. `flotation→regrind`, `cleaner→rougher`) are voted on identically to forward edges.
- `corpus_prior_scores()` — P(edge | both stages exist) from the training transition counts.
- `blend_connection_scores()` — convex combination of KNN and corpus prior (alpha knob).
- `threshold_connections()` — single tunable cutoff to binarize.
- All four model paths (xgboost / knn / blended / polynomial) now route edges through this pipeline so we have apples-to-apples connection F1 across stage models.

CLI knobs added: `--edge-threshold` (default 0.3), `--edge-alpha` (default 1.0 = pure KNN).

Tracked metric: `mean_f1` from `evaluate_connections` (`quantitative/eval.py`), plus per-edge tp/fp/fn breakdown printed by `print_report` for top-N edges by support and worst-N by F1. Reference points: prior `connection_predictor.py` SageMaker XGBoost run val F1=0.45.

Steps:
- [x] 1. Refactor `quantitative/similarity.py` — replace `knn_vote_connections` stub with score → blend → threshold pipeline
- [x] 2. Add per-edge tally to `evaluate_connections` and surface top/worst edges in `print_report`
- [x] 3. Wire through `predictor.run_experiment` and add CLI flags
- [ ] 4. Sweep `--edge-threshold` and `--edge-alpha` on the val set, record best combo
- [ ] 5. Recycle preservation check — confirm a known recycle edge survives in per-edge tally
- [ ] 6. Compare KNN vs corpus baseline vs SageMaker XGBoost connection F1

Files:
- [x] `quantitative/similarity.py` — `score_connection_candidates`, `corpus_prior_scores`, `blend_connection_scores`, `threshold_connections`
- [x] `quantitative/eval.py` — `per_edge` field on `evaluate_connections`
- [x] `quantitative/predictor.py` — `run_experiment` edge block, `--edge-threshold`, `--edge-alpha`

## ML Phase 5 — Anti-Overfitting [IN PROGRESS]

Goal: Shrink the val/OOF F1 gap on the V2 stage predictor. On the 65 gold-evaluable
docs the gap was stage_F1 0.893 vs 0.668 and reach_F1 0.775 vs 0.542 — >0.20
either way, indicating the val set had been leaking into model selection through
threshold tuning. Target: val/OOF gap ≤ 0.10 on reach_F1.

- [x] **5A. OOF-based threshold tuning** — edge + recycle thresholds were being
  selected on val reach_F1, which leaked val into the final pipeline. Rewired to
  sweep on OOF instead (`quantitative/tune_thresholds_oof.py`), picks saved to
  `classification/xgb_checkpoint_v2/tuned_thresholds.json`. `predict.py` and
  `eval_gold.py` now load those tuned values. Result on gold subset: val reach_F1
  0.775 → 0.703, OOF 0.542 → 0.594, **gap 0.233 → 0.109**.
- [x] **5B. Tighter regularization in stage trainer** — added `reg_lambda` to
  the Phase-2 grid search alongside `reg_alpha`, and dropped `subsample` from
  0.8 → 0.7 in `classification/stage_predictor.py`. Effect lands on next
  SageMaker train (`train-stages` job).
- [ ] **5C. Feature cut** — drop noisy TF-IDF SVD dims + near-zero-coverage
  binary keywords, target ≤400 features (from ~627). Deferred: invasive, needs
  a SageMaker run to validate no Tier-1 stage regresses.
- [ ] **5D. Nested cross-validation** — wrap an inner K-fold (hyperparameter
  tuning) inside an outer K-fold (reporting) so the val set is never consulted
  during model selection. Deferred: structural retrain.
- [x] **5E. Enlarged val split** — `--val-pct` default 0.20 → 0.30 in
  `stage_predictor.py`. Trades ~10% of training signal for statistical stability
  on the headline F1. Takes effect on next train.
- [x] **5F. Test-set lockdown** — enforced by memory policy: test split is never
  read during development; only val + OOF are used for iteration.

Files:
- [x] `quantitative/tune_thresholds_oof.py` — sweep-on-OOF threshold picker
- [x] `classification/xgb_checkpoint_v2/tuned_thresholds.json` — picked thresholds
- [x] `classification/predict.py` — loads tuned thresholds
- [x] `quantitative/eval_gold.py` — loads tuned thresholds
- [x] `classification/stage_predictor.py` — reg_lambda grid, subsample=0.7, val_pct=0.30

## ML Phase 4 — Layout Optimization GAT [IN PROGRESS]

Goal: Train a GAT that produces optimized (x, y) coordinates for process flow diagram layout. Flow is left-to-right (input/crushing left, final product right).

Architecture: Hybrid algorithmic + neural — generate pseudo-ground-truth via `nx.multipartite_layout`, then train supervised + physics-informed refinement losses.

- Node features (35-dim): TERMS.md category one-hot (10), topological features (6), corpus statistics (8), section association (11)
- Edge features (3-dim): log-compressed weight, direction encoding, edge type
- Model: GATConv(35→64, heads=2) → GATConv(128→32, heads=2) → Linear(64→2) → (x, y) per node
- Loss: L_supervised + 0.3*L_flow + 0.2*L_crossing + 0.2*L_spacing + 0.3*L_branch

Steps:
- [x] 1. layout_gat.py — LayoutGAT model, corpus stats, feature building, training loop, inference
- [x] 2. sagemaker_graph.py train-layout subcommand
- [x] 3. flowsheet_predictor.py — add positions to predict_flowsheet() output
- [x] 4. stage_network.py — consume learned positions in visualize_network()
- [x] 5. Train on SageMaker (100 epochs, 12min, best_loss=0.2454, 585 stage types, 14K params)
- [x] 6. Add physics-informed losses (L_process_order, L_recycle, L_mass_balance) + constraint projection layer

Files:
- [x] classification/layout_gat.py — LayoutGAT model, training, inference
- [x] classification/sagemaker_graph.py — train-layout subcommand
- [x] classification/flowsheet_predictor.py — layout positions in predict_flowsheet()
- [x] pipeline/stage_network.py — learned positions in visualize_network()

---

## Completed: Infrastructure & Deployment [COMPLETE]

Local pipeline, table extraction, Lambda architecture, and AWS deployment are all done.
54/54 tests passing. Key files: `config.py`, `storage.py`, `lambda_handler.py`, `aggregate_results.py`, `template.yaml`, `Dockerfile`.

- S3-triggered Lambda (container image, 3GB, 10min) processes PDFs and writes per-PDF JSON
- Aggregation step combines results into Parquet (flowsheets, tables, embeddings, documents)
- CLIP runs locally during aggregation, not on Lambda
- SQLite for local dev; S3 + Parquet for production
- (Optional, future) AWS Athena for ad-hoc SQL queries over Parquet

---

## Data Storage Policy

All pipeline outputs (graphs, models, checkpoints, results) must be stored in S3. No data files should be stored locally. All processing and storage occurs in the cloud. Local machines are used only for orchestration scripts and code — never as a data store.

## Existing Pipeline (supporting infrastructure)

1. pdf_extraction.py — PDF image/text/table extraction
2. flowsheet_extraction.py — LLM vision extraction of stages/units/connections
3. normalize_ids.py — Equipment ID normalization
4. stage_network.py — Directed graph of stage transitions
5. aggregate_results.py — Parquet generation for ML
6. lambda_handler.py — AWS Lambda per-PDF processing

## NI 43-101 Section Reference

┌───────────┬─────────────────────────────────────┬───────────────────────────────────────────────┐
│ Section   │              Content                │          Influence on Flowsheet               │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 1-3       │ Title, TOC, summary                 │ Project scope, key findings, QP conclusions   │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 4         │ Property description                │ Location, tenure, access constraints          │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 5         │ Climate, accessibility              │ Water management, seasonal constraints        │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 6         │ History                             │ Past production, historical processing        │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 7-8       │ Geology, mineralogy, deposit type   │ Ore characteristics, mineral associations     │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 9         │ Exploration                         │ Spatial variability, geochemistry             │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 10        │ Drilling                            │ Grade distribution, ore body geometry         │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 11        │ Sample preparation, analysis        │ Assay methods, QA/QC, analytical complexity   │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 12        │ Data verification                   │ Data confidence, independent checks           │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 13        │ Metallurgical testing               │ Recovery rates, process response              │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 14-15     │ Resource/reserve estimates           │ Grade, tonnage, scale                         │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 16        │ Mining methods                      │ Feed characteristics, production rate         │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 17        │ Recovery methods (FLOWSHEETS)        │ TARGET — what was designed                    │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 18        │ Infrastructure                      │ Power, water, transport constraints           │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 19        │ Market studies                      │ Commodity prices, economic drivers            │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 20        │ Environmental                       │ Permitting, closure, tailings constraints     │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 21-22     │ Costs, economics                    │ Capital/operating constraints, viability      │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 23        │ Interpretation, conclusions          │ QP synthesis of all prior data               │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 24        │ Recommendations                     │ Recommended next steps, further work          │
├───────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
│ 25-27     │ References, certificates             │ Supporting material (low signal)              │
└───────────┴─────────────────────────────────────┴───────────────────────────────────────────────┘

## Verification

- ML Phase 2: build sample document graph from tmp/results/, visualize node/edge counts
- ML Phase 3: train on available documents, compare F1 vs 0.7 baseline
