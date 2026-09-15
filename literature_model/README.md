# Mineral Processing Literature Platform

Automated pipeline for extracting structured process engineering data from mineral processing plant technical reports (NI 43-101, feasibility studies, etc.).

## What It Does

1. **PDF Extraction** — Identifies and extracts process flow diagrams, tables, and page text from large technical PDFs
2. **LLM Flowsheet Analysis** — Uses Claude (via AWS Bedrock) to parse flowsheet images into structured JSON (stages, equipment, connections)
3. **Normalization** — Maps equipment IDs to canonical forms using a standard terminology reference
4. **Stage Network** — Builds directed graphs of process stage transitions across documents
5. **Classification** — SageMaker-based graph neural network for document similarity and stage prediction

## Architecture

```
pipeline/               Core extraction pipeline
  pdf_extraction.py       Image, table, and text extraction from PDFs
  flowsheet_extraction.py LLM-based flowsheet parsing
  normalize_ids.py        Equipment ID normalization
  stage_network.py        Process stage transition graphs
  main.py                 Local orchestrator
  lambda_handler.py       AWS Lambda entry point (per-PDF)
  config.py               Centralized configuration (env vars)
  storage.py              Local/S3 storage abstraction
  database.py             SQLite layer with Pydantic validation

classification/         ML training and graph construction
  sagemaker_graph.py      SageMaker job orchestration
  stage_predictor.py      XGBoost stage prediction model
  cross_doc_similarity.py Cross-document similarity computation
  config.py               Centralized S3/model configuration
```

## Setup

### Prerequisites
- Python 3.11+
- AWS CLI configured (`aws configure`) — Bedrock access on the configured account
- Bedrock model access enabled for `anthropic.claude-sonnet-4-6` (or the configured `BEDROCK_MODEL`)

### Local Development

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
```

LLM calls use AWS Bedrock via boto3 and inherit credentials from the default
AWS credential chain (env vars, `~/.aws/credentials`, or instance role). No
`.env` secrets are required for the Bedrock path.

Run the pipeline locally:
```bash
cd pipeline
python main.py
```

Run tests:
```bash
pytest pipeline/ -v
```

### AWS Deployment

Requires SAM CLI and Docker.

```bash
sam build
sam deploy --stack-name mineral-pipeline \
  --region us-east-1 \
  --resolve-image-repos --resolve-s3 \
  --capabilities CAPABILITY_IAM \
  --no-confirm-changeset
```

Upload PDFs to trigger Lambda processing:
```bash
python pipeline/batch_upload.py mineral-pipeline-pipeline ./pdfs
```

## Configuration

All settings are controlled via environment variables with sensible defaults. See `pipeline/config.py` and `classification/config.py` for the full list.

Key variables:
| Variable | Description | Default |
|---|---|---|
| `BEDROCK_MODEL` | Claude model for flowsheet extraction | `us.anthropic.claude-sonnet-4-6` |
| `AWS_REGION` | Region for Bedrock client | `us-east-1` |
| `SAGEMAKER_BUCKET` | S3 bucket for ML artifacts | (per-account) |
| `PIPELINE_BUCKET` | S3 bucket for pipeline data | `mineral-pipeline-pipeline` |

## Directory Structure

```
dev/
  pipeline/         Core extraction + Lambda handlers
  classification/   ML model training scripts
  template.yaml     SAM infrastructure template
  Dockerfile        Lambda container image
  TERMS.md          Standard equipment terminology reference
  PLAN.md           Project roadmap
```

## Testing

```bash
pytest pipeline/ -v    # 54 tests covering DB, pipeline, error isolation
```
