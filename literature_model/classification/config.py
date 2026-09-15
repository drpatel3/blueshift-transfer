"""Centralized configuration for classification scripts.

Reads S3 bucket and model URIs from environment variables so that
AWS account IDs are never hardcoded in source files.
"""

import os

SAGEMAKER_BUCKET = os.getenv(
    "SAGEMAKER_BUCKET", "sagemaker-us-east-1-666109694894"
)

DEBERTA_MODEL_URI = os.getenv(
    "DEBERTA_MODEL_URI",
    f"s3://{SAGEMAKER_BUCKET}/deberta-train/"
    "deberta-train-20260227145404/output/model.tar.gz",
)

# LLM predictor (Amazon Bedrock — uses AWS credits via existing IAM/boto3)
LLM_PREDICTOR_PROVIDER = os.getenv("LLM_PREDICTOR_PROVIDER", "bedrock")
LLM_PREDICTOR_MODEL = os.getenv(
    "LLM_PREDICTOR_MODEL",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
)
LLM_PREDICTOR_REGION = os.getenv(
    "LLM_PREDICTOR_REGION",
    os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
)
