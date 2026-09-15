#!/bin/bash
# Deploy the mineral processing pipeline to AWS Lambda.
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials
#   - AWS SAM CLI installed
#   - Docker running (for container image build)
#
# Usage:
#   ./deploy.sh                          # Deploy with defaults
#   ./deploy.sh --stack-name my-stack    # Custom stack name

set -euo pipefail

STACK_NAME="${1:-mineral-pipeline}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

echo "=== Mineral Processing Pipeline Deployment (Bedrock-only) ==="
echo "Stack:  $STACK_NAME"
echo "Region: $REGION"
echo ""

# Build the container image
echo "--- Building container image ---"
sam build

# Deploy — no OpenAI parameter; LLM calls go through Bedrock via IAM.
echo "--- Deploying to AWS ---"
sam deploy \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --resolve-image-repos \
    --resolve-s3 \
    --capabilities CAPABILITY_IAM \
    --no-confirm-changeset

# Show outputs
echo ""
echo "--- Deployment Complete ---"
aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query "Stacks[0].Outputs" \
    --output table
