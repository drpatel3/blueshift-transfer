"""SageMaker orchestration CLI for DeBERTa text-table relevance classifier.

Launches SageMaker training/scoring jobs — never processes data locally.

Usage:
    python classification/sagemaker_train.py upload-corpus
    python classification/sagemaker_train.py train --epochs 3 --instance ml.g4dn.xlarge
    python classification/sagemaker_train.py status --job-name deberta-train-xxxx
    python classification/sagemaker_train.py upload-pairs --file pairs.json
    python classification/sagemaker_train.py score --train-job deberta-train-xxxx --input s3://bucket/scoring/pairs.json
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import boto3
from sagemaker.core import image_uris
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.configs import Compute, SourceCode, InputData, StoppingCondition
from sagemaker.train.utils import Session


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = SCRIPT_DIR / "corpus.json"
BASE_JOB_NAME = "deberta-train"
STACK_NAME = "mineral-pipeline"


def _get_stack_output(cf_client, key: str) -> str | None:
    """Read an output value from the CloudFormation stack."""
    try:
        resp = cf_client.describe_stacks(StackName=STACK_NAME)
        for output in resp["Stacks"][0].get("Outputs", []):
            if output["OutputKey"] == key:
                return output["OutputValue"]
    except Exception:
        pass
    return None


def _resolve_bucket(args) -> str:
    """Resolve S3 bucket: explicit flag > stack output > default session bucket."""
    if args.bucket:
        return args.bucket
    cf = boto3.client("cloudformation")
    bucket = _get_stack_output(cf, "BucketName")
    if bucket:
        return bucket
    return Session().default_bucket()


def _resolve_role(args) -> str:
    """Resolve SageMaker role: explicit flag > stack output."""
    if args.role:
        return args.role
    cf = boto3.client("cloudformation")
    role = _get_stack_output(cf, "SageMakerRoleArn")
    if role:
        return role
    sys.exit("ERROR: No --role provided and SageMakerRoleArn not found in stack outputs. "
             f"Deploy the stack first or pass --role explicitly.")


def cmd_upload_corpus(args):
    """Upload corpus.json to S3."""
    bucket = _resolve_bucket(args)
    corpus_path = Path(args.corpus) if args.corpus else DEFAULT_CORPUS
    if not corpus_path.exists():
        sys.exit(f"ERROR: Corpus not found at {corpus_path}")

    s3_key = "training/corpus.json"
    s3 = boto3.client("s3")
    print(f"Uploading {corpus_path} to s3://{bucket}/{s3_key} ...")
    s3.upload_file(str(corpus_path), bucket, s3_key)
    print(f"Done. s3://{bucket}/{s3_key}")


def cmd_upload_pairs(args):
    """Upload a pairs JSON to S3 for scoring."""
    bucket = _resolve_bucket(args)
    local_path = Path(args.file)
    if not local_path.exists():
        sys.exit(f"ERROR: File not found: {local_path}")

    s3_key = f"scoring/{local_path.name}"
    s3 = boto3.client("s3")
    print(f"Uploading {local_path} to s3://{bucket}/{s3_key} ...")
    s3.upload_file(str(local_path), bucket, s3_key)
    print(f"Done. s3://{bucket}/{s3_key}")


def _make_source_stage() -> str:
    """Copy only the entry script + requirements to a temp dir (avoids tarballing corpus.json)."""
    stage = tempfile.mkdtemp(prefix="sm_source_")
    shutil.copy2(SCRIPT_DIR / "text_table_classification.py", stage)
    shutil.copy2(SCRIPT_DIR / "requirements.txt", stage)
    return stage


def cmd_train(args):
    """Launch SageMaker training job."""
    bucket = _resolve_bucket(args)
    role = _resolve_role(args)
    session = Session()

    image_uri = image_uris.retrieve(
        framework="huggingface",
        region=session.boto_region_name,
        version="4.36.0",
        py_version="py310",
        image_scope="training",
        instance_type=args.instance,
        base_framework_version="pytorch2.1.0",
    )
    print(f"Training image: {image_uri}")

    source_dir = _make_source_stage()
    trainer = ModelTrainer(
        training_image=image_uri,
        source_code=SourceCode(
            source_dir=source_dir,
            entry_script="text_table_classification.py",
            requirements="requirements.txt",
        ),
        hyperparameters={
            "epochs": str(args.epochs),
            "batch-size": str(args.batch_size),
            "lr": str(args.lr),
            "sample-size": str(args.sample_size),
        },
        compute=Compute(
            instance_count=1,
            instance_type=args.instance,
            volume_size_in_gb=50,
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=7200),
        base_job_name=BASE_JOB_NAME,
        role=role,
    )

    corpus_uri = f"s3://{bucket}/training/corpus.json"
    print(f"Corpus: {corpus_uri}")
    print(f"Instance: {args.instance}, epochs: {args.epochs}, "
          f"batch_size: {args.batch_size}, lr: {args.lr}, sample_size: {args.sample_size}")

    trainer.train(
        input_data_config=[
            InputData(channel_name="training", data_source=corpus_uri),
        ],
        wait=args.wait,
    )

    if args.wait:
        print(f"Training complete. Job: {trainer.latest_training_job.name}")
    else:
        print(f"Training job launched: {trainer.latest_training_job.name}")
        print("Use 'status --job-name <name>' to check progress.")


def cmd_score(args):
    """Launch SageMaker scoring job."""
    bucket = _resolve_bucket(args)
    role = _resolve_role(args)
    session = Session()
    sm_client = session.sagemaker_client

    # Get model artifacts from training job
    train_desc = sm_client.describe_training_job(TrainingJobName=args.train_job)
    model_uri = train_desc["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"Model artifacts: {model_uri}")

    # Resolve input pairs URI
    if args.input.startswith("s3://"):
        pairs_uri = args.input
    else:
        pairs_uri = f"s3://{bucket}/scoring/{args.input}"
    print(f"Input pairs: {pairs_uri}")

    image_uri = image_uris.retrieve(
        framework="huggingface",
        region=session.boto_region_name,
        version="4.36.0",
        py_version="py310",
        image_scope="training",
        instance_type=args.instance,
        base_framework_version="pytorch2.1.0",
    )

    source_dir = _make_source_stage()
    trainer = ModelTrainer(
        training_image=image_uri,
        source_code=SourceCode(
            source_dir=source_dir,
            entry_script="text_table_classification.py",
            requirements="requirements.txt",
        ),
        hyperparameters={
            "score-batch": "",
            "batch-size": str(args.batch_size),
        },
        compute=Compute(
            instance_count=1,
            instance_type=args.instance,
            volume_size_in_gb=50,
        ),
        base_job_name="deberta-score",
        role=role,
    )

    trainer.train(
        input_data_config=[
            InputData(channel_name="test", data_source=pairs_uri),
            InputData(channel_name="model", data_source=model_uri),
        ],
        wait=args.wait,
    )

    if args.wait:
        print(f"Scoring complete. Job: {trainer.latest_training_job.name}")
    else:
        print(f"Scoring job launched: {trainer.latest_training_job.name}")
        print("Use 'status --job-name <name>' to check progress.")


def cmd_status(args):
    """Check SageMaker training job status."""
    sm_client = boto3.client("sagemaker")
    desc = sm_client.describe_training_job(TrainingJobName=args.job_name)
    status = desc["TrainingJobStatus"]
    print(f"Job: {args.job_name}")
    print(f"Status: {status}")
    if status == "Completed":
        print(f"Model: {desc['ModelArtifacts']['S3ModelArtifacts']}")
        duration = desc.get("TrainingEndTime", desc["CreationTime"]) - desc["TrainingStartTime"]
        print(f"Duration: {duration}")
    elif status == "Failed":
        print(f"Failure reason: {desc.get('FailureReason', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(description="SageMaker orchestration for DeBERTa classifier")
    parser.add_argument("--bucket", type=str, default="", help="S3 bucket (default: from stack)")
    parser.add_argument("--role", type=str, default="", help="SageMaker execution role ARN")

    sub = parser.add_subparsers(dest="command")

    # upload-corpus
    p_upload = sub.add_parser("upload-corpus", help="Upload corpus.json to S3")
    p_upload.add_argument("--corpus", type=str, help="Path to corpus.json (default: classification/corpus.json)")

    # upload-pairs
    p_pairs = sub.add_parser("upload-pairs", help="Upload pairs JSON to S3 for scoring")
    p_pairs.add_argument("--file", type=str, required=True, help="Local pairs JSON file")

    # train
    p_train = sub.add_parser("train", help="Launch SageMaker training job")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=2e-5)
    p_train.add_argument("--sample-size", type=int, default=50_000)
    p_train.add_argument("--instance", type=str, default="ml.g4dn.xlarge")
    p_train.add_argument("--wait", action="store_true", help="Block until job completes")

    # score
    p_score = sub.add_parser("score", help="Launch SageMaker scoring job")
    p_score.add_argument("--train-job", type=str, required=True, help="Training job name (for model artifacts)")
    p_score.add_argument("--input", type=str, required=True, help="S3 URI or filename of pairs JSON")
    p_score.add_argument("--batch-size", type=int, default=64)
    p_score.add_argument("--instance", type=str, default="ml.g4dn.xlarge")
    p_score.add_argument("--wait", action="store_true", help="Block until job completes")

    # status
    p_status = sub.add_parser("status", help="Check job status")
    p_status.add_argument("--job-name", type=str, required=True)

    args = parser.parse_args()

    if args.command == "upload-corpus":
        cmd_upload_corpus(args)
    elif args.command == "upload-pairs":
        cmd_upload_pairs(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
