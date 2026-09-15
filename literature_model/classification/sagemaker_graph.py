"""SageMaker orchestration CLI for document context graph building.

Launches SageMaker jobs to build document context graphs and cross-document
similarity layers. Never processes data locally.

Usage:
    python classification/sagemaker_graph.py build-graphs --wait
    python classification/sagemaker_graph.py build-similarity --wait
    python classification/sagemaker_graph.py train-gat --wait
    python classification/sagemaker_graph.py status --job-name graph-build-xxxx
"""

import argparse
import json
import os
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
BASE_JOB_NAME = "graph-build"
STACK_NAME = "mineral-pipeline"
from config import SAGEMAKER_BUCKET
from config import DEBERTA_MODEL_URI


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


def _make_source_stage() -> str:
    """Copy entry script + requirements to a temp dir for SageMaker upload."""
    stage = tempfile.mkdtemp(prefix="sm_graph_source_")
    shutil.copy2(SCRIPT_DIR / "graph_pipeline.py", stage)
    shutil.copy2(SCRIPT_DIR / "requirements.txt", stage)
    return stage


def cmd_build_graphs(args):
    """Launch SageMaker job to build document context graphs."""
    bucket = _resolve_bucket(args)
    role = _resolve_role(args)
    # Use the project bucket as the SageMaker session bucket — the execution
    # role only has S3 access on mineral-pipeline-pipeline, not on the
    # account-default sagemaker-us-east-1-<acct> bucket.
    session = Session(default_bucket=bucket)

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

    results_bucket = args.results_bucket
    results_uri = f"s3://{results_bucket}/results/"
    model_uri = args.model_uri or DEBERTA_MODEL_URI

    source_dir = _make_source_stage()
    trainer = ModelTrainer(
        training_image=image_uri,
        source_code=SourceCode(
            source_dir=source_dir,
            entry_script="graph_pipeline.py",
            requirements="requirements.txt",
        ),
        hyperparameters={
            "batch-size": str(args.batch_size),
            "relevance-threshold": str(args.relevance_threshold),
            "checkpoint-bucket": bucket,
            "checkpoint-prefix": args.checkpoint_prefix,
            "results-bucket": results_bucket,
            "results-prefix": "results/",
            **({"allowlist-s3-key": args.allowlist_s3_key}
               if args.allowlist_s3_key else {}),
        },
        compute=Compute(
            instance_count=1,
            instance_type=args.instance,
            volume_size_in_gb=50,
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=14400),
        base_job_name=BASE_JOB_NAME,
        role=role,
        sagemaker_session=session,
    )

    print(f"Results: {results_uri}")
    print(f"DeBERTa model: {model_uri}")
    print(f"Instance: {args.instance}, batch_size: {args.batch_size}, "
          f"relevance_threshold: {args.relevance_threshold}")

    try:
        trainer.train(
            input_data_config=[
                InputData(channel_name="model", data_source=model_uri),
            ],
            wait=args.wait,
        )
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Run complete. Job: {job_name}")
    except Exception as e:
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Job {job_name} failed: {e}")
        if not args.wait:
            raise

    # Launch next run if requested (works whether previous run succeeded or OOM'd)
    if args.wait and args.runs > 1:
        s3_client = boto3.client("s3")
        try:
            resp = s3_client.get_object(Bucket=bucket, Key=args.checkpoint_prefix + "checkpoint.json")
            checkpoint = json.loads(resp["Body"].read())
            done = len(checkpoint)
            print(f"\n{done} docs completed so far. Launching next run...")
            args.runs -= 1
            cmd_build_graphs(args)
        except s3_client.exceptions.NoSuchKey:
            print("All docs processed — no checkpoint remaining.")


def _make_similarity_source_stage() -> str:
    """Copy cross-doc similarity script to a temp dir for SageMaker upload."""
    stage = tempfile.mkdtemp(prefix="sm_similarity_source_")
    shutil.copy2(SCRIPT_DIR / "cross_doc_similarity.py", stage)
    return stage


GRAPHS_BUCKET_DEFAULT = SAGEMAKER_BUCKET


def cmd_build_similarity(args):
    """Launch SageMaker job to compute cross-document similarity edges."""
    role = _resolve_role(args)
    session = Session()
    graphs_bucket = args.graphs_bucket or GRAPHS_BUCKET_DEFAULT

    image_uri = image_uris.retrieve(
        framework="sklearn",
        region=session.boto_region_name,
        version="1.2-1",
        image_scope="training",
        instance_type=args.instance,
    )
    print(f"Training image: {image_uri}")

    source_dir = _make_similarity_source_stage()
    trainer = ModelTrainer(
        training_image=image_uri,
        source_code=SourceCode(
            source_dir=source_dir,
            entry_script="cross_doc_similarity.py",
        ),
        hyperparameters={
            "graphs-bucket": graphs_bucket,
            "graphs-prefix": "graphs_v2/",
            "top-k": str(args.top_k),
        },
        compute=Compute(
            instance_count=1,
            instance_type=args.instance,
            volume_size_in_gb=10,
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=1800),
        base_job_name="cross-doc-sim",
        role=role,
    )

    print(f"Graphs source: s3://{graphs_bucket}/graphs/")
    print(f"Instance: {args.instance}, top_k: {args.top_k}")

    try:
        trainer.train(wait=args.wait)
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Run complete. Job: {job_name}")
    except Exception as e:
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Job {job_name} failed: {e}")
        if not args.wait:
            raise


def _make_gat_source_stage() -> str:
    """Copy GAT training script + requirements to a temp dir for SageMaker upload."""
    stage = tempfile.mkdtemp(prefix="sm_gat_source_")
    shutil.copy2(SCRIPT_DIR / "flowsheet_predictor.py", stage)
    shutil.copy2(SCRIPT_DIR / "requirements_gat.txt", os.path.join(stage, "requirements.txt"))
    return stage


def cmd_train_gat(args):
    """Launch SageMaker job to train GAT for flowsheet prediction."""
    role = _resolve_role(args)
    session = Session()
    graphs_bucket = args.graphs_bucket or GRAPHS_BUCKET_DEFAULT

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

    source_dir = _make_gat_source_stage()
    trainer = ModelTrainer(
        training_image=image_uri,
        source_code=SourceCode(
            source_dir=source_dir,
            entry_script="flowsheet_predictor.py",
            requirements="requirements.txt",
        ),
        hyperparameters={
            "graphs-bucket": graphs_bucket,
            "graphs-prefix": "graphs_v2/",
            "epochs": str(args.epochs),
            "lr": str(args.lr),
            "patience": str(args.patience),
            "folds": str(args.folds),
        },
        compute=Compute(
            instance_count=1,
            instance_type=args.instance,
            volume_size_in_gb=50,
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=7200),
        base_job_name="gat-flowsheet",
        role=role,
    )

    print(f"Graphs source: s3://{graphs_bucket}/graphs/")
    print(f"Instance: {args.instance}, epochs: {args.epochs}, lr: {args.lr}, "
          f"patience: {args.patience}, folds: {args.folds}")

    try:
        trainer.train(wait=args.wait)
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Run complete. Job: {job_name}")
    except Exception as e:
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Job {job_name} failed: {e}")
        if not args.wait:
            raise


def _make_layout_source_stage() -> str:
    """Copy layout GAT training script + requirements to a temp dir for SageMaker upload."""
    stage = tempfile.mkdtemp(prefix="sm_layout_source_")
    shutil.copy2(SCRIPT_DIR / "layout_gat.py", stage)
    shutil.copy2(SCRIPT_DIR / "requirements_gat.txt", os.path.join(stage, "requirements.txt"))
    return stage


def cmd_train_layout(args):
    """Launch SageMaker job to train layout GAT."""
    role = _resolve_role(args)
    session = Session()
    graphs_bucket = args.graphs_bucket or GRAPHS_BUCKET_DEFAULT

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

    source_dir = _make_layout_source_stage()
    trainer = ModelTrainer(
        training_image=image_uri,
        source_code=SourceCode(
            source_dir=source_dir,
            entry_script="layout_gat.py",
            requirements="requirements.txt",
        ),
        hyperparameters={
            "graphs-bucket": graphs_bucket,
            "graphs-prefix": "graphs_v2/",
            "epochs": str(args.epochs),
            "lr": str(args.lr),
        },
        compute=Compute(
            instance_count=1,
            instance_type=args.instance,
            volume_size_in_gb=50,
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=10800),
        base_job_name="gat-layout",
        role=role,
    )

    print(f"Graphs source: s3://{graphs_bucket}/graphs/")
    print(f"Instance: {args.instance}, epochs: {args.epochs}, lr: {args.lr}")

    try:
        trainer.train(wait=args.wait)
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Run complete. Job: {job_name}")
    except Exception as e:
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Job {job_name} failed: {e}")
        if not args.wait:
            raise


def _make_stages_source_stage() -> str:
    """Copy XGBoost stage predictor script + requirements to temp dir for SageMaker."""
    stage = tempfile.mkdtemp(prefix="sm_stages_source_")
    shutil.copy2(SCRIPT_DIR / "stage_predictor.py", stage)
    # xgboost + imbalanced-learn for SMOTE oversampling of rare stages.
    # Tree SHAP is built into XGBoost's C++ core (no shap/numba package required).
    reqs = "xgboost>=2.0\nimbalanced-learn>=0.11\n"
    with open(os.path.join(stage, "requirements.txt"), "w") as f:
        f.write(reqs)
    return stage


def _make_connections_source_stage() -> str:
    """Copy connection predictor scripts to temp dir."""
    stage = tempfile.mkdtemp(prefix="sm_conn_source_")
    shutil.copy2(SCRIPT_DIR / "connection_predictor.py", stage)
    shutil.copy2(SCRIPT_DIR / "stage_predictor.py", stage)
    shutil.copy2(SCRIPT_DIR / "test_v2_mapping.py", stage)
    reqs = "xgboost>=2.0\nimbalanced-learn>=0.11\n"
    with open(os.path.join(stage, "requirements.txt"), "w") as f:
        f.write(reqs)
    return stage


def cmd_train_connections(args):
    """Launch SageMaker job to train connection predictor."""
    role = _resolve_role(args)
    session = Session()
    graphs_bucket = args.graphs_bucket or GRAPHS_BUCKET_DEFAULT

    image_uri = image_uris.retrieve(
        framework="sklearn", region=session.boto_region_name,
        version="1.2-1", image_scope="training", instance_type=args.instance)
    print(f"Image: {image_uri}")

    source_dir = _make_connections_source_stage()
    trainer = ModelTrainer(
        training_image=image_uri,
        source_code=SourceCode(
            source_dir=source_dir,
            entry_script="connection_predictor.py",
            requirements="requirements.txt",
        ),
        hyperparameters={
            "graphs-bucket": graphs_bucket,
            "graphs-prefix": "graphs/",
            "folds": str(args.folds),
            "test-pct": str(args.test_pct),
            "val-pct": str(args.val_pct),
            "vocab-version": args.vocab_version,
        },
        compute=Compute(instance_count=1, instance_type=args.instance, volume_size_in_gb=10),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=1800),
        base_job_name="xgb-connections",
        role=role,
    )
    print(f"Training connection model on s3://{graphs_bucket}/graphs/")
    try:
        trainer.train(wait=args.wait)
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Job: {job_name}")
    except Exception as e:
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Job {job_name}: {e}")
        if not args.wait:
            raise


def _make_edges_source_stage() -> str:
    """Copy edge predictor scripts + requirements to temp dir for SageMaker."""
    stage = tempfile.mkdtemp(prefix="sm_edges_source_")
    shutil.copy2(SCRIPT_DIR / "edge_predictor.py", stage)
    shutil.copy2(SCRIPT_DIR / "stage_predictor.py", stage)
    shutil.copy2(SCRIPT_DIR / "test_v2_mapping.py", stage)
    reqs = "xgboost>=2.0\nimbalanced-learn>=0.11\n"
    with open(os.path.join(stage, "requirements.txt"), "w") as f:
        f.write(reqs)
    return stage


def cmd_train_edges(args):
    """Launch SageMaker job to train XGBoost edge predictors (V2 vocabulary)."""
    role = _resolve_role(args)
    session = Session()
    graphs_bucket = args.graphs_bucket or GRAPHS_BUCKET_DEFAULT

    image_uri = image_uris.retrieve(
        framework="sklearn",
        region=session.boto_region_name,
        version="1.2-1",
        image_scope="training",
        instance_type=args.instance,
    )
    print(f"Image: {image_uri}")

    source_dir = _make_edges_source_stage()
    trainer = ModelTrainer(
        training_image=image_uri,
        source_code=SourceCode(
            source_dir=source_dir,
            entry_script="edge_predictor.py",
            requirements="requirements.txt",
        ),
        hyperparameters={
            "graphs-bucket": graphs_bucket,
            "graphs-prefix": "graphs/",
            "folds": str(args.folds),
            "min-support": str(args.min_support),
            "test-pct": str(args.test_pct),
            "val-pct": str(args.val_pct),
        },
        compute=Compute(
            instance_count=1,
            instance_type=args.instance,
            volume_size_in_gb=10,
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=7200),
        base_job_name="xgb-edges",
        role=role,
    )

    print(f"Training V2 edge models on s3://{graphs_bucket}/graphs/")
    print(f"Instance: {args.instance}, folds: {args.folds}, min_support: {args.min_support}")
    print(f"Split: {1-args.test_pct-args.val_pct:.0%} train / {args.val_pct:.0%} val / {args.test_pct:.0%} test")

    try:
        trainer.train(wait=args.wait)
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Job: {job_name}")
    except Exception as e:
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Job {job_name}: {e}")
        if not args.wait:
            raise


def cmd_train_stages(args):
    """Launch SageMaker job to train XGBoost stage predictors."""
    role = _resolve_role(args)
    session = Session()
    graphs_bucket = args.graphs_bucket or GRAPHS_BUCKET_DEFAULT

    image_uri = image_uris.retrieve(
        framework="sklearn",
        region=session.boto_region_name,
        version="1.2-1",
        image_scope="training",
        instance_type=args.instance,
    )
    print(f"Training image: {image_uri}")

    source_dir = _make_stages_source_stage()
    trainer = ModelTrainer(
        training_image=image_uri,
        source_code=SourceCode(
            source_dir=source_dir,
            entry_script="stage_predictor.py",
            requirements="requirements.txt",
        ),
        hyperparameters={
            "graphs-bucket": graphs_bucket,
            "graphs-prefix": "graphs/",
            "folds": str(args.folds),
            "min-support": str(args.min_support),
            "vocab-version": args.vocab_version,
        },
        compute=Compute(
            instance_count=1,
            instance_type=args.instance,
            volume_size_in_gb=10,
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=7200),
        base_job_name="xgb-stages",
        role=role,
    )

    print(f"Graphs source: s3://{graphs_bucket}/graphs/")
    print(f"Instance: {args.instance}, folds: {args.folds}, min_support: {args.min_support}")

    try:
        trainer.train(wait=args.wait)
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Run complete. Job: {job_name}")
    except Exception as e:
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Job {job_name} failed: {e}")
        if not args.wait:
            raise


def _make_build_index_source_stage() -> str:
    """Copy build-index script + requirements to a temp dir for SageMaker upload."""
    stage = tempfile.mkdtemp(prefix="sm_buildindex_source_")
    shutil.copy2(SCRIPT_DIR / "flowsheet_predictor.py", stage)
    shutil.copy2(SCRIPT_DIR / "build_index_entry.py", stage)
    shutil.copy2(SCRIPT_DIR / "requirements_gat.txt", os.path.join(stage, "requirements.txt"))
    return stage


def cmd_build_index(args):
    """Launch SageMaker job to build inference index from a trained GAT checkpoint."""
    role = _resolve_role(args)
    session = Session()
    graphs_bucket = args.graphs_bucket or GRAPHS_BUCKET_DEFAULT

    # Resolve model artifact URI from training job
    model_uri = args.model_uri
    if not model_uri and args.job_name:
        sm_client = boto3.client("sagemaker")
        desc = sm_client.describe_training_job(TrainingJobName=args.job_name)
        model_uri = desc["ModelArtifacts"]["S3ModelArtifacts"]
    if not model_uri:
        sys.exit("ERROR: Provide --model-uri or --job-name to locate the trained GAT checkpoint.")

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

    source_dir = _make_build_index_source_stage()
    trainer = ModelTrainer(
        training_image=image_uri,
        source_code=SourceCode(
            source_dir=source_dir,
            entry_script="build_index_entry.py",
            requirements="requirements.txt",
        ),
        hyperparameters={
            "graphs-bucket": graphs_bucket,
            "graphs-prefix": "graphs_v2/",
            "threshold": str(args.threshold),
        },
        compute=Compute(
            instance_count=1,
            instance_type=args.instance,
            volume_size_in_gb=50,
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=3600),
        base_job_name="gat-build-index",
        role=role,
    )

    print(f"Model checkpoint: {model_uri}")
    print(f"Graphs source: s3://{graphs_bucket}/graphs/")
    print(f"Instance: {args.instance}, threshold: {args.threshold}")

    try:
        trainer.train(
            input_data_config=[
                InputData(channel_name="model", data_source=model_uri),
            ],
            wait=args.wait,
        )
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Run complete. Job: {job_name}")
    except Exception as e:
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Job {job_name} failed: {e}")
        if not args.wait:
            raise


def _make_llm_predict_source_stage() -> str:
    """Copy LLM predictor scripts + requirements to temp dir for SageMaker."""
    stage = tempfile.mkdtemp(prefix="sm_llm_predict_source_")
    shutil.copy2(SCRIPT_DIR / "llm_predict_entry.py", stage)
    shutil.copy2(SCRIPT_DIR / "llm_predictor.py", stage)
    shutil.copy2(SCRIPT_DIR / "stage_predictor.py", stage)
    # Requirements: xgboost+imblearn for augment mode (no LLM deps needed —
    # Bedrock calls go through boto3, which is preinstalled on SageMaker)
    reqs = "xgboost>=2.0\nimbalanced-learn>=0.11\n"
    with open(os.path.join(stage, "requirements.txt"), "w") as f:
        f.write(reqs)
    return stage


def _make_llm_eval_source_stage() -> str:
    """Copy LLM eval scripts + requirements to temp dir for SageMaker."""
    stage = tempfile.mkdtemp(prefix="sm_llm_eval_source_")
    shutil.copy2(SCRIPT_DIR / "llm_eval_entry.py", stage)
    shutil.copy2(SCRIPT_DIR / "llm_predictor.py", stage)
    shutil.copy2(SCRIPT_DIR / "stage_predictor.py", stage)
    shutil.copy2(SCRIPT_DIR / "test_v2_mapping.py", stage)
    reqs = "xgboost>=2.0\nimbalanced-learn>=0.11\n"
    with open(os.path.join(stage, "requirements.txt"), "w") as f:
        f.write(reqs)
    return stage


def cmd_eval_llm(args):
    """Launch SageMaker job to evaluate LLM predictions against ground truth."""
    role = _resolve_role(args)
    session = Session()
    graphs_bucket = args.graphs_bucket or GRAPHS_BUCKET_DEFAULT

    image_uri = image_uris.retrieve(
        framework="sklearn",
        region=session.boto_region_name,
        version="1.2-1",
        image_scope="training",
        instance_type=args.instance,
    )
    print(f"Image: {image_uri}")

    source_dir = _make_llm_eval_source_stage()

    hyperparams = {
        "graphs-bucket": graphs_bucket,
        "graphs-prefix": "graphs/",
        "top-k": str(args.top_k),
        "mode": args.mode,
    }
    if args.model_id:
        hyperparams["model-id"] = args.model_id
    if args.max_docs > 0:
        hyperparams["max-docs"] = str(args.max_docs)

    # Longer timeout: ~180 docs × ~30s/doc = ~90 min for hybrid
    max_runtime = 7200 if args.max_docs == 0 else 3600

    trainer = ModelTrainer(
        training_image=image_uri,
        source_code=SourceCode(
            source_dir=source_dir,
            entry_script="llm_eval_entry.py",
            requirements="requirements.txt",
        ),
        hyperparameters=hyperparams,
        compute=Compute(
            instance_count=1,
            instance_type=args.instance,
            volume_size_in_gb=10,
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=max_runtime),
        base_job_name="llm-eval",
        role=role,
    )

    docs_str = f"{args.max_docs} docs" if args.max_docs > 0 else "all labeled docs"
    print(f"Evaluating LLM on {docs_str}")
    print(f"Provider: bedrock, top-k: {args.top_k}")
    print(f"Graphs: s3://{graphs_bucket}/graphs/")
    print(f"Instance: {args.instance}, max_runtime: {max_runtime}s")

    try:
        trainer.train(wait=args.wait)
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Job launched: {job_name}")
        if args.wait:
            _print_eval_results(job_name)
    except Exception as e:
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Job {job_name}: {e}")
        if not args.wait:
            raise


def _print_eval_results(job_name: str):
    """Download and display eval results from completed job."""
    try:
        sm = boto3.client("sagemaker")
        desc = sm.describe_training_job(TrainingJobName=job_name)
        model_uri = desc["ModelArtifacts"]["S3ModelArtifacts"]
        print(f"\nResults at: {model_uri}")

        import tarfile
        import tempfile
        parts = model_uri.replace("s3://", "").split("/", 1)
        s3 = boto3.client("s3")
        tmptar = tempfile.mktemp(suffix=".tar.gz")
        s3.download_file(parts[0], parts[1], tmptar)
        tmpdir = tempfile.mkdtemp()
        with tarfile.open(tmptar, "r:gz") as tar:
            tar.extractall(tmpdir)

        result_path = os.path.join(tmpdir, "llm_eval_results.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                results = json.load(f)
            m = results["metrics"]
            print(f"\n{'=' * 55}")
            print(f"LLM EVAL RESULTS ({m['num_labeled_docs']} docs)")
            print(f"{'=' * 55}")
            print(f"  macro-F1:  {m['macro_f1']:.4f}  "
                  f"(XGBoost: {results['comparison']['xgboost_macro_f1']}, "
                  f"delta: {results['comparison']['delta']:+.4f})")
            print(f"  micro-F1:  {m['micro_f1']:.4f}")
            print(f"  precision: {m['precision']:.4f}")
            print(f"  recall:    {m['recall']:.4f}")
            print(f"  time:      {m['total_time_seconds']:.0f}s "
                  f"({m['avg_time_per_doc']:.1f}s/doc)")
        os.unlink(tmptar)
    except Exception as e:
        print(f"(Could not read results: {e})")


def cmd_predict_llm(args):
    """Launch SageMaker job to run LLM flowsheet prediction."""
    role = _resolve_role(args)
    session = Session()
    graphs_bucket = args.graphs_bucket or GRAPHS_BUCKET_DEFAULT

    image_uri = image_uris.retrieve(
        framework="sklearn",
        region=session.boto_region_name,
        version="1.2-1",
        image_scope="training",
        instance_type=args.instance,
    )
    print(f"Image: {image_uri}")

    source_dir = _make_llm_predict_source_stage()

    hyperparams = {
        "graphs-bucket": graphs_bucket,
        "graphs-prefix": "graphs/",
        "head-grade": str(args.head_grade),
        "deposit-type": args.deposit_type,
        "ore-type": args.ore_type,
        "throughput-tpd": str(args.throughput_tpd),
        "mode": args.mode,
        "split": args.split,
        "top-k": str(args.top_k),
    }
    if args.model_id:
        hyperparams["model-id"] = args.model_id

    trainer = ModelTrainer(
        training_image=image_uri,
        source_code=SourceCode(
            source_dir=source_dir,
            entry_script="llm_predict_entry.py",
            requirements="requirements.txt",
        ),
        hyperparameters=hyperparams,
        compute=Compute(
            instance_count=1,
            instance_type=args.instance,
            volume_size_in_gb=10,
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=1800),
        base_job_name="llm-predict",
        role=role,
    )

    print(f"Prediction: {args.head_grade}% Cu, {args.deposit_type or 'any'} deposit, "
          f"{args.ore_type or 'any'} ore, {args.throughput_tpd} tpd")
    print(f"Mode: {args.mode}, Provider: bedrock")
    print(f"Graphs: s3://{graphs_bucket}/graphs/")
    print(f"Instance: {args.instance}")

    try:
        trainer.train(wait=args.wait)
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Job launched: {job_name}")
        if args.wait:
            # Download and display result
            sm = boto3.client("sagemaker")
            desc = sm.describe_training_job(TrainingJobName=job_name)
            model_uri = desc["ModelArtifacts"]["S3ModelArtifacts"]
            print(f"\nResults at: {model_uri}")
            # Try to read the summary
            try:
                import tarfile
                import tempfile
                parts = model_uri.replace("s3://", "").split("/", 1)
                s3 = boto3.client("s3")
                tmptar = tempfile.mktemp(suffix=".tar.gz")
                s3.download_file(parts[0], parts[1], tmptar)
                with tarfile.open(tmptar, "r:gz") as tar:
                    for member in tar.getmembers():
                        if member.name.endswith("prediction_summary.txt"):
                            f = tar.extractfile(member)
                            if f:
                                print(f.read().decode())
                os.unlink(tmptar)
            except Exception as e:
                print(f"(Could not read summary: {e})")
    except Exception as e:
        job_name = trainer.training_job_name if hasattr(trainer, "training_job_name") else "unknown"
        print(f"Job {job_name} failed: {e}")
        if not args.wait:
            raise


def cmd_status(args):
    """Check SageMaker job status."""
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


def cmd_download(args):
    """Download graph outputs from S3 to local directory."""
    sm_client = boto3.client("sagemaker")
    s3 = boto3.client("s3")

    desc = sm_client.describe_training_job(TrainingJobName=args.job_name)
    model_uri = desc["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"Model artifacts: {model_uri}")

    # Download tar.gz
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse S3 URI
    parts = model_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]

    local_tar = output_dir / "model.tar.gz"
    print(f"Downloading to {local_tar} ...")
    s3.download_file(bucket, key, str(local_tar))

    # Extract
    import tarfile
    print(f"Extracting to {output_dir} ...")
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(output_dir)

    local_tar.unlink()
    print(f"Done. Graphs saved to {output_dir}")

    # Show summary if present
    summary_path = output_dir / "graphs" / "graphs_summary.json"
    if summary_path.exists():
        import json
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"\nSummary:")
        print(f"  Total docs: {summary.get('total_docs', '?')}")
        print(f"  Docs with stages: {summary.get('docs_with_stages', '?')}")
        print(f"  Context-only: {summary.get('docs_context_only', '?')}")
        print(f"  Total nodes: {summary.get('total_nodes', '?')}")
        print(f"  Total edges: {summary.get('total_edges', '?')}")
        print(f"  Stage vocab size: {summary.get('stage_vocab_size', '?')}")


def main():
    parser = argparse.ArgumentParser(description="SageMaker orchestration for document context graphs")
    parser.add_argument("--bucket", type=str, default="", help="S3 bucket (default: from stack)")
    parser.add_argument("--role", type=str, default="", help="SageMaker execution role ARN")

    sub = parser.add_subparsers(dest="command")

    # build-graphs
    p_build = sub.add_parser("build-graphs", help="Launch graph building job on SageMaker")
    p_build.add_argument("--batch-size", type=int, default=8)
    p_build.add_argument("--relevance-threshold", type=float, default=0.5)
    p_build.add_argument("--instance", type=str, default="ml.g4dn.xlarge")
    p_build.add_argument("--model-uri", type=str, default="",
                         help="DeBERTa model S3 URI (default: trained model)")
    p_build.add_argument("--wait", action="store_true", help="Block until job completes")
    p_build.add_argument("--runs", type=int, default=1,
                         help="Number of sequential runs (100 docs each, auto-resume)")
    p_build.add_argument("--results-bucket", type=str, default="mineral-pipeline-pipeline",
                         help="S3 bucket containing pipeline result JSONs")
    p_build.add_argument("--checkpoint-prefix", type=str, default="graphs_v3/",
                         help="S3 prefix for graph outputs (default: graphs_v3/; use graphs_au/ for gold)")
    p_build.add_argument("--allowlist-s3-key", type=str, default="",
                         help="S3 key (in checkpoint bucket) of allowlist JSON; "
                              "graph_pipeline will only process docs whose result_keys appear there")

    # build-similarity
    p_sim = sub.add_parser("build-similarity", help="Launch cross-document similarity job on SageMaker")
    p_sim.add_argument("--top-k", type=int, default=20, help="Neighbors per document")
    p_sim.add_argument("--instance", type=str, default="ml.m5.large")
    p_sim.add_argument("--graphs-bucket", type=str, default="",
                       help="S3 bucket containing graphs (default: sagemaker default bucket)")
    p_sim.add_argument("--wait", action="store_true", help="Block until job completes")

    # train-gat
    p_gat = sub.add_parser("train-gat", help="Launch GAT flowsheet prediction training on SageMaker")
    p_gat.add_argument("--epochs", type=int, default=200)
    p_gat.add_argument("--lr", type=float, default=1e-3)
    p_gat.add_argument("--patience", type=int, default=30)
    p_gat.add_argument("--folds", type=int, default=5)
    p_gat.add_argument("--instance", type=str, default="ml.g4dn.xlarge")
    p_gat.add_argument("--graphs-bucket", type=str, default="",
                       help="S3 bucket containing graphs (default: sagemaker default bucket)")
    p_gat.add_argument("--wait", action="store_true", help="Block until job completes")

    # train-layout
    p_layout = sub.add_parser("train-layout", help="Launch layout GAT training on SageMaker")
    p_layout.add_argument("--epochs", type=int, default=500)
    p_layout.add_argument("--lr", type=float, default=1e-3)
    p_layout.add_argument("--instance", type=str, default="ml.g4dn.xlarge")
    p_layout.add_argument("--graphs-bucket", type=str, default="",
                          help="S3 bucket containing graphs (default: sagemaker default bucket)")
    p_layout.add_argument("--wait", action="store_true", help="Block until job completes")

    # train-connections
    p_conn = sub.add_parser("train-connections", help="Train connection predictor (which stage pairs are linked)")
    p_conn.add_argument("--folds", type=int, default=5)
    p_conn.add_argument("--test-pct", type=float, default=0.10)
    p_conn.add_argument("--val-pct", type=float, default=0.20)
    p_conn.add_argument("--instance", type=str, default="ml.m5.large")
    p_conn.add_argument("--graphs-bucket", type=str, default="")
    p_conn.add_argument("--vocab-version", type=str, default="v1", choices=["v1", "v2"])
    p_conn.add_argument("--wait", action="store_true")

    # train-edges
    p_edges = sub.add_parser("train-edges", help="Train V2 edge predictors (stage transitions)")
    p_edges.add_argument("--folds", type=int, default=5)
    p_edges.add_argument("--min-support", type=int, default=3)
    p_edges.add_argument("--test-pct", type=float, default=0.10)
    p_edges.add_argument("--val-pct", type=float, default=0.20)
    p_edges.add_argument("--instance", type=str, default="ml.m5.large")
    p_edges.add_argument("--graphs-bucket", type=str, default="")
    p_edges.add_argument("--wait", action="store_true")

    # train-stages
    p_stages = sub.add_parser("train-stages", help="Launch XGBoost stage prediction training on SageMaker")
    p_stages.add_argument("--folds", type=int, default=5)
    p_stages.add_argument("--min-support", type=int, default=3,
                          help="Minimum docs per stage to include in vocab")
    p_stages.add_argument("--vocab-version", type=str, default="v1",
                          choices=["v1", "v2"],
                          help="v1 (23 stages) or v2 (37 stages with sequence distinctions)")
    p_stages.add_argument("--instance", type=str, default="ml.m5.large")
    p_stages.add_argument("--graphs-bucket", type=str, default="",
                          help="S3 bucket containing graphs (default: sagemaker default bucket)")
    p_stages.add_argument("--wait", action="store_true", help="Block until job completes")

    # build-index
    p_idx = sub.add_parser("build-index", help="Build inference index from trained GAT checkpoint")
    p_idx.add_argument("--job-name", type=str, default="",
                       help="GAT training job name (to find model artifact)")
    p_idx.add_argument("--model-uri", type=str, default="",
                       help="Direct S3 URI to model.tar.gz (alternative to --job-name)")
    p_idx.add_argument("--threshold", type=float, default=0.5,
                       help="Stage prediction threshold")
    p_idx.add_argument("--instance", type=str, default="ml.g4dn.xlarge")
    p_idx.add_argument("--graphs-bucket", type=str, default="",
                       help="S3 bucket containing graphs")
    p_idx.add_argument("--wait", action="store_true", help="Block until job completes")

    # eval-llm
    p_eval = sub.add_parser("eval-llm", help="Evaluate LLM predictions against ground truth")
    p_eval.add_argument("--mode", type=str, default="hybrid",
                        choices=["standalone", "hybrid", "xgboost"],
                        help="hybrid=XGBoost+LLM, standalone=LLM only, xgboost=baseline")
    p_eval.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Which split to evaluate (default: val)")
    p_eval.add_argument("--model-id", type=str, default="",
                        help="Bedrock inference-profile ID (default: Claude Sonnet 4)")
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.add_argument("--max-docs", type=int, default=0,
                        help="Max labeled docs to eval (0=all, 10 for quick test)")
    p_eval.add_argument("--instance", type=str, default="ml.m5.large")
    p_eval.add_argument("--graphs-bucket", type=str, default="")
    p_eval.add_argument("--wait", action="store_true")

    # predict-llm
    p_llm = sub.add_parser("predict-llm", help="Run LLM flowsheet prediction on SageMaker")
    p_llm.add_argument("--head-grade", type=float, default=1.0,
                        help="Cu head grade percent (default: 1.0)")
    p_llm.add_argument("--deposit-type", type=str, default="",
                        help="Deposit type (porphyry, skarn, vms, etc.)")
    p_llm.add_argument("--ore-type", type=str, default="",
                        help="Ore type (sulfide, oxide, mixed)")
    p_llm.add_argument("--throughput-tpd", type=float, default=0,
                        help="Plant throughput in tonnes/day")
    p_llm.add_argument("--mode", type=str, default="standalone",
                        choices=["standalone", "augment"],
                        help="standalone (LLM only) or augment (XGBoost + LLM)")
    p_llm.add_argument("--top-k", type=int, default=5)
    p_llm.add_argument("--model-id", type=str, default="",
                        help="Bedrock inference-profile ID (default: Claude Sonnet 4)")
    p_llm.add_argument("--instance", type=str, default="ml.m5.large")
    p_llm.add_argument("--graphs-bucket", type=str, default="")
    p_llm.add_argument("--wait", action="store_true", help="Block until job completes")

    # status
    p_status = sub.add_parser("status", help="Check job status")
    p_status.add_argument("--job-name", type=str, required=True)

    # download
    p_download = sub.add_parser("download", help="Download graph outputs from S3")
    p_download.add_argument("--job-name", type=str, required=True)
    p_download.add_argument("--output-dir", type=str, default="classification/graphs")

    args = parser.parse_args()

    if args.command == "build-graphs":
        cmd_build_graphs(args)
    elif args.command == "build-similarity":
        cmd_build_similarity(args)
    elif args.command == "train-gat":
        cmd_train_gat(args)
    elif args.command == "train-layout":
        cmd_train_layout(args)
    elif args.command == "train-connections":
        cmd_train_connections(args)
    elif args.command == "train-edges":
        cmd_train_edges(args)
    elif args.command == "train-stages":
        cmd_train_stages(args)
    elif args.command == "build-index":
        cmd_build_index(args)
    elif args.command == "eval-llm":
        cmd_eval_llm(args)
    elif args.command == "predict-llm":
        cmd_predict_llm(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "download":
        cmd_download(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
