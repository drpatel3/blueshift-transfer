"""Reassemble the gitignored classification/standalone/ distribution bundle.

The standalone bundle is a self-contained predictor (val macro-F1=0.86) meant
to be copied to a fresh machine. It's not checked into git — this script
rebuilds it from canonical sources whenever needed.

Sources:
- Template files (this repo): classification/standalone_template/
  - predict.py, llm_units.py, README.md, requirements.txt
- Reused library code (this repo):
  - classification/pfs_visual.py -> standalone/pfs_visual.py
  - quantitative/eval.py         -> standalone/eval.py
  - quantitative/similarity.py   -> standalone/similarity.py
  - quantitative/grade_map.json  -> standalone/grade_map.json
- Model checkpoint (this repo): classification/xgb_checkpoint_v2/
  - 8 files mirrored into standalone/xgb_checkpoint_v2/

Usage:
    python classification/build_standalone_bundle.py
    python classification/build_standalone_bundle.py --verify   # list expected files
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSIFICATION_DIR = REPO_ROOT / "classification"
QUANTITATIVE_DIR = REPO_ROOT / "quantitative"
TEMPLATE_DIR = CLASSIFICATION_DIR / "standalone_template"
BUNDLE_DIR = CLASSIFICATION_DIR / "standalone"
CHECKPOINT_SRC = CLASSIFICATION_DIR / "xgb_checkpoint_v2"
CHECKPOINT_DST = BUNDLE_DIR / "xgb_checkpoint_v2"

TEMPLATE_FILES = ["predict.py", "llm_units.py", "README.md", "requirements.txt"]

REUSED_SOURCES = {
    "pfs_visual.py":   CLASSIFICATION_DIR / "pfs_visual.py",
    "eval.py":         QUANTITATIVE_DIR / "eval.py",
    "similarity.py":   QUANTITATIVE_DIR / "similarity.py",
    "grade_map.json":  QUANTITATIVE_DIR / "grade_map.json",
}

CHECKPOINT_FILES = [
    "doc_edges.json",
    "doc_tables.json",
    "oof_predictions.json",
    "stage_vocab.json",
    "test_predictions.json",
    "train_joint_support.json",
    "train_transition_counts.json",
    "val_predictions.json",
]

ENV_EXAMPLE_BODY = (
    "# Optional Bedrock overrides — remove the # to activate\n"
    "# LLM_UNITS_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0\n"
    "# LLM_UNITS_REGION=us-east-1\n"
)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_env_example() -> None:
    (BUNDLE_DIR / ".env.example").write_text(ENV_EXAMPLE_BODY, encoding="utf-8")


def _expected_files() -> list[Path]:
    files = [BUNDLE_DIR / name for name in TEMPLATE_FILES]
    files += [BUNDLE_DIR / name for name in REUSED_SOURCES]
    files += [CHECKPOINT_DST / name for name in CHECKPOINT_FILES]
    files.append(BUNDLE_DIR / ".env.example")
    return files


def build() -> None:
    for name in TEMPLATE_FILES:
        src = TEMPLATE_DIR / name
        if not src.exists():
            sys.exit(f"Missing template file: {src}")
        _copy(src, BUNDLE_DIR / name)

    for name, src in REUSED_SOURCES.items():
        if not src.exists():
            sys.exit(f"Missing canonical source: {src}")
        _copy(src, BUNDLE_DIR / name)

    for name in CHECKPOINT_FILES:
        src = CHECKPOINT_SRC / name
        if not src.exists():
            sys.exit(f"Missing checkpoint file: {src}")
        _copy(src, CHECKPOINT_DST / name)

    _write_env_example()

    print(f"Built bundle at {BUNDLE_DIR}")
    print(f"  {len(TEMPLATE_FILES)} template files + "
          f"{len(REUSED_SOURCES)} reused sources + "
          f"{len(CHECKPOINT_FILES)} checkpoint files + .env.example")


def verify() -> None:
    missing = [p for p in _expected_files() if not p.exists()]
    if missing:
        for p in missing:
            print(f"MISSING: {p}")
        sys.exit(1)
    print(f"All {len(_expected_files())} expected files present in {BUNDLE_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="Check the bundle is complete without rebuilding")
    args = parser.parse_args()

    if args.verify:
        verify()
    else:
        build()


if __name__ == "__main__":
    main()
