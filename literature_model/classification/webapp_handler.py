"""Lambda entrypoint for the Flowsheet Predictor web app.

Wraps the Flask app at classification/app.py with apig-wsgi so it can be
invoked via a Lambda Function URL (Lambda v2 / API Gateway HTTP-API event
shape). Static + dynamic routes both go through Flask.

Deployed via WebAppFunction in template.yaml (AuthType: NONE for public).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# At runtime this file lives at /var/task/webapp_handler.py (Lambda); the
# repo layout is mirrored under /var/task/classification/ + /var/task/quantitative/.
# When running locally from the repo (classification/webapp_handler.py), the
# script's parent IS classification/ — the loop below handles both cases.
SELF_DIR = Path(__file__).resolve().parent
CANDIDATES = [
    SELF_DIR,
    SELF_DIR / "classification",
    SELF_DIR / "quantitative",
    SELF_DIR.parent / "quantitative",
]
for sub in CANDIDATES:
    if sub.exists() and str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

# Force matplotlib to use a non-GUI backend before any Flask import path
# triggers a render.
os.environ.setdefault("MPLBACKEND", "Agg")
# Image cache must live under /tmp on Lambda (read-only filesystem otherwise).
os.environ.setdefault("PFS_IMG_DIR", "/tmp/pfs_cache")
Path(os.environ["PFS_IMG_DIR"]).mkdir(parents=True, exist_ok=True)

from app import app as flask_app  # noqa: E402
from apig_wsgi import make_lambda_handler  # noqa: E402

handler = make_lambda_handler(flask_app)
