"""
Centralized configuration for the mineral processing pipeline.

All hardcoded paths, model names, and API keys are replaced with
environment variables that have sensible defaults matching existing behavior.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directory (project root)
BASE_DIR = Path(__file__).parent

# --- API keys ---
LLAMA_API_KEY = os.getenv("LLAMAPARSER_API", "")

# --- Paths ---
PDF_DIR = Path(os.getenv("PDF_DIR", str(BASE_DIR / "test_pdfs" / "main_pdfs")))
RESULTS_PATH = Path(os.getenv("RESULTS_PATH", str(BASE_DIR / "results.json")))
NORMALIZATION_DICT_PATH = Path(
    os.getenv("NORMALIZATION_DICT_PATH",
              str(BASE_DIR / "results_test_normalization.json"))
)
EXTRACTED_IMAGES_DIR = Path(
    os.getenv("EXTRACTED_IMAGES_DIR", str(BASE_DIR / "extracted_images"))
)
STAGE_NETWORK_OUTPUT = Path(
    os.getenv("STAGE_NETWORK_OUTPUT", str(BASE_DIR / "stage_network.png"))
)
TERMS_PATH = Path(os.getenv("TERMS_PATH", str(BASE_DIR.parent / "TERMS.md")))
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "data" / "pipeline.db"))

# --- AWS Bedrock (only LLM provider — OpenAI paths removed) ---
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL = os.getenv("BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-6")
BEDROCK_REFINEMENT_MODEL = os.getenv("BEDROCK_REFINEMENT_MODEL", "us.anthropic.claude-sonnet-4-6")
# Legacy aliases so code referring to LLM_MODEL still works.
LLM_MODEL = BEDROCK_MODEL
LLM_REFINEMENT_MODEL = BEDROCK_REFINEMENT_MODEL

# --- Feature flags ---
ENABLE_CLIP = os.getenv("ENABLE_CLIP", "true").lower() in ("true", "1", "yes")
MAX_PDFS = int(os.getenv("MAX_PDFS", "5"))
EXTRACTION_METHOD = os.getenv("EXTRACTION_METHOD", "terms")  # "terms" or "refinement"

# --- PDF extraction defaults ---
FLOWSHEET_START_PAGE = int(os.getenv("FLOWSHEET_START_PAGE", "5"))
IMAGE_START_PAGE = int(os.getenv("IMAGE_START_PAGE", "5"))
MAX_ASPECT_RATIO = float(os.getenv("MAX_ASPECT_RATIO", "2.4"))
