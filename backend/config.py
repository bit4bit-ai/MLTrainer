import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPORTS_DIR = PROJECT_ROOT / "exports"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
DEMO_DATA_DIR = BASE_DIR / "demo_data"

# Ensure runtime directories exist
for p in [DATA_DIR, EXPORTS_DIR, CHECKPOINTS_DIR, DEMO_DATA_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Server Config
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*"
]

SUPPORTED_TASKS = [
    "classification",
    "detection",
    "anomaly",
    "segmentation"
]
