"""Central configuration (paths + pipeline constants)."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"
MODELS_DIR = ROOT / "models"
CONFIGS_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
LOG_DIR = RESULTS_DIR / "logs"
MAPS_DIR = ROOT / "maps"
CACHE_DIR = ROOT / "cache"

SEED = 42

# Candidate generation
CAND_RADIUS_M = 50.0      # search radius around each GPS fix
CAND_MAX_K = 12           # max candidates kept per fix (nearest first)
DENSIFY_STEP_M = 10.0     # arc sampling step for the KD-tree index

# Feature engineering
HEADING_HALF_WINDOW = 2   # heading from p[t-k] -> p[t+k]
CONTEXT_HALF_WINDOW = 3   # window for temporal distance smoothing

for _d in (MODELS_DIR, CONFIGS_DIR, RESULTS_DIR, FIG_DIR, LOG_DIR, MAPS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
