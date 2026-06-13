"""Configuration for the motif clustering task."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "signals_data.csv"

# Motif extraction
USE_ADAPTIVE_THRESHOLD = True
GLOBAL_THRESHOLD = 110.0
STEADY_PERCENTILE = 80
THRESHOLD_OFFSET = 10.0
MIN_LEN = 20
MAX_GAP = 5

# Shape representation
FIXED_LEN = 120
N_PCA_COMPS = 10

# Length-constrained clustering
# The task explicitly says that strongly different motif lengths should not be mixed.
# Therefore the final algorithm first separates motifs into temporal-length bands,
# then separates shape within the longer bands.
N_LENGTH_BANDS = 4
SPLIT_SHAPE_BANDS = [2, 3]  # medium and long length bands are split into early/late shape families
SHAPE_SPLIT_K = 2

# Evaluation/runtime
SAMPLE_SIZE = 3000
RANDOM_STATE = 42
BATCH_SIZE = 1024
