from pathlib import Path
from datetime import datetime

"""Centralized configuration for training and prediction."""


# execution mode
# 0 -> only prediction
# 1 -> only training
# 2 -> training followed by prediction
OBJECTS = ["prediction", "training", "both"]
OBJECT = 0

# ----- Basic experiment info -----
# Only change PROJECT_NAME for each run. The experiment directory must exist
# under ``experiments/`` and contain exactly one Excel file with the training and
# prediction data.
PROJECT_NAME = "example_project"

# ----- Derived paths based on the directory layout -----
BASE_DIR = Path("experiments") / PROJECT_NAME

FINGERPRINT_DIR = BASE_DIR / "fingerprints"
RESULTS_DIR = BASE_DIR / "results"

# fingerprint generation

# The Excel file is automatically detected under ``BASE_DIR``.
SMILES_INPUT_PATH = BASE_DIR


# training settings
MODELS = ["dt"]
FINGERPRINTS = ["MACCS"]

TRAIN_FILE_PATH = BASE_DIR

TRAIN_FP_PATH = FINGERPRINT_DIR
DATA_NAME = PROJECT_NAME

# prediction settings

PREDICT_LIST_PATH = BASE_DIR
MODEL_PATH_BASE = Path("../Final_model_save/ToxCast_v.4.2_model_total")
PREDICT_FP_PATH = FINGERPRINT_DIR
PREDICT_SMILES_PATH = BASE_DIR



def validate_paths():
    """Ensure required files and sheets exist for the experiment."""
    from toxcast_pkg.common import find_single_excel_file, check_required_sheets

    p = SMILES_INPUT_PATH
    if Path(p).is_dir():
        p = find_single_excel_file(p)
    check_required_sheets(p, ["data", "assay_list"])



