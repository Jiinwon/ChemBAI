from pathlib import Path
from datetime import datetime

"""Centralized configuration for training and prediction."""


# execution mode
# 0 -> only prediction
# 1 -> only training
OBJECTS = ["prediction", "training"]
OBJECT = 0
# model selection
# 0 -> best F1 model (assay_name only)
# 1 -> user-specified model and fingerprint
MODEL_SELECTION_OPTIONS = ["best_f1", "model+mf"]
MODEL_SELECTION = 1
# model version
# 1 -> original ToxCast_model
# 2 -> ToxCast_model_v.2
VERSION = 1
# ----- Basic experiment info -----
# Only change PROJECT_NAME for each run. The experiment directory must exist
# under ``experiments/`` and contain exactly one Excel file with the training and
# prediction data.
PROJECT_NAME = "NGRA"
REF_FILE_PATH = '/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/ToxCast_model/data/ToxCast_v.4.1_v.2/fingerprints'



# ----- Derived paths based on the directory layout -----
BASE_DIR = Path("experiments") / OBJECTS[OBJECT] / PROJECT_NAME

FINGERPRINT_DIR = BASE_DIR / "fingerprints"
FINGERPRINT_OUTPUT_DIR = FINGERPRINT_DIR
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


# base directories for each selection mode
MODEL_PATH_BASE_0 = Path("../Final_model_save/ToxCast_model(F1)")
MODEL_PATH_BASE_1 = Path("../Final_model_save/ToxCast_v.4.2_model_total")

PREDICT_FP_PATH = FINGERPRINT_DIR
PREDICT_SMILES_PATH = BASE_DIR


def get_model_base():
    """Return the base directory for prediction models."""
    return MODEL_PATH_BASE_0 if MODEL_SELECTION == 0 else MODEL_PATH_BASE_1



def validate_paths():
    """Ensure required files and sheets exist for the experiment."""
    from toxcast_pkg.common import find_single_excel_file, check_required_sheets

    p = SMILES_INPUT_PATH
    if Path(p).is_dir():
        p = find_single_excel_file(p)
    check_required_sheets(p, ["data", "assay_list"])



