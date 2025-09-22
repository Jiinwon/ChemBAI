from pathlib import Path
from datetime import datetime

"""Centralized configuration for training and prediction."""


# execution mode
# 0 -> only prediction
# 1 -> only training
OBJECTS = ["prediction", "training"]
OBJECT = 1
# model selection
# 0 -> best F1 model (assay_name only)
# 1 -> user-specified model and fingerprint
MODEL_SELECTION_OPTIONS = ["best_f1", "model+mf"]
MODEL_SELECTION = 1
# model version
# 1 -> original ToxCast_model
# 2 -> ToxCast_model_v.2
# 3 -> ToxCast_model_v.3 (pre-split train/val/test data)
VERSION = 3
# ----- Basic experiment info -----
# Only change PROJECT_NAME for each run. The experiment directory must exist
# under ``experiments/`` and contain the required inputs for the selected version.
PROJECT_NAME = "KGML"
REF_FILE_PATH = Path('/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/ToxCast_model/data/ToxCast_v.4.1_v.2/fingerprints')



# ----- Derived paths based on the directory layout -----
# Resolve paths relative to this file so experiments live in the top-level
# ``ToxCast_model/experiments`` regardless of the selected VERSION or the
# current working directory.
ROOT_DIR = Path(__file__).resolve().parent
BASE_DIR = ROOT_DIR / "experiments" / OBJECTS[OBJECT] / PROJECT_NAME

# training settings
MODELS = ["dt", "rf", "xgb", "gbt", "logistic"]
FINGERPRINTS = ["MACCS", "Morgan", "RDKit", "Pattern", "Layered"]

if VERSION == 3:
    TRAIN_DIR = BASE_DIR / "train"
    VAL_DIR = BASE_DIR / "val"
    TEST_DIR = BASE_DIR / "test"

    TRAIN_FILE_PATH = TRAIN_DIR / "train_df.csv"
    VAL_FILE_PATH = VAL_DIR / "val_df.csv"
    TEST_FILE_PATH = TEST_DIR / "test_df.csv"

    TRAIN_FP_PATH = TRAIN_DIR / "fingerprints"
    VAL_FP_PATH = VAL_DIR / "fingerprints"
    TEST_FP_PATH = TEST_DIR / "fingerprints"

    FINGERPRINT_DIR = TRAIN_FP_PATH
    FINGERPRINT_OUTPUT_DIR = FINGERPRINT_DIR
    RESULTS_DIR = BASE_DIR / "results"

    SMILES_INPUT_PATH = TRAIN_FILE_PATH

    # prediction defaults
    PREDICT_SPLIT = "test"
    PREDICT_LIST_PATH = TEST_FILE_PATH
    PREDICT_FP_PATH = TEST_FP_PATH
    PREDICT_SMILES_PATH = TEST_FILE_PATH
else:
    FINGERPRINT_DIR = BASE_DIR / "fingerprints"
    FINGERPRINT_OUTPUT_DIR = FINGERPRINT_DIR
    RESULTS_DIR = BASE_DIR / "results"

    # fingerprint generation
    # The Excel file is automatically detected under ``BASE_DIR``.
    SMILES_INPUT_PATH = BASE_DIR

    TRAIN_FILE_PATH = BASE_DIR
    VAL_FILE_PATH = BASE_DIR
    TEST_FILE_PATH = BASE_DIR

    TRAIN_FP_PATH = FINGERPRINT_DIR
    VAL_FP_PATH = FINGERPRINT_DIR
    TEST_FP_PATH = FINGERPRINT_DIR

    PREDICT_LIST_PATH = BASE_DIR
    PREDICT_FP_PATH = FINGERPRINT_DIR
    PREDICT_SMILES_PATH = BASE_DIR

DATA_NAME = PROJECT_NAME


# base directories for each selection mode
MODEL_PATH_BASE_0 = Path("../Final_model_save/ToxCast_model(F1)")
MODEL_PATH_BASE_1 = Path("../Final_model_save/ToxCast_v.4.2_model_total")


def get_model_base():
    """Return the base directory for prediction models."""
    return MODEL_PATH_BASE_0 if MODEL_SELECTION == 0 else MODEL_PATH_BASE_1



def validate_paths():
    """Ensure required input files exist for the selected version."""

    if VERSION == 3:
        required_paths = [
            TRAIN_FILE_PATH,
            VAL_FILE_PATH,
            TEST_FILE_PATH,
            TRAIN_FP_PATH,
            VAL_FP_PATH,
            TEST_FP_PATH,
        ]
        for path in required_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Required path missing for VERSION=3: {path}")
    else:
        from toxcast_pkg.common import find_single_excel_file, check_required_sheets

        p = SMILES_INPUT_PATH
        if Path(p).is_dir():
            p = find_single_excel_file(p)
        check_required_sheets(p, ["data", "assay_list"])



