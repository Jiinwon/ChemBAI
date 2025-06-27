from pathlib import Path
from datetime import datetime

"""Centralized configuration for training and prediction."""


# execution mode: 0 -> prediction, 1 -> training
OBJECTS = ["prediction", "training"]
OBJECT = 0


# ----- Basic experiment info -----
# Only change PROJECT_NAME (and optionally EXPERIMENT_DATE) for each run.
PROJECT_NAME = "example_project"
EXPERIMENT_DATE = datetime.now().strftime("%y%m%d")

# ----- Derived paths based on the directory layout -----
BASE_DIR = Path("experiments") / EXPERIMENT_DATE / PROJECT_NAME
DATA_FILE = BASE_DIR / f"{PROJECT_NAME}.xlsx"
FINGERPRINT_DIR = BASE_DIR / "fingerprints"
RESULTS_DIR = BASE_DIR / "results"

# fingerprint generation
SMILES_INPUT_PATH = DATA_FILE
FINGERPRINT_OUTPUT_DIR = FINGERPRINT_DIR

# training settings
MODELS = ["dt"]
FINGERPRINTS = ["MACCS"]

TRAIN_FILE_PATH = DATA_FILE
TRAIN_FP_PATH = FINGERPRINT_DIR
DATA_NAME = PROJECT_NAME

# prediction settings
PREDICT_LIST_PATH = BASE_DIR / "assay_list.xlsx"
MODEL_PATH_BASE = Path("../Final_model_save/ToxCast_v.4.2_model_total")
PREDICT_FP_PATH = FINGERPRINT_DIR
PREDICT_SMILES_PATH = DATA_FILE



