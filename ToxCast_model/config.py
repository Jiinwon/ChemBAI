from pathlib import Path

"""Centralized configuration for training and prediction."""

# execution mode: 0 -> prediction, 1 -> training
OBJECTS = ["prediction", "training"]
OBJECT = 0

# common directories
INPUT_ROOT = Path("data")
OUTPUT_ROOT = Path("results")

# fingerprint generation
SMILES_INPUT_PATH = "./data/example_data_DBPs/for_train/example_DBPs_ER.xlsx"
FINGERPRINT_OUTPUT_DIR = "./data/example_data_DBPs/for_train/fingerprints"

# training settings
MODELS = ["dt"]
FINGERPRINTS = ["MACCS"]
TRAIN_FILE_PATH = "./data/example_data_DBPs/for_train/example_DBPs_ER.xlsx"
TRAIN_FP_PATH = "./data/example_data_DBPs/for_train/fingerprint_outputs"
DATA_NAME = "example_DBPs_ER"

# prediction settings
PREDICT_LIST_PATH = "./prediction/example_prediction_DBPs/example_assay_list_ER.xlsx"
MODEL_PATH_BASE = "../Final_model_save/ToxCast_v.4.2_model_total"
PREDICT_FP_PATH = "./data/example_data_DBPs/for_predict/fingerprints"
PREDICT_SMILES_PATH = "./data/example_data_DBPs/for_predict/example_DBPs_for_pred.xlsx"


