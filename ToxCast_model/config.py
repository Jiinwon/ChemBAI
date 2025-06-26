# Configuration file for training and prediction

# ----- Execution mode -----
# Choose between prediction or training.
# OBJECTS[0] -> "prediction", OBJECTS[1] -> "training"
OBJECTS = ["prediction", "training"]
# Set OBJECT to 0 for prediction or 1 for training
OBJECT = 0

# ----- Fingerprint generation -----
# Source Excel file containing SMILES for fingerprint generation
SMILES_INPUT_PATH = "./data/example_data_DBPs/for_train/example_DBPs_ER.xlsx"
# Directory where fingerprint CSV files will be written
FINGERPRINT_OUTPUT_DIR = "./data/example_data_DBPs/for_train/fingerprints"

# ----- Training -----
# Models to train
MODELS = ['dt']  # e.g. ['xgb', 'gbt', 'rf', 'logistic']
# Fingerprint types used for training
FINGERPRINTS = ['MACCS']  # e.g. ['Morgan', 'RDKit', 'Layered', 'Pattern']
# Training Excel file path
TRAIN_FILE_PATH = "./data/example_data_DBPs/for_train/example_DBPs_ER.xlsx"
# Directory containing fingerprint CSVs for training
TRAIN_FP_PATH = "./data/example_data_DBPs/for_train/fingerprint_outputs"
# Name used for saving results
DATA_NAME = "example_DBPs_ER"

# ----- Prediction -----
# Excel file specifying models to load for prediction
PREDICT_LIST_PATH = "./prediction/example_prediction_DBPs/example_assay_list_ER.xlsx"
# Base directory where trained models are stored
MODEL_PATH_BASE = "./results/example_DBPs_ER/2025-03-25/model_save_path"
# Directory containing fingerprint CSVs for chemicals to predict
PREDICT_FP_PATH = "./data/example_data_DBPs/for_predict/fingerprints"
# Excel file with SMILES for the chemicals to predict
PREDICT_SMILES_PATH = "./data/example_data_DBPs/for_predict/example_DBPs_for_pred.xlsx"


