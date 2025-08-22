# utils/model_utils.py

import joblib
import logging

def save_best_model(model, best_f1, save_path):
    joblib.dump(model, save_path)
    logging.info(f"Best model saved with F1 score: {best_f1}")
