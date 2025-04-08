# utils/logging_utils.py

import logging

def setup_logger(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

def log_performance(model_key, seed, precision, recall, f1, accuracy):
    logging.info(f"Model: {model_key}, Seed: {seed}, Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}")
