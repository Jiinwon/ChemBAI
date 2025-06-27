#!/usr/bin/env python
"""Run training or prediction pipeline for a given dataset."""

from pathlib import Path
import argparse
import logging
import os
import pandas as pd
import joblib
from toxcast_pkg.smiles2fing import Smiles2Fing
import config

# base directories for input data and output results
input_root = config.INPUT_ROOT
output_root = config.OUTPUT_ROOT


def resolve_paths(dataset: str):
    """Return key paths derived from dataset name."""
    base = input_root / dataset
    smiles_file = base / "smiles.xlsx"
    fingerprint_dir = base / "fingerprints"
    result_file = output_root / "predictions.xlsx"
    return smiles_file, fingerprint_dir, result_file


def generate_fingerprints(smiles_file: Path, fingerprint_dir: Path):
    logging.info("Generating fingerprints from %s", smiles_file)
    if not smiles_file.exists():
        logging.error("SMILES file missing: %s", smiles_file)
        return
    df = pd.read_excel(smiles_file)
    os.makedirs(fingerprint_dir, exist_ok=True)
    for fp in ["MACCS", "Morgan", "RDKit", "Layered", "Pattern"]:
        ms_none_idx, fp_df = Smiles2Fing(df["SMILES"], fp)
        fp_df.to_csv(fingerprint_dir / f"{fp}.csv", index=False)
        pd.DataFrame(ms_none_idx).to_csv(
            fingerprint_dir / f"{fp}_dropidx.csv", index=False
        )
        logging.info("Saved %s fingerprints", fp)


def load_models(result_dir: Path):
    models = {}
    model_base = result_dir / "model_save_path"
    if not model_base.exists():
        logging.warning("Model directory not found: %s", model_base)
        return models
    for joblib_file in model_base.rglob("*.joblib"):
        models[joblib_file.stem] = joblib.load(joblib_file)
        logging.info("Loaded model %s", joblib_file.name)
    return models


def predict(models: dict, fingerprint_dir: Path, output_file: Path):
    if not models:
        logging.error("No models loaded; skipping prediction")
        return
    results = {}
    for fp_file in fingerprint_dir.glob("*.csv"):
        if fp_file.name.endswith("_dropidx.csv"):
            continue
        data = pd.read_csv(fp_file)
        for name, model in models.items():
            col = f"{name}_{fp_file.stem}"
            results[col] = model.predict(data)
    if results:
        out_df = pd.DataFrame(results)
        os.makedirs(output_file.parent, exist_ok=True)
        out_df.to_excel(output_file, index=False)
        logging.info("Predictions saved to %s", output_file)
    else:
        logging.warning("No prediction results to save")


def main():
    parser = argparse.ArgumentParser(description="Run pipeline")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    args = parser.parse_args()

    smiles, fp_dir, result_file = resolve_paths(args.dataset)
    generate_fingerprints(smiles, fp_dir)
    models = load_models(output_root / args.dataset)
    predict(models, fp_dir, result_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
