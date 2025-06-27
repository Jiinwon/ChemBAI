# ChemBAI

This repository contains utilities for training and predicting toxicological activity using the ToxCast v4.1 data set.

## Features

- Molecular fingerprints: **MACCS**, **Morgan**, **RDKit**, **Pattern**, **Layered**
- Algorithms: Decision Tree, Logistic Regression, Gradient Boost Tree, XGBoost, Random Forest
- Centralised configuration via `ToxCast_model/config.py`
- Single entry script `run_pipeline.sh` to generate fingerprints, train models and run predictions
- Prediction runs save results in timestamped folders and write a `metadata.json` summary

## Project layout

```
Final_model_save/           # Pretrained models
HowToPredict.md             # Additional docs
ToxCast_model/              # Main source code
├─ experiments/             # Input and output of each experiment
├─ prediction/              # Prediction utilities
├─ run/                     # Training scripts (dt.py, rf.py, ...)
└─ toxcast_pkg/             # Helper modules
```

## Quick start

1. Create `ToxCast_model/experiments/<PROJECT_NAME>/` and place a single Excel file containing the `data` and `assay_list` sheets inside the folder.
2. Open `ToxCast_model/config.py` and set `PROJECT_NAME` and the execution mode `OBJECT` (0 for prediction, 1 for training).
3. Run the pipeline from the project root:

```bash
bash run_pipeline.sh
```

Fingerprints are generated only once and stored under `experiments/<PROJECT_NAME>/fingerprints/`. Prediction results are saved under `experiments/<PROJECT_NAME>/results/<timestamp>/` with `metadata.json` summarising the run.

## Environment setup

Install dependencies with conda using `environment.yml` or via `pip install -r requirements.txt`.

## Training scripts

The `ToxCast_model/run` directory contains standalone scripts for each algorithm. They perform cross-validation to select the best model and save it as a joblib file for later prediction.

## Prediction

`ToxCast_model/prediction/Predict_data.py` loads trained models specified in `config.py` and generates predictions for each assay. The script appends the original SMILES strings and writes an Excel file with the predictions.

## License

This project is licensed under the terms of the [MIT License](LICENSE).
