# ChemBAI

This repository contains utilities for training and predicting toxicological activity using the ToxCast v4.1 data set.

## Features

- Molecular fingerprints: **MACCS**, **Morgan**, **RDKit**, **Pattern**, **Layered**
- Algorithms: Decision Tree, Logistic Regression, Gradient Boost Tree, XGBoost, Random Forest
- Centralised configuration via `ToxCast_model/config.py`
- Single entry script `run_pipeline.sh` to generate fingerprints, train models and run predictions
- Prediction runs save results in timestamped folders and write a `metadata.json` summary in `experiments/<PROJECT_NAME>/results/`

## Project layout

```
Final_model_save/           # Pretrained models
ToxCast_model/              # Main source code
├─ experiments/             # Input and output of each experiment
├─ prediction/              # Prediction utilities
├─ run/                     # Training scripts (dt.py, rf.py, ...)
└─ toxcast_pkg/             # Helper modules
```

## Quick start

1. Prepare an Excel file containing the `data` and `assay_list` sheets.
2. Copy `Template/template_for_predict(PROJECT_NAME)` into `ToxCast_model/experiments/` and rename the folder to your project name.
3. Place the Excel file from step&nbsp;1 inside this new folder.
4. Edit `ToxCast_model/config.py` and set `PROJECT_NAME` to the folder name from step&nbsp;2.
5. Set `OBJECT = 0` in `config.py` for prediction mode.
6. Set `MODEL_SELECTION = 0` in `config.py` for best F1 mode
7. Run the pipeline from the project root:

```bash
bash run_pipeline.sh
```

### Local usage

If Bash is not available, run the same steps using the Python helper:

```bash
python run_local.py download-template --out .
python run_local.py predict
```

For an interactive option, launch the simple GUI:

```bash
python run_local_gui.py
```

Use the buttons to download the template, select your filled Excel file and run
the prediction pipeline on your local machine.

Fingerprints are generated only once and stored under `experiments/<PROJECT_NAME>/fingerprints/`. Prediction results are saved under `experiments/<PROJECT_NAME>/results/<timestamp>/`, and a cumulative `metadata.json` is written to `experiments/<PROJECT_NAME>/results/`.

### Building standalone binaries

Install `pyinstaller` and run the helper script to create an executable under the
`Release` directory. The script bundles the `Template` and `ToxCast_model`
folders so the program can be distributed without the rest of the repository.
Build the executable on each platform you want to support:

```bash
pip install pyinstaller
python build_release.py
```

On macOS, running `python build_release.py` creates `ChemBAI_Predictor` (or a
`.app` bundle depending on your PyInstaller version) inside `Release/`. Double
click this file to launch the GUI. Use it to download the template, select your
input file and run predictions locally.

You must build the binary on each target platform (macOS or Windows) because the
executables are platform specific.

## Environment setup

Install dependencies with conda using `environment.yml` or via `pip install -r requirements.txt`.

## Training scripts

The `ToxCast_model/run` directory contains standalone scripts for each algorithm. They perform cross-validation to select the best model and save it as a joblib file for later prediction.

## Prediction

`ToxCast_model/prediction/Predict_data.py` loads trained models specified in `config.py` and generates predictions for each assay. The script appends the original SMILES strings and writes an Excel file with the predictions.

## License

This project is licensed under the terms of the [MIT License](LICENSE).
