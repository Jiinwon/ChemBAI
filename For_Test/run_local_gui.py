#!/usr/bin/env python3
"""Simple GUI for running predictions locally without Bash.

This program provides buttons to download the input template, select the
filled Excel file, and run the prediction pipeline using ``run_local``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from For_Test import run_local


def download_template() -> None:
    """Ask the user where to copy the template."""
    out_dir = filedialog.askdirectory(title="Select destination folder")
    if out_dir:
        run_local.download_template(out_dir)


def select_input() -> None:
    """Prompt for the filled Excel file."""
    file_path = filedialog.askopenfilename(
        title="Select filled template", filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    if file_path:
        selected_file.set(file_path)


def run_prediction() -> None:
    """Copy the selected file to the experiment directory and run prediction."""
    fp = selected_file.get()
    if not fp:
        messagebox.showerror("Error", "Please select an input file first")
        return
    import ToxCast_model.config as cfg

    dest_dir = Path("../ToxCast_model") / "experiments" / cfg.PROJECT_NAME
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(fp, dest_dir)
    try:
        run_local.run_prediction()
    except Exception as e:  # pragma: no cover - GUI only
        messagebox.showerror("Error", str(e))
    else:
        messagebox.showinfo("Finished", "Prediction completed")


app = tk.Tk()
app.title("ChemBAI Predictor")

selected_file = tk.StringVar(app)

btn_dl = tk.Button(app, text="Download Template", command=download_template)
btn_dl.pack(fill="x", padx=10, pady=5)

btn_select = tk.Button(app, text="Select Input", command=select_input)
btn_select.pack(fill="x", padx=10, pady=5)

label_file = tk.Label(app, textvariable=selected_file)
label_file.pack(fill="x", padx=10)

btn_run = tk.Button(app, text="Run Prediction", command=run_prediction)
btn_run.pack(fill="x", padx=10, pady=5)

app.mainloop()
