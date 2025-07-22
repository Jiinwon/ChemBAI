import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import joblib

# Allow importing from ToxCast_model package
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "ToxCast_model"))

from toxcast_pkg.smiles2fing import Smiles2Fing

MODEL_BASE = "/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/Final_model_save"

class ToxPredictorApp:
    """Simple GUI application for running toxicity predictions."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ToxPredictor v1.0")

        self.file_path = ""

        # UI setup
        select_btn = tk.Button(root, text="Select File", command=self.select_file)
        select_btn.pack(pady=5)

        start_btn = tk.Button(root, text="Start Prediction", command=self.start_prediction)
        start_btn.pack(pady=5)

        self.status_var = tk.StringVar(value="No file selected")
        status_label = tk.Label(root, textvariable=self.status_var)
        status_label.pack(pady=5)

        self.progress = ttk.Progressbar(root, length=300)
        self.progress.pack(pady=5)

    def select_file(self) -> None:
        """Display file chooser to select the Excel template."""
        path = filedialog.askopenfilename(title="Select template.xlsx", filetypes=[("Excel files", "*.xlsx")])
        if path:
            self.file_path = path
            self.status_var.set(os.path.basename(path))

    def start_prediction(self) -> None:
        """Run predictions using the selected Excel file."""
        if not self.file_path:
            messagebox.showerror("Error", "Please select the Excel template file.")
            return

        try:
            data_df = pd.read_excel(self.file_path, sheet_name="data")
            assay_df = pd.read_excel(self.file_path, sheet_name="assay_list")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to read Excel file: {exc}")
            return

        self.progress["maximum"] = len(data_df)

        for idx, row in data_df.iterrows():
            try:
                assay = row["assay_name"]
                smiles = row["SMILES"]
                cfg_row = assay_df[assay_df["assay_name"] == assay]
                if cfg_row.empty:
                    raise FileNotFoundError(f"Assay '{assay}' not found in assay_list sheet")
                mf = cfg_row.iloc[0]["MF"]
                alg = cfg_row.iloc[0]["Algorithm"]

                model_path = os.path.join(MODEL_BASE, assay, mf, f"{alg}.joblib")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(model_path)

                model = joblib.load(model_path)

                none_idx, fp_df = Smiles2Fing([smiles], mf)
                if none_idx:
                    raise ValueError(f"Invalid SMILES at row {idx+1}")

                pred = model.predict(fp_df)[0]
                data_df.loc[idx, assay] = pred

                self.status_var.set(f"Processed {idx+1}/{len(data_df)}")
                self.progress["value"] = idx + 1
                self.root.update_idletasks()
            except Exception as exc:
                messagebox.showerror("Error", str(exc))
                return

        try:
            with pd.ExcelWriter(self.file_path, engine="openpyxl") as writer:
                data_df.to_excel(writer, sheet_name="data", index=False)
                assay_df.to_excel(writer, sheet_name="assay_list", index=False)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save Excel file: {exc}")
            return

        self.status_var.set("Prediction completed")
        messagebox.showinfo("Done", "All predictions completed successfully")


def main() -> None:
    root = tk.Tk()
    app = ToxPredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
