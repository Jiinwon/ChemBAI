from setuptools import setup

APP = ['ToxPredictor.py']
DATA_FILES = ['template.xlsx', 
              ('Final_model_save', ['Final_model_save/ACEA_AR_agonist_80hr_Layered_dt/ACEA_AR_agonist_80hr_best_model_Layered_dt.joblib'])]
OPTIONS = {
    'argv_emulation': True,
    'packages': ['pandas', 'joblib', 'openpyxl', 'tkinter'],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)