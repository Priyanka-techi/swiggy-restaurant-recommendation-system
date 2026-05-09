"""
setup_and_run.py
────────────────
Run this ONCE to:
  1. Delete stale artefact files (cleaned_data.csv, encoded_data.csv,
     encoder.pkl, kmeans_model.pkl, __pycache__, venv)
  2. Re-run preprocessing from scratch on swiggy.csv
  3. Print instructions to launch the Streamlit app

Usage:
    python setup_and_run.py
"""

import os
import sys
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Files / folders to delete ─────────────────────────────────────────────────
STALE_FILES = [
    "cleaned_data.csv",
    "encoded_data.csv",
    "encoder.pkl",
    "kmeans_model.pkl",
]

STALE_DIRS = [
    "__pycache__",
    "venv",
]


def delete_stale():
    print("=" * 60)
    print("  Step 1: Cleaning stale artefacts")
    print("=" * 60)

    for fname in STALE_FILES:
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"  [DELETED] {fname}")
        else:
            print(f"  [SKIP]    {fname}  (not found)")

    for dname in STALE_DIRS:
        dpath = os.path.join(BASE_DIR, dname)
        if os.path.exists(dpath):
            shutil.rmtree(dpath)
            print(f"  [DELETED] {dname}/")
        else:
            print(f"  [SKIP]    {dname}/  (not found)")

    print()


def run_preprocessing():
    print("=" * 60)
    print("  Step 2: Running preprocessing on swiggy.csv")
    print("=" * 60)
    import data_preprocessing
    data_preprocessing.main()
    print()


def print_launch_instructions():
    print("=" * 60)
    print("  Step 3: Launch the Streamlit app")
    print("=" * 60)
    print()
    print("  Run the following command in your terminal:")
    print()
    print("      streamlit run app.py")
    print()
    print("  The app will open in your browser automatically.")
    print("=" * 60)


if __name__ == "__main__":
    delete_stale()
    run_preprocessing()
    print_launch_instructions()
