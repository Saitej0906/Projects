"""
main.py
-------
Runs the full pipeline end-to-end, in order. Run this from the project's
root folder with:

    python src/main.py

Each stage saves its own outputs to data/ or outputs/, so you can also run
any individual script on its own (e.g. `python src/eda.py`) once the
earlier stages have produced their files.
"""

import subprocess
import sys

STAGES = [
    "src/data_loading.py",
    "src/eda.py",
    "src/feature_engineering.py",
    "src/regression_model.py",
    "src/classification_model.py",
    "src/feature_importance.py",
]

if __name__ == "__main__":
    for stage in STAGES:
        print(f"\n{'=' * 60}\nRUNNING: {stage}\n{'=' * 60}")
        result = subprocess.run([sys.executable, stage])
        if result.returncode != 0:
            print(f"Stage {stage} failed - stopping pipeline.")
            sys.exit(1)
    print("\nPipeline complete. See outputs/figures/ and outputs/metrics/ for results.")
