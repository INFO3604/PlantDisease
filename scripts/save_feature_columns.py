import json
from pathlib import Path

import pandas as pd

FEATURES_CSV = Path("data/processed/features.csv")
EXPORT_DIR = Path("models/exports")
OUTPUT_PATH = EXPORT_DIR / "feature_columns.json"


def main():
    df = pd.read_csv(FEATURES_CSV)

    feature_columns = [
        c for c in df.columns
        if c not in ["label", "image_id"]
    ]

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(feature_columns, f, indent=2)

    print(f"Saved {len(feature_columns)} feature columns to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()