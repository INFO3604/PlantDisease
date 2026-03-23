"""Run preprocessing + feature extraction only.

This script does not train classifiers. It:
1. Loads class-folder images.
2. Runs preprocessing for each image.
3. Saves image/mask outputs into data/demo_output.
4. Extracts features and saves a CSV.

Usage:
    python run_preprocess_and_extract_features.py
    python run_preprocess_and_extract_features.py --images-per-class 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from run_complete_pipeline import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    extract_feature_dataframe,
    load_images_with_labels,
    preprocess_images,
    save_preprocessing_outputs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DEMO_FEATURES_CSV = DEFAULT_OUTPUT_DIR / "features.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing + feature extraction only")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_DEMO_FEATURES_CSV)
    parser.add_argument("--images-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("PlantDisease Preprocess + Feature Extraction")
    logger.info("=" * 70)

    images, labels, image_paths, image_ids = load_images_with_labels(
        input_dir=args.input_dir,
        images_per_class=args.images_per_class,
        seed=args.seed,
    )
    if not images:
        raise RuntimeError("No images loaded from input directory")

    processed_images, leaf_masks, disease_masks, yellow_masks, brown_masks = preprocess_images(images)

    save_preprocessing_outputs(
        image_paths=image_paths,
        processed_images=processed_images,
        leaf_masks=leaf_masks,
        disease_masks=disease_masks,
        yellow_masks=yellow_masks,
        brown_masks=brown_masks,
        output_dir=args.output_dir,
    )

    features_df = extract_feature_dataframe(
        processed_images=processed_images,
        leaf_masks=leaf_masks,
        disease_masks=disease_masks,
        yellow_masks=yellow_masks,
        brown_masks=brown_masks,
        image_ids=image_ids,
        labels=labels,
    )

    args.features_csv.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(args.features_csv, index=False)
    logger.info("Features saved to: %s", args.features_csv)

    logger.info("=" * 70)
    logger.info("Preprocessing + feature extraction finished successfully")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
