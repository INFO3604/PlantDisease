"""Run the full PlantDisease classical CV pipeline end-to-end.

Pipeline steps:
1. Load images from class-folder input directory.
2. Run preprocessing (background removal, shadow removal, segmentation).
3. Save preprocessing outputs to demo-style folders.
4. Extract features (Gabor, Colour, Morphology) and save CSV.
5. Train/evaluate Logistic Regression, SVM, and Random Forest.
6. Save model comparison and per-class comparison tables.

Usage:
    python run_complete_pipeline.py
    python run_complete_pipeline.py --input-dir data/demo_input --images-per-class 30
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from plantdisease.data.preprocess.pipeline import PreprocessingPipeline
from plantdisease.features.extract_features import extract_features_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "demo_input"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "demo_output"
DEFAULT_FEATURES_CSV = PROJECT_ROOT / "exports" / "features.csv"
DEFAULT_COMPARISON_CSV = PROJECT_ROOT / "exports" / "classifier_comparison.csv"
DEFAULT_PER_CLASS_CSV = PROJECT_ROOT / "exports" / "per_class_comparison.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_images_with_labels(
    input_dir: Path,
    images_per_class: int | None = None,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[str], List[str], List[str]]:
    """Load class-labeled images from nested class folders.

    Returns images, class labels, source image paths, and unique image IDs.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    random.seed(seed)
    images: List[np.ndarray] = []
    labels: List[str] = []
    image_paths: List[str] = []
    image_ids: List[str] = []

    class_root = input_dir / "image" if (input_dir / "image").is_dir() else input_dir
    class_dirs = sorted([d for d in class_root.iterdir() if d.is_dir()])
    logger.info("Found %d class folders in %s", len(class_dirs), class_root)

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])

        if images_per_class is not None and len(files) > images_per_class:
            files = random.sample(files, images_per_class)

        logger.info("Loading %d images from class '%s'", len(files), class_name)

        for file_path in files:
            img = cv2.imread(str(file_path))
            if img is None:
                logger.warning("Skipping unreadable image: %s", file_path)
                continue

            images.append(img)
            labels.append(class_name)
            image_paths.append(str(file_path))
            image_ids.append(f"{class_name}__{file_path.stem}")

    logger.info("Total loaded images: %d", len(images))
    return images, labels, image_paths, image_ids


def preprocess_images(
    images: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Run preprocessing pipeline for each image and return processed outputs/masks."""
    pipeline = PreprocessingPipeline(target_size=(300, 300), normalize=False)

    processed_images: List[np.ndarray] = []
    leaf_masks: List[np.ndarray] = []
    disease_masks: List[np.ndarray] = []
    yellow_masks: List[np.ndarray] = []
    brown_masks: List[np.ndarray] = []

    logger.info("Preprocessing %d images", len(images))
    for i, image in enumerate(images, start=1):
        if i % 50 == 0 or i == len(images):
            logger.info("Preprocessed %d/%d", i, len(images))

        try:
            result = pipeline.run(image)
            processed_images.append(result.shadow_removed)
            leaf_masks.append(result.leaf_mask)
            disease_masks.append(result.disease_mask)
            yellow_masks.append(result.yellow_mask)
            brown_masks.append(result.brown_mask)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Preprocessing failed on image %d: %s", i - 1, exc)
            processed_images.append(None)
            leaf_masks.append(None)
            disease_masks.append(None)
            yellow_masks.append(None)
            brown_masks.append(None)

    return processed_images, leaf_masks, disease_masks, yellow_masks, brown_masks


def save_preprocessing_outputs(
    image_paths: List[str],
    processed_images: List[np.ndarray],
    leaf_masks: List[np.ndarray],
    disease_masks: List[np.ndarray],
    yellow_masks: List[np.ndarray],
    brown_masks: List[np.ndarray],
    output_dir: Path,
) -> None:
    """Save processed images and masks under output_dir grouped by class."""
    output_map = {
        "image": processed_images,
        "leaf_mask": leaf_masks,
        "disease_mask": disease_masks,
        "yellow_mask": yellow_masks,
        "brown_mask": brown_masks,
    }

    for kind in output_map:
        (output_dir / kind).mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(image_paths):
        src_path = Path(src)
        class_name = src_path.parent.name
        stem = src_path.stem

        for kind, values in output_map.items():
            arr = values[i]
            if arr is None:
                continue

            class_dir = output_dir / kind / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            out_path = class_dir / f"{stem}.png"

            to_write = arr
            if to_write.dtype != np.uint8:
                to_write = np.clip(to_write * 255.0, 0, 255).astype(np.uint8)

            cv2.imwrite(str(out_path), to_write)

    logger.info("Saved preprocessing outputs to: %s", output_dir)


def extract_feature_dataframe(
    processed_images: List[np.ndarray],
    leaf_masks: List[np.ndarray],
    disease_masks: List[np.ndarray],
    yellow_masks: List[np.ndarray],
    brown_masks: List[np.ndarray],
    image_ids: List[str],
    labels: List[str],
) -> pd.DataFrame:
    """Extract features for valid preprocessed samples and attach labels."""
    valid_indices = [i for i, img in enumerate(processed_images) if img is not None]
    if not valid_indices:
        raise RuntimeError("No valid preprocessed images found for feature extraction")

    df = extract_features_batch(
        image_list=[processed_images[i] for i in valid_indices],
        leaf_mask_list=[leaf_masks[i] for i in valid_indices],
        disease_mask_list=[disease_masks[i] for i in valid_indices],
        yellow_mask_list=[yellow_masks[i] for i in valid_indices],
        brown_mask_list=[brown_masks[i] for i in valid_indices],
        image_ids=[image_ids[i] for i in valid_indices],
    )

    # Keep image_id as a normal column for CSV readability.
    df = df.reset_index().rename(columns={"index": "image_id"})
    df.insert(0, "label", [labels[i] for i in valid_indices])

    logger.info("Extracted features: %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def run_classifiers(features_csv_path: Path) -> Dict[str, Dict]:
    """Train/evaluate LR, SVM, and RF on a CSV feature table."""
    sys.path.insert(0, str(PROJECT_ROOT / "models"))
    from logistic_regression.classify import train as train_lr
    from svm.classify import train as train_svm
    from utils import evaluate

    df = pd.read_csv(features_csv_path)
    if "label" not in df.columns:
        raise ValueError("features.csv must contain a 'label' column")

    y_labels = df["label"].values
    feature_df = df.drop(columns=[c for c in ["label", "image_id"] if c in df.columns])
    X = feature_df.select_dtypes(include=[np.number]).values.astype(np.float32)

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns available for training")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    class_names = label_encoder.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logger.info("Classifier split: %d train / %d test", X_train.shape[0], X_test.shape[0])

    models = {
        "logistic_regression": train_lr,
        "svm": train_svm,
    }
    results: Dict[str, Dict] = {}

    for model_name, train_fn in models.items():
        logger.info("Training/evaluating %s", model_name)
        model = train_fn(X_train, y_train)
        eval_result = evaluate(
            model,
            X_test,
            y_test,
            class_names,
            model_name=model_name.replace("_", " ").title(),
        )

        y_pred = model.predict(X_test)
        report_dict = classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        results[model_name] = {
            "accuracy": float(eval_result["accuracy"]),
            "macro_f1": float(eval_result["macro_f1"]),
            "per_class": {
                cls: {
                    "precision": float(report_dict[cls]["precision"]),
                    "recall": float(report_dict[cls]["recall"]),
                    "f1": float(report_dict[cls]["f1-score"]),
                    "support": int(report_dict[cls]["support"]),
                }
                for cls in class_names
            },
        }

    return results


def compare_results(
    results: Dict[str, Dict],
    comparison_csv: Path,
    per_class_csv: Path,
) -> None:
    """Save global and per-class model comparison tables."""
    comparison_rows = []
    per_class_rows = []

    for model_name, payload in results.items():
        pretty_model = model_name.replace("_", " ").title()
        comparison_rows.append(
            {
                "model": pretty_model,
                "accuracy": payload["accuracy"],
                "macro_f1": payload["macro_f1"],
            }
        )

        for class_name, metrics in payload["per_class"].items():
            per_class_rows.append(
                {
                    "model": pretty_model,
                    "class": class_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "support": metrics["support"],
                }
            )

    comparison_df = pd.DataFrame(comparison_rows).sort_values("accuracy", ascending=False)
    per_class_df = pd.DataFrame(per_class_rows).sort_values(["class", "model"])

    comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    per_class_csv.parent.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(comparison_csv, index=False)
    per_class_df.to_csv(per_class_csv, index=False)

    logger.info("Model comparison saved to: %s", comparison_csv)
    logger.info("Per-class comparison saved to: %s", per_class_csv)
    logger.info("\n%s", comparison_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete PlantDisease CV pipeline")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--per-class-csv", type=Path, default=DEFAULT_PER_CLASS_CSV)
    parser.add_argument("--images-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("PlantDisease Complete Pipeline")
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

    results = run_classifiers(args.features_csv)
    compare_results(results, args.comparison_csv, args.per_class_csv)

    logger.info("=" * 70)
    logger.info("Complete pipeline finished successfully")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()