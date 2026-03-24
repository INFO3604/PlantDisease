import sys
import random
from pathlib import Path
import cv2
import pandas as pd
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

# Optional: reduce warning spam from empty masks
logging.getLogger().setLevel(logging.ERROR)

# Import your pipeline + feature extraction
from plantdisease.data.preprocess.pipeline import PreprocessingPipeline
from plantdisease.features import extract_features_from_pipeline_result

# -------------------------------
# CONFIG
# -------------------------------
DATA_DIR = Path("data/raw/train")
SAMPLE_PER_CLASS = 500
PROGRESS_EVERY = 25  # print progress every 25 images
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

pipeline = PreprocessingPipeline()
all_features = []

# Get only class folders
class_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]

print("=" * 80)
print("FEATURE EXTRACTION STARTED")
print("=" * 80)
print(f"Found {len(class_dirs)} class folders")
print(f"Sampling {SAMPLE_PER_CLASS} images per class")
print()

# -------------------------------
# LOOP THROUGH CLASSES
# -------------------------------
for class_idx, class_dir in enumerate(class_dirs, start=1):
    class_name = class_dir.name
    image_paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))

    if len(image_paths) == 0:
        print(f"[{class_idx}/{len(class_dirs)}] {class_name}: no images found, skipping")
        continue

    # Sample images
    sampled_paths = random.sample(
        image_paths,
        min(SAMPLE_PER_CLASS, len(image_paths))
    )

    print("=" * 80)
    print(f"[{class_idx}/{len(class_dirs)}] CLASS: {class_name}")
    print(f"Images selected: {len(sampled_paths)}")
    print("=" * 80)

    # -------------------------------
    # PROCESS EACH IMAGE
    # -------------------------------
    for img_idx, img_path in enumerate(sampled_paths, start=1):
        try:
            if img_idx == 1 or img_idx % PROGRESS_EVERY == 0 or img_idx == len(sampled_paths):
                print(f"  -> [{class_name}] {img_idx}/{len(sampled_paths)} : {img_path.name}")

            image = cv2.imread(str(img_path))

            if image is None:
                print(f"  !! Failed to read image: {img_path}")
                continue

            # Run preprocessing pipeline
            result = pipeline.run(image)

            # Extract features with lighter Gabor settings
            features = extract_features_from_pipeline_result(
                result,
                gabor_frequencies=(0.1, 0.2),
                gabor_orientations=2,
                compute_leaf_colour_stats=False
            )

            # Add labels + id
            features["label"] = class_name
            features["image_id"] = img_path.stem

            all_features.append(features)

        except Exception as e:
            print(f"  !! Error processing {img_path.name}: {e}")

# -------------------------------
# SAVE TO DATAFRAME
# -------------------------------
df = pd.DataFrame(all_features)
df.set_index("image_id", inplace=True)

output_path = Path("data/processed/features.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path)

print("\n" + "=" * 80)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 80)
print(f"Saved features to: {output_path.resolve()}")
print(f"Shape: {df.shape}")