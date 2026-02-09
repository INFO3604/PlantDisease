"""
test_resize_standardize.py

Tests resizing + standardization and saves a processed output image
into data/test_images.
Works with your structure: src/plantdisease/preprocess/resize_standardize.py
"""

import os
import sys
import cv2
import numpy as np

# -------------------------------
# Ensure Python can import "plantdisease" from src/
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from plantdisease.data.preprocess.resize_standardize import preprocess_image


# -------------------------------
# Paths (image goes in data/test_images)
# -------------------------------
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test_images")
INPUT_IMAGE_PATH = os.path.join(TEST_DIR, "leaf_test.jpg")
OUTPUT_IMAGE_PATH = os.path.join(TEST_DIR, "leaf_test_processed.jpg")

TARGET_SIZE = (224, 224)


def main():
    print("Starting preprocessing test...\n")
    print("Project root:", PROJECT_ROOT)
    print("Using test directory:", TEST_DIR)

    if not os.path.exists(INPUT_IMAGE_PATH):
        raise FileNotFoundError(
            f"Test image not found: {INPUT_IMAGE_PATH}\n"
            f"Make sure your image is named leaf_test.jpg and is inside data/test_images."
        )

    # Run preprocessing (resize + [0,1] normalization)
    processed = preprocess_image(
        image_path=INPUT_IMAGE_PATH,
        target_size=TARGET_SIZE,
        use_z_score=False
    )

    # Validation prints
    print("\n Preprocessing completed.")
    print("Processed shape:", processed.shape)  # expected (224, 224, 3)
    print("Min pixel:", float(processed.min())) # expected ~0.0
    print("Max pixel:", float(processed.max())) # expected ~1.0

    # Convert back to uint8 for saving as a normal image file
    out_img = (processed * 255).clip(0, 255).astype(np.uint8)

    # preprocess_image outputs RGB; OpenCV saves as BGR
    out_img_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    # Save output right in data/test_images
    ok = cv2.imwrite(OUTPUT_IMAGE_PATH, out_img_bgr)
    if not ok:
        raise RuntimeError(f"Failed to save output image to: {OUTPUT_IMAGE_PATH}")

    print(f"\n Saved processed output to:\n{OUTPUT_IMAGE_PATH}")
    print("\nTest completed successfully.")


if __name__ == "__main__":
    main()
