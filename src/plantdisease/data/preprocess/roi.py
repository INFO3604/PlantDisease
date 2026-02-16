import cv2
import numpy as np
from pathlib import Path

def crop_leaf(image_path, output_path, padding=10):
    """
    Detect and crop the leaf region from an image.
    More robust and accurate ROI detection.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read: {image_path}")
        return

    original = img.copy()

    # Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Reduce noise if and where necessary
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Auto invert if needed
    if cv2.countNonZero(binary) > binary.size / 2:
        binary = cv2.bitwise_not(binary)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print(f"No leaf detected in: {image_path}")
        return

    # Filter by area (remove small noise contours)
    min_area = 1000
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not contours:
        print(f"No significant leaf detected in: {image_path}")
        return

    # Largest contour assumed as leaf
    largest = max(contours, key=cv2.contourArea)

    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(largest)

    # Add padding safely
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, img.shape[1] - x)
    h = min(h + 2 * padding, img.shape[0] - y)

    cropped = original[y:y+h, x:x+w]

    cv2.imwrite(str(output_path), cropped)
    print(f"Processed: {image_path.name}")
    

def process_folder(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    input_path = Path(input_folder)

    image_files = [f for f in input_path.iterdir()
                   if f.suffix.lower() in extensions]

    if not image_files:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(image_files)} images")

    for img_file in image_files:
        output_file = Path(output_folder) / img_file.name
        crop_leaf(img_file, output_file)

    print("Done!")
    

if __name__ == "__main__":
    input_folder = "data/Tomato___Septoria_leaf_spot" #input folder 
    output_folder = "data/processed" #output folder for processed images for next stage
    process_folder(input_folder, output_folder)
