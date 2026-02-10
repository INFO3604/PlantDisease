"""Test background removal on a single image first."""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import your function (or copy it here)
def remove_background_hsv(image_path, output_path=None, lower_green=None, upper_green=None):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return None
    
    if lower_green is None:
        lower_green = np.array([25, 40, 40])
    if upper_green is None:
        upper_green = np.array([95, 255, 255])
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(img, img, mask=mask)
    
    return result, mask

# Get the current script location
current_script = Path(__file__)  # This is test_single_image.py location
print(f"Script location: {current_script}")

# Navigate to the data folder
# preprocess/ → data/
data_folder = current_script.parent.parent  # Goes up from preprocess to data
print(f"Data folder: {data_folder}")

# Define paths
raw_images_folder = data_folder / "raw_images"
processed_images_folder = data_folder / "processed_images"

print(f"Raw images folder: {raw_images_folder}")
print(f"Processed images folder: {processed_images_folder}")

# Find a test image (using your specific path or find any image)
test_image_path = raw_images_folder / "Pepper,_bell___healthy" / "0a3f2927-4410-46a3-bfda-5f4769a5aaf8___JR_HL 8275.JPG"

# If that specific image doesn't exist, find any image
if not test_image_path.exists():
    print(f"\n❌ Specific image not found: {test_image_path}")
    
    # Find any image in raw_images
    image_found = False
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        images = list(raw_images_folder.rglob(ext))
        if images:
            test_image_path = images[0]
            image_found = True
            break
    
    if not image_found:
        print("❌ No images found in raw_images folder!")
        print(f"   Please add images to: {raw_images_folder}")
        exit()

print(f"\n✅ Testing on image: {test_image_path.name}")
print(f"   Full path: {test_image_path}")

# Create processed_images folder if it doesn't exist
processed_images_folder.mkdir(exist_ok=True)
print(f"✅ Processed folder: {processed_images_folder}")

# Process the image
result, mask = remove_background_hsv(test_image_path)

if result is not None:
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    original = cv2.imread(str(test_image_path))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask (Green Areas)")
    axes[1].axis('off')
    
    # Result
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    axes[2].imshow(result_rgb)
    axes[2].set_title("Background Removed")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # ===== SAVE TO processed_images FOLDER =====
    # Preserve folder structure if image is in subfolder
    # Get relative path from raw_images folder
    try:
        relative_path = test_image_path.relative_to(raw_images_folder)
        
        # Create same folder structure in processed_images
        output_subfolder = processed_images_folder / relative_path.parent
        output_subfolder.mkdir(parents=True, exist_ok=True)
        
        # Create output filename
        output_filename = f"processed_{test_image_path.stem}{test_image_path.suffix}"
        output_path = output_subfolder / output_filename
        
    except ValueError:
        # If image is directly in raw_images (not in subfolder)
        output_filename = f"processed_{test_image_path.name}"
        output_path = processed_images_folder / output_filename
    
    # Save the processed image
    success = cv2.imwrite(str(output_path), result)
    
    if success:
        print(f"\n✅ SUCCESS! Saved processed image to:")
        print(f"   {output_path}")
        print(f"\n📁 The processed_images folder now contains:")
        # List contents of processed_images folder
        for item in processed_images_folder.rglob("*"):
            if item.is_file():
                print(f"   - {item.relative_to(processed_images_folder)}")
    else:
        print(f"❌ Failed to save image to: {output_path}")
    
else:
    print("❌ Failed to process image - result is None")