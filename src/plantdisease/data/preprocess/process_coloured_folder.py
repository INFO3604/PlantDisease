"""Main script to process plant images with folder structure preservation."""
import cv2
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_background_hsv(image_path, output_path=None, lower_green=None, upper_green=None):
    """
    Remove background using HSV color space (for green leaves).
    """
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
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
    
    return result

def process_plant_images_recursive(input_folder, output_folder):
    """
    Process all plant images in a folder and preserve subfolder structure.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG']
    
    # Find all image files (recursive search)
    image_files = []
    for ext in extensions:
        image_files.extend(list(input_path.rglob(ext)))
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}")
    print("-" * 60)
    
    # Process each image
    successful = 0
    for i, img_file in enumerate(image_files, 1):
        try:
            # Get relative path from input folder
            relative_path = img_file.relative_to(input_path)
            
            # Create corresponding output path
            output_file = output_path / relative_path.parent / f"processed_{img_file.name}"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"[{i}/{len(image_files)}] Processing: {relative_path}")
            
            # Remove background
            result = remove_background_hsv(
                image_path=img_file,
                output_path=output_file,
                lower_green=np.array([25, 40, 40]),
                upper_green=np.array([95, 255, 255])
            )
            
            if result is not None:
                successful += 1
                print(f"  ✓ Saved to: {output_file.relative_to(output_path)}")
            else:
                print(f"  ✗ Failed: {relative_path}")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("-" * 60)
    print(f"🎉 Processing Complete!")
    print(f"   Successfully processed: {successful}/{len(image_files)} images")
    print(f"   Output folder: {output_folder}")
    
    # Show what was created
    if successful > 0:
        print(f"\n📁 Processed folder structure:")
        processed_folders = set()
        for file in output_path.rglob("*.jpg"):
            processed_folders.add(file.parent.relative_to(output_path))
        
        for folder in sorted(processed_folders):
            file_count = len(list((output_path / folder).glob("*.jpg"))) + \
                         len(list((output_path / folder).glob("*.png")))
            print(f"   {folder}/ - {file_count} images")

# MAIN EXECUTION
if __name__ == "__main__":
    # === CONFIGURE THESE PATHS ===
    # For YOUR specific folder structure:
    INPUT_FOLDER = r"C:\Users\robyn\Downloads\project_articles\PlantDisease\src\plantdisease\data\raw_images\Tomato___healthy"  # Change this to your actual input folder
    OUTPUT_FOLDER = r"C:\Users\robyn\Downloads\project_articles\PlantDisease\src\plantdisease\data\processed_images\Tomato"
    
    # Verify paths exist
    input_path = Path(INPUT_FOLDER)
    if not input_path.exists():
        print(f"❌ Input folder does not exist: {INPUT_FOLDER}")
        print(f"   Please create it and add your plant images")
        exit()
    
    print("🔍 Checking input folder...")
    images = list(input_path.rglob("*.jpg")) + list(input_path.rglob("*.png"))
    print(f"   Found {len(images)} images in {INPUT_FOLDER}")
    
    if len(images) > 0:  # FIXED THIS LINE!
        print(f"   Sample files:")
        for img in images[:3]:
            print(f"     - {img.relative_to(input_path)}")
        if len(images) > 3:
            print(f"     ... and {len(images)-3} more")
    
    print("\n🚀 Starting processing...")
    process_plant_images_recursive(INPUT_FOLDER, OUTPUT_FOLDER)