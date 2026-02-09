"""Data augmentation for training."""
import logging
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageAugmenter:
    """Image augmentation utilities."""
    
    def __init__(self, horizontal_flip=True, vertical_flip=False, 
                 rotation_range=15, zoom_range=0.1):
        """
        Initialize augmenter.
        
        Args:
            horizontal_flip: Whether to use horizontal flipping
            vertical_flip: Whether to use vertical flipping
            rotation_range: Max rotation angle in degrees
            zoom_range: Max zoom factor (+/- from 1.0)
        """
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
    
    def flip_horizontal(self, image):
        """Flip image horizontally."""
        return cv2.flip(image, 1)
    
    def flip_vertical(self, image):
        """Flip image vertically."""
        return cv2.flip(image, 0)
    
    def rotate(self, image, angle):
        """
        Rotate image by angle (degrees).
        
        Args:
            image: Input image
            angle: Rotation angle in degrees (-angle_range to +angle_range)
        
        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def zoom(self, image, zoom_factor):
        """
        Zoom image (crop and resize).
        
        Args:
            image: Input image
            zoom_factor: Zoom factor (> 1.0 for zoom in, < 1.0 for zoom out)
        
        Returns:
            Zoomed image
        """
        h, w = image.shape[:2]
        crop_h = int(h / zoom_factor)
        crop_w = int(w / zoom_factor)
        
        y = (h - crop_h) // 2
        x = (w - crop_w) // 2
        
        cropped = image[y:y+crop_h, x:x+crop_w]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        return zoomed
    
    def augment(self, image):
        """
        Apply random augmentations to image.
        
        Args:
            image: Input image
        
        Returns:
            Augmented image
        """
        augmented = image.copy()
        
        if self.horizontal_flip and np.random.rand() > 0.5:
            augmented = self.flip_horizontal(augmented)
        
        if self.vertical_flip and np.random.rand() > 0.5:
            augmented = self.flip_vertical(augmented)
        
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            augmented = self.rotate(augmented, angle)
        
        if self.zoom_range > 0:
            zoom_factor = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            augmented = self.zoom(augmented, zoom_factor)
        
        return augmented
