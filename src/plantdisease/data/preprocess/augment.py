"""
Data augmentation module for plant disease classification training.

This module provides augmentation transforms that:
- Preserve disease semantics (lesion characteristics)
- Handle geometric and photometric variations
- Support both OpenCV-based (for preprocessing) and PyTorch-based (for training) workflows

IMPORTANT: Augmentations are applied at TRAINING TIME only, not saved to disk.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    # Geometric transforms
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_range: float = 20.0  # degrees
    scale_range: Tuple[float, float] = (0.9, 1.1)
    
    # Color/intensity transforms
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    
    # Random crop
    random_crop_enabled: bool = True
    random_crop_scale: Tuple[float, float] = (0.8, 1.0)
    
    # Noise
    gaussian_noise_std: float = 0.0  # 0 = disabled
    
    def to_dict(self) -> Dict:
        return {
            'horizontal_flip': self.horizontal_flip,
            'vertical_flip': self.vertical_flip,
            'rotation_range': self.rotation_range,
            'scale_range': list(self.scale_range),
            'brightness_range': list(self.brightness_range),
            'contrast_range': list(self.contrast_range),
            'saturation_range': list(self.saturation_range),
            'random_crop_enabled': self.random_crop_enabled,
            'random_crop_scale': list(self.random_crop_scale),
            'gaussian_noise_std': self.gaussian_noise_std
        }


class ImageAugmenter:
    """
    Image augmentation for plant disease detection training.
    
    Implements augmentations that preserve disease characteristics
    while introducing realistic variations.
    """
    
    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize augmenter.
        
        Args:
            config: Augmentation configuration
            random_seed: Random seed for reproducibility
        """
        self.config = config or AugmentationConfig()
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        """Flip image horizontally."""
        return cv2.flip(image, 1)
    
    def flip_vertical(self, image: np.ndarray) -> np.ndarray:
        """Flip image vertically."""
        return cv2.flip(image, 0)
    
    def rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by angle (degrees).
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
        
        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, rotation_matrix, (w, h),
            borderMode=cv2.BORDER_REFLECT
        )
        return rotated
    
    def scale(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Scale/zoom image.
        
        Args:
            image: Input image
            scale_factor: Scale factor (>1 zoom in, <1 zoom out)
        
        Returns:
            Scaled image (same size as input)
        """
        h, w = image.shape[:2]
        
        if scale_factor > 1.0:
            # Zoom in: crop center then resize up
            crop_h = int(h / scale_factor)
            crop_w = int(w / scale_factor)
            y = (h - crop_h) // 2
            x = (w - crop_w) // 2
            cropped = image[y:y+crop_h, x:x+crop_w]
            scaled = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
        else:
            # Zoom out: resize down then pad to original size
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Pad to original size
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            scaled = cv2.copyMakeBorder(
                resized,
                pad_y, h - new_h - pad_y,
                pad_x, w - new_w - pad_x,
                cv2.BORDER_REFLECT
            )
        
        return scaled
    
    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image brightness.
        
        Args:
            image: Input image (BGR)
            factor: Brightness factor (1.0 = no change)
        
        Returns:
            Adjusted image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return adjusted
    
    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image contrast.
        
        Args:
            image: Input image
            factor: Contrast factor (1.0 = no change)
        
        Returns:
            Adjusted image
        """
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        adjusted = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        return adjusted
    
    def adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust color saturation.
        
        Args:
            image: Input image (BGR)
            factor: Saturation factor (1.0 = no change)
        
        Returns:
            Adjusted image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return adjusted
    
    def random_crop(
        self,
        image: np.ndarray,
        scale_range: Tuple[float, float] = (0.8, 1.0)
    ) -> np.ndarray:
        """
        Random crop with resize back to original size.
        
        Args:
            image: Input image
            scale_range: Range of crop scales
        
        Returns:
            Cropped and resized image
        """
        h, w = image.shape[:2]
        scale = np.random.uniform(*scale_range)
        
        crop_h = int(h * scale)
        crop_w = int(w * scale)
        
        y = np.random.randint(0, h - crop_h + 1)
        x = np.random.randint(0, w - crop_w + 1)
        
        cropped = image[y:y+crop_h, x:x+crop_w]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        return resized
    
    def add_gaussian_noise(self, image: np.ndarray, std: float) -> np.ndarray:
        """
        Add Gaussian noise to image.
        
        Args:
            image: Input image
            std: Standard deviation of noise
        
        Returns:
            Noisy image
        """
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy
    
    def augment(self, image: np.ndarray, return_params: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Apply random augmentations to image.
        
        Augmentations are chosen randomly based on configuration.
        
        Args:
            image: Input image (BGR format)
            return_params: If True, also return the augmentation parameters used
        
        Returns:
            Augmented image, or (augmented_image, params_dict) if return_params=True
        """
        augmented = image.copy()
        params = {}
        
        # Horizontal flip
        if self.config.horizontal_flip and np.random.rand() > 0.5:
            augmented = self.flip_horizontal(augmented)
            params['horizontal_flip'] = True
        
        # Vertical flip
        if self.config.vertical_flip and np.random.rand() > 0.5:
            augmented = self.flip_vertical(augmented)
            params['vertical_flip'] = True
        
        # Rotation
        if self.config.rotation_range > 0:
            angle = np.random.uniform(
                -self.config.rotation_range,
                self.config.rotation_range
            )
            augmented = self.rotate(augmented, angle)
            params['rotation'] = angle
        
        # Scale/zoom
        if self.config.scale_range != (1.0, 1.0):
            scale = np.random.uniform(*self.config.scale_range)
            augmented = self.scale(augmented, scale)
            params['scale'] = scale
        
        # Brightness
        if self.config.brightness_range != (1.0, 1.0):
            brightness = np.random.uniform(*self.config.brightness_range)
            augmented = self.adjust_brightness(augmented, brightness)
            params['brightness'] = brightness
        
        # Contrast
        if self.config.contrast_range != (1.0, 1.0):
            contrast = np.random.uniform(*self.config.contrast_range)
            augmented = self.adjust_contrast(augmented, contrast)
            params['contrast'] = contrast
        
        # Saturation
        if self.config.saturation_range != (1.0, 1.0):
            saturation = np.random.uniform(*self.config.saturation_range)
            augmented = self.adjust_saturation(augmented, saturation)
            params['saturation'] = saturation
        
        # Random crop
        if self.config.random_crop_enabled:
            augmented = self.random_crop(augmented, self.config.random_crop_scale)
            params['random_crop'] = True
        
        # Gaussian noise
        if self.config.gaussian_noise_std > 0:
            augmented = self.add_gaussian_noise(augmented, self.config.gaussian_noise_std)
            params['gaussian_noise_std'] = self.config.gaussian_noise_std
        
        if return_params:
            return augmented, params
        return augmented
    
    def augment_batch(
        self,
        images: List[np.ndarray],
        num_augmentations: int = 1
    ) -> List[np.ndarray]:
        """
        Augment a batch of images.
        
        Args:
            images: List of input images
            num_augmentations: Number of augmented versions per image
        
        Returns:
            List of augmented images
        """
        augmented = []
        for img in images:
            for _ in range(num_augmentations):
                augmented.append(self.augment(img))
        return augmented


def get_pytorch_transforms(config: Optional[AugmentationConfig] = None):
    """
    Get PyTorch-compatible transforms for training.
    
    Requires torchvision to be installed.
    
    Args:
        config: Augmentation configuration
    
    Returns:
        torchvision.transforms.Compose object
    """
    try:
        import torchvision.transforms as T
    except ImportError:
        raise ImportError("torchvision required for PyTorch transforms")
    
    config = config or AugmentationConfig()
    
    transforms_list = []
    
    # Geometric transforms
    if config.horizontal_flip:
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))
    
    if config.vertical_flip:
        transforms_list.append(T.RandomVerticalFlip(p=0.5))
    
    if config.rotation_range > 0:
        transforms_list.append(T.RandomRotation(config.rotation_range))
    
    # Random resized crop (combines scale + crop)
    if config.random_crop_enabled:
        transforms_list.append(
            T.RandomResizedCrop(
                size=224,  # Will be resized in dataset
                scale=config.random_crop_scale,
                ratio=(0.9, 1.1)
            )
        )
    
    # Color jitter
    transforms_list.append(
        T.ColorJitter(
            brightness=max(0, config.brightness_range[1] - 1),
            contrast=max(0, config.contrast_range[1] - 1),
            saturation=max(0, config.saturation_range[1] - 1),
            hue=0.05
        )
    )
    
    # Convert to tensor and normalize
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return T.Compose(transforms_list)


def get_validation_transforms():
    """
    Get transforms for validation/test (no augmentation).
    
    Returns:
        torchvision.transforms.Compose object
    """
    try:
        import torchvision.transforms as T
    except ImportError:
        raise ImportError("torchvision required for PyTorch transforms")
    
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
