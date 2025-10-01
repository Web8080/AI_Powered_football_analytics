#!/usr/bin/env python3
"""
Advanced Data Preprocessing Pipeline for SoccerNet
==================================================

Senior Data Scientist Approach:
- Advanced data augmentation strategies
- Weather and lighting condition simulation
- Motion blur for fast-paced action
- Geometric transformations for different camera angles
- Color space augmentations
- Football-specific augmentations

Author: Victor
Date: 2025
"""

import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Any, Optional
import random
import logging

logger = logging.getLogger(__name__)

class AdvancedFootballAugmentation:
    """
    Advanced augmentation pipeline specifically designed for football videos
    """
    
    def __init__(self, image_size: int = 640, training: bool = True):
        self.image_size = image_size
        self.training = training
        
        # Weather conditions augmentation
        self.weather_augmentations = A.Compose([
            # Rain simulation
            A.RandomRain(
                slant_lower=-10, slant_upper=10, 
                drop_length=20, drop_width=1, 
                drop_color=(200, 200, 200), 
                blur_value=1, brightness_coefficient=0.7, 
                rain_type="drizzle", p=0.3
            ),
            
            # Snow simulation
            A.RandomSnow(
                snow_point_lower=0.1, snow_point_upper=0.3, 
                brightness_coeff=2.5, snow_point_value=0.2, p=0.2
            ),
            
            # Shadow effects
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1), 
                num_shadows_lower=1, num_shadows_upper=2, 
                shadow_dimension=5, p=0.3
            ),
            
            # Sun flare effects
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5), 
                angle_lower=0, angle_upper=1, 
                num_flare_circles_lower=6, num_flare_circles_upper=10, 
                src_radius=400, src_color=(255, 255, 255), p=0.2
            ),
            
            # Fog simulation
            A.RandomFog(
                fog_coef_lower=0.1, fog_coef_upper=0.3, 
                alpha_coef=0.1, p=0.2
            ),
        ])
        
        # Motion and blur augmentations for fast-paced action
        self.motion_augmentations = A.Compose([
            # Motion blur for fast player movement
            A.MotionBlur(blur_limit=7, p=0.4),
            
            # Gaussian blur for depth of field
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            
            # Noise for low-light conditions
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            
            # Compression artifacts
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
        ])
        
        # Geometric transformations for different camera angles
        self.geometric_augmentations = A.Compose([
            # Perspective transformation for different camera positions
            A.Perspective(scale=(0.05, 0.1), p=0.4),
            
            # Rotation for tilted cameras
            A.Rotate(limit=15, p=0.3),
            
            # Shift, scale, rotate combination
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.4
            ),
            
            # Elastic transformation for field deformation
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            
            # Grid distortion for wide-angle lens effects
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            
            # Optical distortion
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.05, p=0.2),
        ])
        
        # Color space augmentations for different lighting conditions
        self.color_augmentations = A.Compose([
            # Brightness and contrast for different times of day
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.6
            ),
            
            # Hue, saturation, value for different lighting
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
            ),
            
            # Gamma correction for different exposure
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # CLAHE for contrast enhancement
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            
            # Tone curve adjustment
            A.RandomToneCurve(scale=0.1, p=0.2),
            
            # Color jitter
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            
            # RGB shift for different camera sensors
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2),
        ])
        
        # Football-specific augmentations
        self.football_specific = A.Compose([
            # Grid shuffle to simulate different field patterns
            A.RandomGridShuffle(grid=(2, 2), p=0.1),
            
            # Channel shuffle for different color spaces
            A.ChannelShuffle(p=0.1),
            
            # Cutout for occlusion simulation
            A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32, 
                min_holes=1, min_height=8, min_width=8, 
                fill_value=0, p=0.2
            ),
            
            # Random erasing
            A.RandomErasing(
                scale=(0.02, 0.33), ratio=(0.3, 3.3), 
                value=0, p=0.2
            ),
        ])
        
        # Advanced augmentation combinations
        self.advanced_augmentations = A.Compose([
            # Mixup-style augmentation
            A.MixUp(alpha=0.2, p=0.1),
            
            # CutMix-style augmentation
            A.CutMix(alpha=1.0, p=0.1),
            
            # Mosaic augmentation
            A.Mosaic(img_scale=(640, 640), pad_val=0, p=0.1),
        ])
        
        # Create training pipeline
        if self.training:
            self.training_pipeline = A.Compose([
                # Resize to target size
                A.LongestMaxSize(max_size=image_size, p=1.0),
                A.PadIfNeeded(
                    min_height=image_size, min_width=image_size, 
                    border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0
                ),
                
                # Apply augmentations with probability
                A.OneOf([
                    self.weather_augmentations,
                    self.motion_augmentations,
                    self.geometric_augmentations,
                    self.color_augmentations,
                    self.football_specific,
                ], p=0.8),
                
                # Advanced augmentations
                A.OneOf([
                    self.advanced_augmentations,
                ], p=0.2),
                
                # Normalize
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            # Validation pipeline (no augmentation)
            self.validation_pipeline = A.Compose([
                A.LongestMaxSize(max_size=image_size, p=1.0),
                A.PadIfNeeded(
                    min_height=image_size, min_width=image_size, 
                    border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __call__(self, image: np.ndarray, bboxes: List[List[float]] = None, 
                 labels: List[int] = None) -> Dict[str, Any]:
        """
        Apply augmentations to image and annotations
        """
        if self.training:
            pipeline = self.training_pipeline
        else:
            pipeline = self.validation_pipeline
        
        # Prepare inputs
        inputs = {'image': image}
        if bboxes is not None:
            inputs['bboxes'] = bboxes
        if labels is not None:
            inputs['class_labels'] = labels
        
        # Apply augmentations
        transformed = pipeline(**inputs)
        
        return transformed

class FootballSpecificAugmentation:
    """
    Football-specific augmentation techniques
    """
    
    def __init__(self):
        self.field_colors = [
            (34, 139, 34),   # Forest green
            (0, 100, 0),     # Dark green
            (50, 205, 50),   # Lime green
            (144, 238, 144), # Light green
        ]
        
        self.jersey_colors = [
            (255, 0, 0),     # Red
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (0, 255, 0),     # Green
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
            (255, 255, 255), # White
            (0, 0, 0),       # Black
        ]
    
    def simulate_weather_conditions(self, image: np.ndarray, weather: str = 'random') -> np.ndarray:
        """
        Simulate different weather conditions
        """
        if weather == 'random':
            weather = random.choice(['sunny', 'cloudy', 'rainy', 'foggy', 'snowy'])
        
        if weather == 'rainy':
            return self._add_rain_effect(image)
        elif weather == 'snowy':
            return self._add_snow_effect(image)
        elif weather == 'foggy':
            return self._add_fog_effect(image)
        elif weather == 'cloudy':
            return self._add_cloudy_effect(image)
        else:  # sunny
            return self._add_sunny_effect(image)
    
    def _add_rain_effect(self, image: np.ndarray) -> np.ndarray:
        """Add rain effect to image"""
        # Create rain drops
        rain_drops = np.zeros_like(image)
        height, width = image.shape[:2]
        
        for _ in range(random.randint(100, 500)):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            length = random.randint(5, 20)
            
            # Draw rain drop
            for i in range(length):
                if y + i < height:
                    rain_drops[y + i, x] = [200, 200, 200]
        
        # Blend with original image
        result = cv2.addWeighted(image, 0.8, rain_drops, 0.2, 0)
        
        # Add slight blur
        result = cv2.GaussianBlur(result, (3, 3), 0)
        
        return result
    
    def _add_snow_effect(self, image: np.ndarray) -> np.ndarray:
        """Add snow effect to image"""
        snow_flakes = np.zeros_like(image)
        height, width = image.shape[:2]
        
        for _ in range(random.randint(50, 200)):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(2, 8)
            
            # Draw snow flake
            cv2.circle(snow_flakes, (x, y), size, (255, 255, 255), -1)
        
        # Blend with original image
        result = cv2.addWeighted(image, 0.9, snow_flakes, 0.1, 0)
        
        return result
    
    def _add_fog_effect(self, image: np.ndarray) -> np.ndarray:
        """Add fog effect to image"""
        # Create fog overlay
        fog = np.ones_like(image) * 200
        
        # Add gradient effect
        height, width = image.shape[:2]
        for y in range(height):
            alpha = 1 - (y / height) * 0.5
            fog[y] = fog[y] * alpha
        
        # Blend with original image
        result = cv2.addWeighted(image, 0.7, fog.astype(np.uint8), 0.3, 0)
        
        return result
    
    def _add_cloudy_effect(self, image: np.ndarray) -> np.ndarray:
        """Add cloudy effect to image"""
        # Reduce brightness and contrast
        result = cv2.convertScaleAbs(image, alpha=0.8, beta=-20)
        
        return result
    
    def _add_sunny_effect(self, image: np.ndarray) -> np.ndarray:
        """Add sunny effect to image"""
        # Increase brightness and contrast
        result = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
        
        return result
    
    def simulate_lighting_conditions(self, image: np.ndarray, lighting: str = 'random') -> np.ndarray:
        """
        Simulate different lighting conditions
        """
        if lighting == 'random':
            lighting = random.choice(['daylight', 'evening', 'night', 'stadium_lights', 'floodlights'])
        
        if lighting == 'daylight':
            return self._daylight_effect(image)
        elif lighting == 'evening':
            return self._evening_effect(image)
        elif lighting == 'night':
            return self._night_effect(image)
        elif lighting == 'stadium_lights':
            return self._stadium_lights_effect(image)
        else:  # floodlights
            return self._floodlights_effect(image)
    
    def _daylight_effect(self, image: np.ndarray) -> np.ndarray:
        """Simulate daylight conditions"""
        # Slight increase in brightness and saturation
        result = cv2.convertScaleAbs(image, alpha=1.1, beta=10)
        
        # Convert to HSV and increase saturation
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.1)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return result
    
    def _evening_effect(self, image: np.ndarray) -> np.ndarray:
        """Simulate evening conditions"""
        # Reduce brightness and add warm tone
        result = cv2.convertScaleAbs(image, alpha=0.8, beta=-30)
        
        # Add warm color cast
        result[:, :, 2] = cv2.multiply(result[:, :, 2], 1.1)  # Increase red channel
        
        return result
    
    def _night_effect(self, image: np.ndarray) -> np.ndarray:
        """Simulate night conditions"""
        # Significantly reduce brightness
        result = cv2.convertScaleAbs(image, alpha=0.4, beta=-50)
        
        # Add noise for low-light conditions
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        result = cv2.add(result, noise)
        
        return result
    
    def _stadium_lights_effect(self, image: np.ndarray) -> np.ndarray:
        """Simulate stadium lighting"""
        # Moderate brightness with cool tone
        result = cv2.convertScaleAbs(image, alpha=0.9, beta=-10)
        
        # Add cool color cast
        result[:, :, 0] = cv2.multiply(result[:, :, 0], 1.1)  # Increase blue channel
        
        return result
    
    def _floodlights_effect(self, image: np.ndarray) -> np.ndarray:
        """Simulate floodlight conditions"""
        # High contrast with bright spots
        result = cv2.convertScaleAbs(image, alpha=1.3, beta=20)
        
        # Add vignette effect
        height, width = image.shape[:2]
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2
        mask = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = mask / mask.max()
        mask = 1 - mask * 0.3
        
        for i in range(3):
            result[:, :, i] = result[:, :, i] * mask
        
        return result

class AdvancedDataPreprocessor:
    """
    Advanced data preprocessing pipeline
    """
    
    def __init__(self, image_size: int = 640, training: bool = True):
        self.image_size = image_size
        self.training = training
        
        # Initialize augmentation components
        self.albumentations = AdvancedFootballAugmentation(image_size, training)
        self.football_aug = FootballSpecificAugmentation()
        
        # Additional preprocessing
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def preprocess_image(self, image: np.ndarray, bboxes: List[List[float]] = None, 
                        labels: List[int] = None) -> Dict[str, Any]:
        """
        Preprocess image with advanced augmentations
        """
        # Apply football-specific augmentations
        if self.training:
            # Random weather simulation
            weather = random.choice(['sunny', 'cloudy', 'rainy', 'foggy', 'snowy'])
            image = self.football_aug.simulate_weather_conditions(image, weather)
            
            # Random lighting simulation
            lighting = random.choice(['daylight', 'evening', 'night', 'stadium_lights', 'floodlights'])
            image = self.football_aug.simulate_lighting_conditions(image, lighting)
        
        # Apply albumentations
        transformed = self.albumentations(image, bboxes, labels)
        
        return transformed
    
    def create_training_batch(self, images: List[np.ndarray], 
                            bboxes_list: List[List[List[float]]] = None,
                            labels_list: List[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Create training batch with advanced preprocessing
        """
        batch_images = []
        batch_bboxes = []
        batch_labels = []
        
        for i, image in enumerate(images):
            bboxes = bboxes_list[i] if bboxes_list else None
            labels = labels_list[i] if labels_list else None
            
            # Preprocess image
            transformed = self.preprocess_image(image, bboxes, labels)
            
            batch_images.append(transformed['image'])
            if 'bboxes' in transformed:
                batch_bboxes.append(transformed['bboxes'])
            if 'class_labels' in transformed:
                batch_labels.append(transformed['class_labels'])
        
        # Stack tensors
        result = {'images': torch.stack(batch_images)}
        
        if batch_bboxes:
            result['bboxes'] = batch_bboxes
        if batch_labels:
            result['labels'] = batch_labels
        
        return result

def test_augmentations():
    """
    Test the augmentation pipeline
    """
    # Create a sample image
    sample_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Initialize preprocessor
    preprocessor = AdvancedDataPreprocessor(training=True)
    
    # Test different augmentations
    weather_conditions = ['sunny', 'cloudy', 'rainy', 'foggy', 'snowy']
    lighting_conditions = ['daylight', 'evening', 'night', 'stadium_lights', 'floodlights']
    
    print("Testing weather conditions...")
    for weather in weather_conditions:
        result = preprocessor.football_aug.simulate_weather_conditions(sample_image, weather)
        print(f"Weather: {weather} - Shape: {result.shape}")
    
    print("\nTesting lighting conditions...")
    for lighting in lighting_conditions:
        result = preprocessor.football_aug.simulate_lighting_conditions(sample_image, lighting)
        print(f"Lighting: {lighting} - Shape: {result.shape}")
    
    print("\nTesting full preprocessing pipeline...")
    result = preprocessor.preprocess_image(sample_image)
    print(f"Final result - Shape: {result['image'].shape}, Type: {type(result['image'])}")

if __name__ == "__main__":
    test_augmentations()

