"""
Advanced data augmentation for weather and lighting conditions.
Ensures model robustness across different environmental conditions.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Optional, Tuple, Any
import random
import logging

logger = logging.getLogger(__name__)


class WeatherAugmentation:
    """Weather-specific augmentation techniques."""
    
    @staticmethod
    def add_rain(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Add rain effect to image."""
        height, width = image.shape[:2]
        
        # Create rain streaks
        rain_image = image.copy()
        num_drops = int(intensity * 1000)
        
        for _ in range(num_drops):
            x = random.randint(0, width)
            y = random.randint(0, height)
            length = random.randint(10, 30)
            thickness = random.randint(1, 2)
            
            # Draw rain streak
            cv2.line(rain_image, (x, y), (x, y + length), (255, 255, 255), thickness)
        
        # Blend with original image
        alpha = intensity * 0.3
        return cv2.addWeighted(image, 1 - alpha, rain_image, alpha, 0)
    
    @staticmethod
    def add_snow(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Add snow effect to image."""
        height, width = image.shape[:2]
        
        # Create snow overlay
        snow_overlay = np.zeros_like(image)
        num_flakes = int(intensity * 500)
        
        for _ in range(num_flakes):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(2, 8)
            
            # Draw snowflake
            cv2.circle(snow_overlay, (x, y), size, (255, 255, 255), -1)
        
        # Add blur to snow
        snow_overlay = cv2.GaussianBlur(snow_overlay, (3, 3), 0)
        
        # Blend with original image
        alpha = intensity * 0.4
        return cv2.addWeighted(image, 1 - alpha, snow_overlay, alpha, 0)
    
    @staticmethod
    def add_fog(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Add fog effect to image."""
        height, width = image.shape[:2]
        
        # Create fog overlay
        fog_overlay = np.ones_like(image) * 200
        
        # Add noise to fog
        noise = np.random.normal(0, 20, image.shape).astype(np.uint8)
        fog_overlay = np.clip(fog_overlay + noise, 0, 255)
        
        # Add blur to fog
        fog_overlay = cv2.GaussianBlur(fog_overlay, (15, 15), 0)
        
        # Blend with original image
        alpha = intensity * 0.6
        return cv2.addWeighted(image, 1 - alpha, fog_overlay, alpha, 0)
    
    @staticmethod
    def add_shadows(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Add shadow effects to image."""
        height, width = image.shape[:2]
        
        # Create shadow mask
        shadow_mask = np.ones_like(image, dtype=np.float32)
        
        # Add multiple shadow areas
        num_shadows = random.randint(2, 5)
        for _ in range(num_shadows):
            # Random shadow position and size
            center_x = random.randint(0, width)
            center_y = random.randint(0, height)
            radius = random.randint(50, 150)
            
            # Create circular shadow
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            
            # Apply shadow with gradient
            shadow_strength = random.uniform(0.3, 0.7) * intensity
            shadow_mask[mask] *= (1 - shadow_strength)
        
        # Apply shadow
        shadowed_image = image.astype(np.float32) * shadow_mask
        return np.clip(shadowed_image, 0, 255).astype(np.uint8)


class LightingAugmentation:
    """Lighting-specific augmentation techniques."""
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        if factor == 1.0:
            return image
        
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Apply brightness adjustment
        brightened = img_float * factor
        
        # Clip values
        return np.clip(brightened, 0, 255).astype(np.uint8)
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast."""
        if factor == 1.0:
            return image
        
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Apply contrast adjustment
        contrasted = (img_float - 128) * factor + 128
        
        # Clip values
        return np.clip(contrasted, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_spotlight(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Add spotlight effect to image."""
        height, width = image.shape[:2]
        
        # Create spotlight mask
        spotlight_mask = np.zeros((height, width), dtype=np.float32)
        
        # Random spotlight position
        center_x = random.randint(width // 4, 3 * width // 4)
        center_y = random.randint(height // 4, 3 * height // 4)
        radius = random.randint(100, 200)
        
        # Create circular spotlight
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        spotlight_mask = np.exp(-(distance ** 2) / (2 * (radius / 3) ** 2))
        
        # Apply spotlight
        spotlight_strength = intensity * 0.5
        spotlighted = image.astype(np.float32)
        
        for c in range(3):
            spotlighted[:, :, c] = spotlighted[:, :, c] * (1 + spotlight_mask * spotlight_strength)
        
        return np.clip(spotlighted, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_flare(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Add lens flare effect to image."""
        height, width = image.shape[:2]
        
        # Create flare overlay
        flare_overlay = np.zeros_like(image, dtype=np.float32)
        
        # Random flare position
        flare_x = random.randint(width // 4, 3 * width // 4)
        flare_y = random.randint(height // 4, 3 * height // 4)
        
        # Create multiple flare circles
        num_flares = random.randint(3, 7)
        for i in range(num_flares):
            # Flare properties
            radius = random.randint(20, 80)
            brightness = random.uniform(0.3, 0.8) * intensity
            
            # Offset flare position
            offset_x = random.randint(-50, 50)
            offset_y = random.randint(-50, 50)
            flare_center_x = flare_x + offset_x
            flare_center_y = flare_y + offset_y
            
            # Create circular flare
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt((x - flare_center_x) ** 2 + (y - flare_center_y) ** 2)
            flare_mask = np.exp(-(distance ** 2) / (2 * (radius / 3) ** 2))
            
            # Add to flare overlay
            flare_overlay += flare_mask[:, :, np.newaxis] * brightness * 255
        
        # Blend with original image
        alpha = 0.3
        flared_image = image.astype(np.float32) + flare_overlay * alpha
        return np.clip(flared_image, 0, 255).astype(np.uint8)
    
    @staticmethod
    def simulate_time_of_day(image: np.ndarray, time_of_day: str) -> np.ndarray:
        """Simulate different times of day."""
        if time_of_day == "dawn":
            # Cool, blue tint
            return LightingAugmentation._apply_color_tint(image, [0.9, 0.95, 1.1])
        elif time_of_day == "morning":
            # Warm, golden tint
            return LightingAugmentation._apply_color_tint(image, [1.1, 1.05, 0.95])
        elif time_of_day == "noon":
            # Neutral, bright
            return LightingAugmentation.adjust_brightness(image, 1.2)
        elif time_of_day == "afternoon":
            # Warm, orange tint
            return LightingAugmentation._apply_color_tint(image, [1.05, 1.0, 0.9])
        elif time_of_day == "evening":
            # Cool, purple tint
            return LightingAugmentation._apply_color_tint(image, [0.95, 0.9, 1.05])
        elif time_of_day == "night":
            # Dark, blue tint
            darkened = LightingAugmentation.adjust_brightness(image, 0.3)
            return LightingAugmentation._apply_color_tint(darkened, [0.8, 0.85, 1.2])
        else:
            return image
    
    @staticmethod
    def _apply_color_tint(image: np.ndarray, tint_factors: List[float]) -> np.ndarray:
        """Apply color tint to image."""
        tinted = image.astype(np.float32)
        
        for c, factor in enumerate(tint_factors):
            tinted[:, :, c] *= factor
        
        return np.clip(tinted, 0, 255).astype(np.uint8)


class CameraAugmentation:
    """Camera-specific augmentation techniques."""
    
    @staticmethod
    def add_motion_blur(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Add motion blur to image."""
        if intensity == 0:
            return image
        
        # Random motion direction
        angle = random.uniform(0, 360)
        length = int(intensity * 20)
        
        # Create motion blur kernel
        kernel = np.zeros((length, length))
        kernel[int((length - 1) / 2), :] = np.ones(length)
        kernel = kernel / length
        
        # Rotate kernel
        center = (length // 2, length // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (length, length))
        
        # Apply blur
        blurred = cv2.filter2D(image, -1, kernel)
        
        # Blend with original
        alpha = intensity * 0.7
        return cv2.addWeighted(image, 1 - alpha, blurred, alpha, 0)
    
    @staticmethod
    def add_camera_shake(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Add camera shake effect to image."""
        if intensity == 0:
            return image
        
        height, width = image.shape[:2]
        
        # Random shake offset
        shake_x = random.randint(-int(intensity * 10), int(intensity * 10))
        shake_y = random.randint(-int(intensity * 10), int(intensity * 10))
        
        # Create transformation matrix
        M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
        
        # Apply transformation
        shaken = cv2.warpAffine(image, M, (width, height))
        
        # Blend with original
        alpha = intensity * 0.5
        return cv2.addWeighted(image, 1 - alpha, shaken, alpha, 0)
    
    @staticmethod
    def add_compression_artifacts(image: np.ndarray, quality: int = 50) -> np.ndarray:
        """Add JPEG compression artifacts."""
        # Encode and decode with low quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        compressed = cv2.imdecode(encoded_img, 1)
        
        return compressed


def get_weather_augmentation_pipeline(weather_conditions: List[str]) -> A.Compose:
    """Get weather-specific augmentation pipeline."""
    transforms = []
    
    for condition in weather_conditions:
        if condition == "rain":
            transforms.append(A.Lambda(image=WeatherAugmentation.add_rain, p=0.3))
        elif condition == "snow":
            transforms.append(A.Lambda(image=WeatherAugmentation.add_snow, p=0.2))
        elif condition == "fog":
            transforms.append(A.Lambda(image=WeatherAugmentation.add_fog, p=0.2))
        elif condition == "shadows":
            transforms.append(A.Lambda(image=WeatherAugmentation.add_shadows, p=0.4))
    
    return A.Compose(transforms)


def get_lighting_augmentation_pipeline(lighting_conditions: List[str]) -> A.Compose:
    """Get lighting-specific augmentation pipeline."""
    transforms = []
    
    for condition in lighting_conditions:
        if condition == "bright":
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.2, p=0.5
            ))
        elif condition == "dark":
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=-0.4, contrast_limit=-0.2, p=0.5
            ))
        elif condition == "spotlight":
            transforms.append(A.Lambda(image=LightingAugmentation.add_spotlight, p=0.3))
        elif condition == "flare":
            transforms.append(A.Lambda(image=LightingAugmentation.add_flare, p=0.2))
        elif condition in ["dawn", "morning", "noon", "afternoon", "evening", "night"]:
            transforms.append(A.Lambda(
                image=lambda img: LightingAugmentation.simulate_time_of_day(img, condition),
                p=0.4
            ))
    
    return A.Compose(transforms)


def get_camera_augmentation_pipeline(camera_conditions: List[str]) -> A.Compose:
    """Get camera-specific augmentation pipeline."""
    transforms = []
    
    for condition in camera_conditions:
        if condition == "motion_blur":
            transforms.append(A.Lambda(image=CameraAugmentation.add_motion_blur, p=0.3))
        elif condition == "camera_shake":
            transforms.append(A.Lambda(image=CameraAugmentation.add_camera_shake, p=0.2))
        elif condition == "compression":
            transforms.append(A.Lambda(image=CameraAugmentation.add_compression_artifacts, p=0.3))
    
    return A.Compose(transforms)


def get_comprehensive_augmentation_pipeline(
    task: str = 'detection',
    weather_conditions: Optional[List[str]] = None,
    lighting_conditions: Optional[List[str]] = None,
    camera_conditions: Optional[List[str]] = None,
    split: str = 'train'
) -> A.Compose:
    """Get comprehensive augmentation pipeline for all conditions."""
    
    if split == 'train':
        # Base augmentations
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.2),
            A.Rotate(limit=15, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Blur(blur_limit=3, p=0.1),
        ]
        
        # Add weather conditions
        if weather_conditions:
            weather_transforms = get_weather_augmentation_pipeline(weather_conditions)
            transforms.extend(weather_transforms.transforms)
        
        # Add lighting conditions
        if lighting_conditions:
            lighting_transforms = get_lighting_augmentation_pipeline(lighting_conditions)
            transforms.extend(lighting_transforms.transforms)
        
        # Add camera conditions
        if camera_conditions:
            camera_transforms = get_camera_augmentation_pipeline(camera_conditions)
            transforms.extend(camera_transforms.transforms)
        
        # Task-specific resizing
        if task == 'detection':
            transforms.append(A.Resize(640, 640))
        elif task == 'pose':
            transforms.append(A.Resize(256, 256))
        elif task == 'events':
            transforms.append(A.Resize(224, 224))
        
        # Convert to tensor
        transforms.append(ToTensorV2())
        
        # Bbox parameters for detection
        if task == 'detection':
            return A.Compose(
                transforms,
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
            )
        # Keypoint parameters for pose
        elif task == 'pose':
            return A.Compose(
                transforms,
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
            )
        else:
            return A.Compose(transforms)
    
    else:  # validation/test
        # Minimal augmentations for validation
        transforms = []
        
        if task == 'detection':
            transforms.append(A.Resize(640, 640))
            transforms.append(ToTensorV2())
            return A.Compose(
                transforms,
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
            )
        elif task == 'pose':
            transforms.append(A.Resize(256, 256))
            transforms.append(ToTensorV2())
            return A.Compose(
                transforms,
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
            )
        elif task == 'events':
            transforms.append(A.Resize(224, 224))
            transforms.append(ToTensorV2())
            return A.Compose(transforms)
    
    return A.Compose([ToTensorV2()])


def create_robust_training_pipeline() -> Dict[str, A.Compose]:
    """Create robust training pipeline for all weather and lighting conditions."""
    
    # Define all possible conditions
    weather_conditions = ["rain", "snow", "fog", "shadows"]
    lighting_conditions = ["bright", "dark", "spotlight", "flare", "dawn", "morning", "noon", "afternoon", "evening", "night"]
    camera_conditions = ["motion_blur", "camera_shake", "compression"]
    
    pipelines = {}
    
    # Detection pipeline with all conditions
    pipelines['detection_robust'] = get_comprehensive_augmentation_pipeline(
        task='detection',
        weather_conditions=weather_conditions,
        lighting_conditions=lighting_conditions,
        camera_conditions=camera_conditions,
        split='train'
    )
    
    # Pose pipeline with all conditions
    pipelines['pose_robust'] = get_comprehensive_augmentation_pipeline(
        task='pose',
        weather_conditions=weather_conditions,
        lighting_conditions=lighting_conditions,
        camera_conditions=camera_conditions,
        split='train'
    )
    
    # Events pipeline with all conditions
    pipelines['events_robust'] = get_comprehensive_augmentation_pipeline(
        task='events',
        weather_conditions=weather_conditions,
        lighting_conditions=lighting_conditions,
        camera_conditions=camera_conditions,
        split='train'
    )
    
    # Validation pipelines (minimal augmentation)
    pipelines['detection_val'] = get_comprehensive_augmentation_pipeline(
        task='detection', split='val'
    )
    pipelines['pose_val'] = get_comprehensive_augmentation_pipeline(
        task='pose', split='val'
    )
    pipelines['events_val'] = get_comprehensive_augmentation_pipeline(
        task='events', split='val'
    )
    
    return pipelines


def test_augmentation_pipeline(image_path: str, output_dir: str = "augmentation_test"):
    """Test augmentation pipeline with sample image."""
    import os
    from pathlib import Path
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load sample image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Test different conditions
    test_conditions = {
        'original': image,
        'rain': WeatherAugmentation.add_rain(image, 0.7),
        'snow': WeatherAugmentation.add_snow(image, 0.6),
        'fog': WeatherAugmentation.add_fog(image, 0.5),
        'shadows': WeatherAugmentation.add_shadows(image, 0.6),
        'bright': LightingAugmentation.adjust_brightness(image, 1.5),
        'dark': LightingAugmentation.adjust_brightness(image, 0.4),
        'spotlight': LightingAugmentation.add_spotlight(image, 0.7),
        'flare': LightingAugmentation.add_flare(image, 0.6),
        'motion_blur': CameraAugmentation.add_motion_blur(image, 0.8),
        'camera_shake': CameraAugmentation.add_camera_shake(image, 0.7),
        'compression': CameraAugmentation.add_compression_artifacts(image, 30)
    }
    
    # Save test images
    for condition, test_image in test_conditions.items():
        output_path = os.path.join(output_dir, f"{condition}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
    
    logger.info(f"Augmentation test images saved to {output_dir}")
