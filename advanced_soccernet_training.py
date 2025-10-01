#!/usr/bin/env python3
"""
Advanced SoccerNet Training Pipeline
====================================

Senior Data Scientist Approach:
- Latest methodologies and feature engineering
- Advanced data augmentation strategies
- Multi-scale training and validation
- Ensemble methods and model optimization
- Comprehensive evaluation metrics

Author: Victor
Date: 2025
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b4, resnet50, densenet121

# Advanced ML Libraries
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Data Processing
import yaml
import requests
import zipfile
import tarfile
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# Custom modules
sys.path.append(str(Path(__file__).parent))
from ml.utils.class_mapping import CLASS_NAMES, CLASS_COLORS
from ml.analytics.statistics import FootballStatistics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedDataAugmentation:
    """
    Advanced data augmentation strategies for football videos
    Based on latest research in sports analytics and computer vision
    """
    
    def __init__(self, image_size: int = 640):
        self.image_size = image_size
        
        # Weather and lighting conditions augmentation
        self.weather_augmentations = A.Compose([
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, 
                        drop_color=(200, 200, 200), blur_value=1, brightness_coefficient=0.7, 
                        rain_type="drizzle", p=0.3),
            A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, 
                        snow_point_value=0.2, p=0.2),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, 
                          shadow_dimension=5, p=0.3),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, 
                           num_flare_circles_lower=6, num_flare_circles_upper=10, 
                           src_radius=400, src_color=(255, 255, 255), p=0.2),
        ])
        
        # Motion blur for fast-paced football action
        self.motion_augmentations = A.Compose([
            A.MotionBlur(blur_limit=7, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.2),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
        ])
        
        # Geometric transformations for different camera angles
        self.geometric_augmentations = A.Compose([
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.Rotate(limit=15, p=0.4),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.4),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        ])
        
        # Color space augmentations for different lighting conditions
        self.color_augmentations = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.RandomToneCurve(scale=0.1, p=0.2),
        ])
        
        # Football-specific augmentations
        self.football_specific = A.Compose([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.05, p=0.2),
            A.RandomGridShuffle(grid=(2, 2), p=0.1),
        ])
        
        # Combined augmentation pipeline
        self.combined_augmentations = A.Compose([
            A.LongestMaxSize(max_size=image_size, p=1.0),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, 
                         border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
            A.OneOf([
                self.weather_augmentations,
                self.motion_augmentations,
                self.geometric_augmentations,
                self.color_augmentations,
                self.football_specific,
            ], p=0.8),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Validation transforms (no augmentation)
        self.validation_transforms = A.Compose([
            A.LongestMaxSize(max_size=image_size, p=1.0),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, 
                         border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

class FeatureEngineering:
    """
    Advanced feature engineering for football analytics
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def extract_spatial_features(self, bbox: List[float], image_shape: Tuple[int, int]) -> Dict[str, float]:
        """Extract spatial features from bounding boxes"""
        x1, y1, x2, y2 = bbox
        img_h, img_w = image_shape[:2]
        
        # Normalize coordinates
        x_center = (x1 + x2) / 2 / img_w
        y_center = (y1 + y2) / 2 / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        
        # Calculate area and aspect ratio
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        # Field position features
        field_zone = self._get_field_zone(x_center, y_center)
        
        return {
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'field_zone': field_zone,
            'distance_from_center': np.sqrt((x_center - 0.5)**2 + (y_center - 0.5)**2),
        }
    
    def _get_field_zone(self, x: float, y: float) -> int:
        """Classify field position into zones"""
        if y < 0.33:
            return 0  # Defensive third
        elif y < 0.67:
            return 1  # Midfield
        else:
            return 2  # Attacking third
    
    def extract_temporal_features(self, track_history: List[Dict]) -> Dict[str, float]:
        """Extract temporal features from tracking history"""
        if len(track_history) < 2:
            return {'speed': 0, 'acceleration': 0, 'direction_change': 0}
        
        # Calculate speed and acceleration
        positions = [(t['x_center'], t['y_center']) for t in track_history[-10:]]  # Last 10 frames
        speeds = []
        accelerations = []
        direction_changes = []
        
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            speed = np.sqrt(dx**2 + dy**2)
            speeds.append(speed)
            
            if i > 1:
                acceleration = speeds[-1] - speeds[-2]
                accelerations.append(acceleration)
                
                # Direction change
                if len(speeds) >= 2:
                    prev_angle = np.arctan2(positions[i-1][1] - positions[i-2][1], 
                                          positions[i-1][0] - positions[i-2][0])
                    curr_angle = np.arctan2(positions[i][1] - positions[i-1][1], 
                                          positions[i][0] - positions[i-1][0])
                    angle_diff = abs(curr_angle - prev_angle)
                    direction_changes.append(min(angle_diff, 2*np.pi - angle_diff))
        
        return {
            'avg_speed': np.mean(speeds) if speeds else 0,
            'max_speed': np.max(speeds) if speeds else 0,
            'avg_acceleration': np.mean(accelerations) if accelerations else 0,
            'avg_direction_change': np.mean(direction_changes) if direction_changes else 0,
            'movement_consistency': 1 - np.std(speeds) if speeds else 0,
        }
    
    def extract_team_features(self, detections: List[Dict]) -> Dict[str, float]:
        """Extract team-level features"""
        team_a_players = [d for d in detections if d.get('class') in [0, 1]]  # team_a_player, team_a_goalkeeper
        team_b_players = [d for d in detections if d.get('class') in [2, 3]]  # team_b_player, team_b_goalkeeper
        
        features = {}
        
        # Team formation features
        if team_a_players:
            team_a_x = [d['x_center'] for d in team_a_players]
            team_a_y = [d['y_center'] for d in team_a_players]
            features.update({
                'team_a_avg_x': np.mean(team_a_x),
                'team_a_avg_y': np.mean(team_a_y),
                'team_a_spread_x': np.std(team_a_x),
                'team_a_spread_y': np.std(team_a_y),
                'team_a_players_count': len(team_a_players),
            })
        
        if team_b_players:
            team_b_x = [d['x_center'] for d in team_b_players]
            team_b_y = [d['y_center'] for d in team_b_players]
            features.update({
                'team_b_avg_x': np.mean(team_b_x),
                'team_b_avg_y': np.mean(team_b_y),
                'team_b_spread_x': np.std(team_b_x),
                'team_b_spread_y': np.std(team_b_y),
                'team_b_players_count': len(team_b_players),
            })
        
        # Ball position relative to teams
        ball_detections = [d for d in detections if d.get('class') == 5]  # ball
        if ball_detections and team_a_players and team_b_players:
            ball_x = ball_detections[0]['x_center']
            ball_y = ball_detections[0]['y_center']
            
            team_a_center_x = np.mean([d['x_center'] for d in team_a_players])
            team_b_center_x = np.mean([d['x_center'] for d in team_b_players])
            
            features.update({
                'ball_closer_to_team_a': abs(ball_x - team_a_center_x) < abs(ball_x - team_b_center_x),
                'ball_field_position': ball_y,  # 0 = defensive, 1 = attacking
            })
        
        return features

class SoccerNetDataset(Dataset):
    """
    Advanced SoccerNet dataset with comprehensive preprocessing
    """
    
    def __init__(self, data_dir: str, split: str = 'train', 
                 augmentations: Optional[AdvancedDataAugmentation] = None,
                 feature_engineering: Optional[FeatureEngineering] = None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.augmentations = augmentations
        self.feature_engineering = feature_engineering
        
        # Load dataset metadata
        self.annotations = self._load_annotations()
        self.images = self._load_images()
        
        logger.info(f"Loaded {len(self.images)} {split} images")
        logger.info(f"Class distribution: {Counter([ann['class'] for ann in self.annotations])}")
    
    def _load_annotations(self) -> List[Dict]:
        """Load and preprocess annotations"""
        annotations = []
        
        # Load SoccerNet annotations
        annotation_files = list(self.data_dir.glob(f"**/*{self.split}*.json"))
        if not annotation_files:
            # Fallback to YOLO format
            annotation_files = list(self.data_dir.glob(f"**/labels/{self.split}/*.txt"))
        
        for ann_file in annotation_files:
            if ann_file.suffix == '.json':
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                    annotations.extend(data.get('annotations', []))
            elif ann_file.suffix == '.txt':
                # YOLO format
                with open(ann_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            annotations.append({
                                'image_id': ann_file.stem,
                                'class': int(parts[0]),
                                'bbox': [float(x) for x in parts[1:5]],
                                'confidence': 1.0
                            })
        
        return annotations
    
    def _load_images(self) -> List[Dict]:
        """Load image metadata"""
        images = []
        
        # Find all images in the dataset
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in image_extensions:
            images.extend(list(self.data_dir.glob(f"**/images/{self.split}/*{ext}")))
            images.extend(list(self.data_dir.glob(f"**/{self.split}/*{ext}")))
        
        return [{'id': img.stem, 'path': str(img)} for img in images]
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_info = self.images[idx]
        image_path = image_info['path']
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        image_annotations = [ann for ann in self.annotations 
                           if ann['image_id'] == image_info['id']]
        
        # Prepare bounding boxes and labels
        bboxes = []
        labels = []
        
        for ann in image_annotations:
            bbox = ann['bbox']
            if len(bbox) == 4:  # YOLO format (normalized)
                bboxes.append(bbox)
            else:  # COCO format (absolute)
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                # Normalize
                img_h, img_w = image.shape[:2]
                bboxes.append([x1/img_w, y1/img_h, x2/img_w, y2/img_h])
            
            labels.append(ann['class'])
        
        # Apply augmentations
        if self.augmentations and self.split == 'train':
            transformed = self.augmentations.combined_augmentations(
                image=image, bboxes=bboxes, class_labels=labels
            )
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['class_labels']
        elif self.augmentations:
            transformed = self.augmentations.validation_transforms(
                image=image, bboxes=bboxes, class_labels=labels
            )
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['class_labels']
        
        # Extract features if feature engineering is enabled
        features = {}
        if self.feature_engineering:
            for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                spatial_features = self.feature_engineering.extract_spatial_features(
                    bbox, image.shape
                )
                features[f'detection_{i}'] = spatial_features
        
        return {
            'image': image,
            'bboxes': torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
            'features': features,
            'image_id': image_info['id']
        }

class AdvancedModel(nn.Module):
    """
    Advanced model architecture with attention mechanisms
    """
    
    def __init__(self, num_classes: int = len(CLASS_NAMES), backbone: str = 'efficientnet_b4'):
        super().__init__()
        self.num_classes = num_classes
        
        # Backbone selection
        if backbone == 'efficientnet_b4':
            self.backbone = efficientnet_b4(pretrained=True)
            feature_dim = 1792
        elif backbone == 'resnet50':
            self.backbone = resnet50(pretrained=True)
            feature_dim = 2048
        elif backbone == 'densenet121':
            self.backbone = densenet121(pretrained=True)
            feature_dim = 1024
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Classification heads
        self.classifier = nn.Linear(256, num_classes)
        self.bbox_regressor = nn.Linear(256, 4)
        self.confidence_head = nn.Linear(256, 1)
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Apply attention
        attended_features, _ = self.attention(
            features.unsqueeze(1), 
            features.unsqueeze(1), 
            features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        # Feature fusion
        fused_features = self.feature_fusion(attended_features)
        
        # Multiple outputs
        classification = self.classifier(fused_features)
        bbox_regression = self.bbox_regressor(fused_features)
        confidence = torch.sigmoid(self.confidence_head(fused_features))
        
        return {
            'classification': classification,
            'bbox_regression': bbox_regression,
            'confidence': confidence
        }

class AdvancedTrainer:
    """
    Advanced trainer with latest optimization techniques
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Advanced optimizer with learning rate scheduling
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.bbox_loss = nn.SmoothL1Loss()
        self.confidence_loss = nn.BCELoss()
        
        # Training metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'mAP50': [],
            'mAP50_95': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)
            bboxes = batch['bboxes'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate losses
            classification_loss = self.classification_loss(outputs['classification'], labels)
            bbox_loss = self.bbox_loss(outputs['bbox_regression'], bboxes)
            confidence_loss = self.confidence_loss(outputs['confidence'].squeeze(), 
                                                 torch.ones_like(labels, dtype=torch.float))
            
            total_batch_loss = classification_loss + 0.5 * bbox_loss + 0.3 * confidence_loss
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_batch_loss.item()
            _, predicted = torch.max(outputs['classification'], 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        
        # Update learning rate
        self.scheduler.step()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                bboxes = batch['bboxes'].to(self.device)
                
                outputs = self.model(images)
                
                # Calculate losses
                classification_loss = self.classification_loss(outputs['classification'], labels)
                bbox_loss = self.bbox_loss(outputs['bbox_regression'], bboxes)
                confidence_loss = self.confidence_loss(outputs['confidence'].squeeze(), 
                                                     torch.ones_like(labels, dtype=torch.float))
                
                total_batch_loss = classification_loss + 0.5 * bbox_loss + 0.3 * confidence_loss
                
                total_loss += total_batch_loss.item()
                _, predicted = torch.max(outputs['classification'], 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0
        }

def download_soccernet_dataset(data_dir: str = "data/soccernet") -> str:
    """
    Download and prepare SoccerNet dataset
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading SoccerNet dataset...")
    
    # SoccerNet v3 download URLs (these would be actual URLs in production)
    urls = {
        'train': 'https://www.soccer-net.org/data/downloads/train.zip',
        'val': 'https://www.soccer-net.org/data/downloads/val.zip',
        'test': 'https://www.soccer-net.org/data/downloads/test.zip'
    }
    
    # For now, create a mock dataset structure
    logger.info("Creating mock SoccerNet dataset structure...")
    
    for split in ['train', 'val', 'test']:
        split_dir = data_path / split
        split_dir.mkdir(exist_ok=True)
        
        # Create mock images and annotations
        for i in range(100 if split == 'train' else 20):
            # Create mock image
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img_path = split_dir / f"image_{i:06d}.jpg"
            cv2.imwrite(str(img_path), img)
            
            # Create mock annotation
            ann_path = split_dir / f"image_{i:06d}.txt"
            with open(ann_path, 'w') as f:
                # Random annotations
                num_objects = np.random.randint(5, 15)
                for _ in range(num_objects):
                    class_id = np.random.randint(0, len(CLASS_NAMES))
                    x_center = np.random.uniform(0.1, 0.9)
                    y_center = np.random.uniform(0.1, 0.9)
                    width = np.random.uniform(0.05, 0.2)
                    height = np.random.uniform(0.1, 0.3)
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    logger.info(f"SoccerNet dataset prepared at: {data_path}")
    return str(data_path)

def create_advanced_training_config() -> Dict[str, Any]:
    """
    Create advanced training configuration
    """
    return {
        'model': {
            'backbone': 'efficientnet_b4',
            'num_classes': len(CLASS_NAMES),
            'input_size': 640,
            'pretrained': True
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'gradient_clip': 1.0,
            'label_smoothing': 0.1,
            'mixup_alpha': 0.2,
            'cutmix_alpha': 1.0
        },
        'data': {
            'augmentation': {
                'weather_prob': 0.3,
                'motion_prob': 0.3,
                'geometric_prob': 0.4,
                'color_prob': 0.5,
                'football_specific_prob': 0.2
            },
            'validation_split': 0.2,
            'stratified_split': True
        },
        'optimization': {
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingWarmRestarts',
            'early_stopping': {
                'patience': 15,
                'min_delta': 0.001
            },
            'model_checkpointing': True,
            'best_metric': 'mAP50'
        },
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'mAP50', 'mAP50_95'],
            'confusion_matrix': True,
            'classification_report': True,
            'per_class_metrics': True
        }
    }

def main():
    """
    Main training pipeline
    """
    parser = argparse.ArgumentParser(description='Advanced SoccerNet Training Pipeline')
    parser.add_argument('--data_dir', type=str, default='data/soccernet',
                       help='Path to SoccerNet dataset')
    parser.add_argument('--output_dir', type=str, default='models/advanced_soccernet',
                       help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for training (cpu/cuda)')
    parser.add_argument('--download_data', action='store_true',
                       help='Download SoccerNet dataset')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to training configuration file')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset if requested
    if args.download_data:
        data_dir = download_soccernet_dataset(args.data_dir)
    else:
        data_dir = args.data_dir
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_advanced_training_config()
    
    logger.info("Starting Advanced SoccerNet Training Pipeline")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Initialize components
    augmentations = AdvancedDataAugmentation(config['model']['input_size'])
    feature_engineering = FeatureEngineering()
    
    # Create datasets
    train_dataset = SoccerNetDataset(
        data_dir, 'train', augmentations, feature_engineering
    )
    val_dataset = SoccerNetDataset(
        data_dir, 'val', augmentations, feature_engineering
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Initialize model
    model = AdvancedModel(
        num_classes=config['model']['num_classes'],
        backbone=config['model']['backbone']
    )
    
    # Initialize trainer
    trainer = AdvancedTrainer(model, args.device)
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        
        # Update metrics
        trainer.metrics['train_loss'].append(train_metrics['loss'])
        trainer.metrics['val_loss'].append(val_metrics['loss'])
        trainer.metrics['train_acc'].append(train_metrics['accuracy'])
        trainer.metrics['val_acc'].append(val_metrics['accuracy'])
        
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'config': config
            }, output_dir / 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['optimization']['early_stopping']['patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'metrics': trainer.metrics,
        'config': config
    }, output_dir / 'final_model.pth')
    
    # Save training metrics
    with open(output_dir / 'training_metrics.json', 'w') as f:
        json.dump(trainer.metrics, f, indent=2)
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Models saved to: {output_dir}")

if __name__ == "__main__":
    main()

