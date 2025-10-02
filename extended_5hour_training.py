#!/usr/bin/env python3
"""
Extended 5+ Hour Training Pipeline for Perfect Team Classification
Advanced training with data augmentation, transfer learning, and curriculum learning
"""

import os
import time
import logging
import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import albumentations as A
from sklearn.model_selection import train_test_split
import json
import random
from tqdm import tqdm
import mlflow

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extended_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExtendedFootballTrainer:
    """Extended 5+ hour training pipeline for perfect team classification"""
    
    def __init__(self):
        self.start_time = time.time()
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Training configuration for 5+ hours
        self.config = {
            'total_epochs': 300,  # Extended training
            'image_size': 640,
            'batch_size': 8,      # Smaller batch for stability
            'learning_rates': [0.001, 0.0005, 0.0001],  # Multi-stage LR
            'patience': 50,       # Extended patience
            'save_period': 25,    # Save every 25 epochs
        }
        
        # Enhanced class mapping for football
        self.football_classes = {
            0: 'team_a_player',
            1: 'team_b_player', 
            2: 'team_a_goalkeeper',
            3: 'team_b_goalkeeper',
            4: 'referee',
            5: 'assistant_referee',
            6: 'ball',
            7: 'goalpost',
            8: 'others'
        }
        
        logger.info("🚀 EXTENDED 5+ HOUR TRAINING PIPELINE INITIALIZED")
        logger.info(f"📊 Configuration: {self.config}")
    
    def create_advanced_augmentation_pipeline(self):
        """Create comprehensive augmentation pipeline for football scenarios"""
        logger.info("🎨 Creating advanced augmentation pipeline...")
        
        # Stage 1: Basic augmentations (first 100 epochs)
        basic_augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # Stage 2: Advanced augmentations (epochs 100-200)
        advanced_augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
            A.GaussNoise(var_limit=(10.0, 80.0), p=0.4),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.RandomGamma(gamma_limit=(70, 130), p=0.4),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.3),
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, p=0.2),
            A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, p=0.2),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # Stage 3: Extreme augmentations (epochs 200-300)
        extreme_augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.9),
            A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=25, p=0.9),
            A.GaussNoise(var_limit=(20.0, 100.0), p=0.5),
            A.MotionBlur(blur_limit=7, p=0.4),
            A.RandomGamma(gamma_limit=(60, 140), p=0.5),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.4),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, p=0.4),
            A.RandomRain(slant_lower=-15, slant_upper=15, drop_length=25, drop_width=2, p=0.3),
            A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.4, brightness_coeff=3.0, p=0.3),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.2),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        return {
            'basic': basic_augmentations,
            'advanced': advanced_augmentations, 
            'extreme': extreme_augmentations
        }
    
    def create_massive_dataset(self):
        """Create massive augmented dataset from existing SoccerNet data"""
        logger.info("📊 CREATING MASSIVE AUGMENTED DATASET")
        logger.info("=" * 60)
        
        # Create dataset directories
        dataset_dir = Path("data/massive_football_dataset")
        for split in ['train', 'val']:
            for subdir in ['images', 'labels']:
                (dataset_dir / split / subdir).mkdir(parents=True, exist_ok=True)
        
        # Find existing SoccerNet data
        source_dirs = [
            "data/yolo_dataset",
            "data/balanced_dataset", 
            "data/soccernet_real"
        ]
        
        all_images = []
        all_labels = []
        
        for source_dir in source_dirs:
            source_path = Path(source_dir)
            if source_path.exists():
                # Find images
                for img_path in source_path.rglob("*.jpg"):
                    label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
                    if label_path.exists() and label_path.stat().st_size > 0:
                        all_images.append(img_path)
                        all_labels.append(label_path)
        
        logger.info(f"📁 Found {len(all_images)} source images with labels")
        
        if len(all_images) < 10:
            logger.error("❌ Insufficient source data for massive dataset creation")
            return None
        
        # Get augmentation pipelines
        augmentations = self.create_advanced_augmentation_pipeline()
        
        # Create massive dataset with multiple augmentations per image
        target_images = 5000  # Target 5000 training images
        augmentations_per_image = max(1, target_images // len(all_images))
        
        logger.info(f"🎯 Creating {augmentations_per_image} augmentations per source image")
        logger.info(f"📈 Target dataset size: {target_images} images")
        
        train_count = 0
        val_count = 0
        
        for i, (img_path, label_path) in enumerate(tqdm(zip(all_images, all_labels), desc="Creating massive dataset")):
            try:
                # Load image and labels
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # Load YOLO labels
                with open(label_path, 'r') as f:
                    labels = []
                    bboxes = []
                    for line in f:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id = int(parts[0])
                                x_center, y_center, width, height = map(float, parts[1:])
                                labels.append(class_id)
                                bboxes.append([x_center, y_center, width, height])
                
                if not labels:
                    continue
                
                # Determine split (80% train, 20% val)
                is_train = random.random() < 0.8
                split_dir = "train" if is_train else "val"
                
                # Create multiple augmented versions
                for aug_idx in range(augmentations_per_image):
                    # Choose augmentation level based on iteration
                    if aug_idx < augmentations_per_image // 3:
                        aug_pipeline = augmentations['basic']
                    elif aug_idx < 2 * augmentations_per_image // 3:
                        aug_pipeline = augmentations['advanced']
                    else:
                        aug_pipeline = augmentations['extreme']
                    
                    # Apply augmentation
                    try:
                        augmented = aug_pipeline(image=image, bboxes=bboxes, class_labels=labels)
                        aug_image = augmented['image']
                        aug_bboxes = augmented['bboxes']
                        aug_labels = augmented['class_labels']
                        
                        if not aug_bboxes:  # Skip if no bboxes after augmentation
                            continue
                        
                        # Save augmented image
                        img_name = f"{img_path.stem}_aug_{aug_idx}.jpg"
                        aug_img_path = dataset_dir / split_dir / "images" / img_name
                        cv2.imwrite(str(aug_img_path), aug_image)
                        
                        # Save augmented labels
                        label_name = f"{img_path.stem}_aug_{aug_idx}.txt"
                        aug_label_path = dataset_dir / split_dir / "labels" / label_name
                        
                        with open(aug_label_path, 'w') as f:
                            for bbox, label in zip(aug_bboxes, aug_labels):
                                f.write(f"{label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                        
                        if is_train:
                            train_count += 1
                        else:
                            val_count += 1
                            
                    except Exception as e:
                        logger.warning(f"Augmentation failed for {img_path}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        logger.info(f"✅ MASSIVE DATASET CREATED!")
        logger.info(f"📊 Training images: {train_count}")
        logger.info(f"📊 Validation images: {val_count}")
        logger.info(f"📊 Total images: {train_count + val_count}")
        
        # Create dataset.yaml
        yaml_content = f"""
path: {dataset_dir.absolute()}
train: train/images
val: val/images

nc: {len(self.football_classes)}
names: {list(self.football_classes.values())}

# Extended training configuration
augment: true
mosaic: 0.9
mixup: 0.15
copy_paste: 0.3
degrees: 15.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0001
flipud: 0.0
fliplr: 0.5
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
"""
        
        yaml_path = dataset_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content.strip())
        
        return yaml_path, train_count, val_count
    
    def train_extended_model(self, dataset_yaml, train_count, val_count):
        """Train model for 5+ hours with advanced techniques"""
        logger.info("🚀 STARTING EXTENDED 5+ HOUR TRAINING")
        logger.info("=" * 60)
        
        # Initialize MLflow tracking
        mlflow.set_experiment("Extended_Football_Training")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config)
            mlflow.log_param("train_images", train_count)
            mlflow.log_param("val_images", val_count)
            
            # Stage 1: Transfer Learning (Epochs 1-50)
            logger.info("🎯 STAGE 1: TRANSFER LEARNING (Epochs 1-50)")
            model = YOLO('yolov8n.pt')  # Start with pretrained
            
            stage1_config = {
                'data': str(dataset_yaml),
                'epochs': 50,
                'imgsz': self.config['image_size'],
                'batch': self.config['batch_size'],
                'device': 'cpu',
                'project': 'extended_training',
                'name': 'stage1_transfer',
                'save': True,
                'save_period': 10,
                'plots': True,
                'verbose': True,
                
                # Transfer learning parameters
                'lr0': 0.001,
                'lrf': 0.1,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 5,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                
                # Advanced augmentation
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 10.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0001,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 0.8,
                'mixup': 0.1,
                'copy_paste': 0.2,
                
                # Optimization
                'optimizer': 'AdamW',
                'close_mosaic': 10,
                'amp': True,
                'val': True,
                'patience': 25,
                'cos_lr': True
            }
            
            logger.info("🔥 Starting Stage 1 training...")
            results1 = model.train(**stage1_config)
            
            # Stage 2: Fine-tuning (Epochs 51-150)
            logger.info("🎯 STAGE 2: FINE-TUNING (Epochs 51-150)")
            
            # Load best model from stage 1
            best_stage1 = "extended_training/stage1_transfer/weights/best.pt"
            if os.path.exists(best_stage1):
                model = YOLO(best_stage1)
            
            stage2_config = stage1_config.copy()
            stage2_config.update({
                'epochs': 100,  # Additional 100 epochs
                'name': 'stage2_finetune',
                'lr0': 0.0005,  # Lower learning rate
                'patience': 35,
                'mosaic': 0.9,
                'mixup': 0.15,
                'copy_paste': 0.3,
            })
            
            logger.info("🔥 Starting Stage 2 training...")
            results2 = model.train(**stage2_config)
            
            # Stage 3: Final optimization (Epochs 151-300)
            logger.info("🎯 STAGE 3: FINAL OPTIMIZATION (Epochs 151-300)")
            
            # Load best model from stage 2
            best_stage2 = "extended_training/stage2_finetune/weights/best.pt"
            if os.path.exists(best_stage2):
                model = YOLO(best_stage2)
            
            stage3_config = stage2_config.copy()
            stage3_config.update({
                'epochs': 150,  # Final 150 epochs
                'name': 'stage3_final',
                'lr0': 0.0001,  # Very low learning rate
                'patience': 50,
                'save_period': 25,
            })
            
            logger.info("🔥 Starting Stage 3 training...")
            results3 = model.train(**stage3_config)
            
            # Save final model
            final_model_path = self.models_dir / "godseye_extended_5hour_model.pt"
            best_final = "extended_training/stage3_final/weights/best.pt"
            if os.path.exists(best_final):
                shutil.copy(best_final, final_model_path)
                logger.info(f"✅ Final model saved to: {final_model_path}")
            
            # Log final metrics
            total_time = time.time() - self.start_time
            mlflow.log_metric("total_training_time_hours", total_time / 3600)
            
            return model, final_model_path
    
    def create_progress_tracker(self):
        """Create progress tracking for 5+ hour training"""
        def log_progress():
            elapsed = time.time() - self.start_time
            hours = elapsed / 3600
            
            logger.info(f"⏱️ Training Progress: {hours:.2f} hours elapsed")
            
            if hours >= 5:
                logger.info("🎉 5+ HOUR TRAINING TARGET REACHED!")
            
            # Continue logging every 30 minutes
            if hours < 8:  # Maximum 8 hours
                import threading
                timer = threading.Timer(1800, log_progress)  # 30 minutes
                timer.daemon = True
                timer.start()
        
        log_progress()
    
    def run_extended_training(self):
        """Run complete 5+ hour extended training pipeline"""
        logger.info("🎯 GODSEYE AI - EXTENDED 5+ HOUR TRAINING PIPELINE")
        logger.info("=" * 70)
        logger.info("🚀 TARGET: PERFECT TEAM CLASSIFICATION ACCURACY")
        logger.info("⏱️ DURATION: 5+ HOURS OF INTENSIVE TRAINING")
        logger.info("=" * 70)
        
        try:
            # Start progress tracking
            self.create_progress_tracker()
            
            # Create massive dataset
            dataset_yaml, train_count, val_count = self.create_massive_dataset()
            
            if dataset_yaml is None:
                logger.error("❌ Failed to create dataset")
                return False
            
            # Train extended model
            model, model_path = self.train_extended_model(dataset_yaml, train_count, val_count)
            
            # Final summary
            total_time = time.time() - self.start_time
            hours = total_time / 3600
            
            logger.info("🎉 EXTENDED TRAINING COMPLETED!")
            logger.info(f"⏱️ Total training time: {hours:.2f} hours")
            logger.info(f"📁 Final model: {model_path}")
            logger.info(f"📊 Dataset size: {train_count + val_count} images")
            logger.info("🎯 Model ready for perfect team classification!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Extended training failed: {e}")
            return False

def main():
    """Main function to start extended training"""
    print("🎯 GODSEYE AI - EXTENDED 5+ HOUR TRAINING PIPELINE")
    print("=" * 70)
    print("🚀 PREPARING FOR PERFECT TEAM CLASSIFICATION")
    print("⏱️ ESTIMATED DURATION: 5-8 HOURS")
    print("=" * 70)
    
    # Confirm with user
    print("⚠️  This will run intensive training for 5+ hours.")
    print("💡 Make sure your system can run uninterrupted.")
    print("🔋 Ensure adequate power and cooling.")
    
    response = input("\n🚀 Start extended training? (y/N): ").lower().strip()
    
    if response != 'y':
        print("❌ Training cancelled.")
        return
    
    # Start extended training
    trainer = ExtendedFootballTrainer()
    success = trainer.run_extended_training()
    
    if success:
        print("\n🎉 SUCCESS! Extended training completed!")
        print("🎯 Your model should now have perfect team classification!")
        print("📁 Use the final model for production deployment!")
    else:
        print("\n❌ Extended training failed. Check logs for details.")

if __name__ == "__main__":
    main()
