#!/usr/bin/env python3
"""
===============================================================================
GODSEYE AI - ROBUST FIXED TRAINING PIPELINE
===============================================================================

Fixed training pipeline addressing all critical issues:
- Proper SoccerNet data loading and preprocessing
- Correct YOLO format conversion
- Robust model architecture and training
- Comprehensive error handling and validation
- Proper evaluation and metrics

Author: Victor
Date: 2025
Version: 2.0.0 - FIXED
"""

import os
import sys
import time
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from ultralytics import YOLO
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.pytorch
from collections import defaultdict, Counter
import shutil
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustFixedTrainer:
    """Fixed trainer addressing all critical issues"""
    
    def __init__(self):
        self.start_time = time.time()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Core classes for football analytics
        self.classes = [
            'team_a_player',      # 0
            'team_b_player',      # 1
            'team_a_goalkeeper',  # 2
            'team_b_goalkeeper',  # 3
            'ball',               # 4
            'referee',            # 5
            'assistant_referee',  # 6
            'others'              # 7
        ]
        
        # Initialize MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Godseye_AI_Fixed_Training")
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_map': [],
            'val_map': [],
            'learning_rate': []
        }
    
    def download_soccernet_safely(self):
        """Safely download SoccerNet dataset with proper error handling"""
        logger.info("🚀 Starting safe SoccerNet download...")
        
        try:
            from SoccerNet.Downloader import SoccerNetDownloader
            
            # Set up local directory
            local_directory = "/Users/user/Football_analytics/data/SoccerNet"
            os.makedirs(local_directory, exist_ok=True)
            
            # Initialize downloader
            downloader = SoccerNetDownloader(LocalDirectory=local_directory)
            downloader.password = "s0cc3rn3t"
            
            # Download labels first (small, fast)
            logger.info("📥 Downloading labels...")
            downloader.downloadGames(
                files=["Labels-v2.json"], 
                split=["train", "valid", "test"]
            )
            
            # Download limited videos for testing
            logger.info("📥 Downloading limited videos for testing...")
            games = downloader.getGames(split="train")
            limited_games = games[:5]  # Only 5 games for testing
            
            for i, game in enumerate(limited_games):
                logger.info(f"Downloading game {i+1}/5: {game}")
                try:
                    downloader.downloadGames(
                        files=["1_224p.mkv"], 
                        split=["train"],
                        games=[game]
                    )
                except Exception as e:
                    logger.warning(f"Failed to download game {game}: {e}")
                    continue
            
            logger.info("✅ Safe SoccerNet download completed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False
    
    def extract_frames_from_videos(self, video_dir, output_dir, max_frames_per_video=100):
        """Extract frames from videos with proper error handling"""
        logger.info("🎬 Extracting frames from videos...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted_count = 0
        
        for video_file in video_dir.rglob("*.mkv"):
            try:
                logger.info(f"Processing video: {video_file.name}")
                
                # Create output directory for this video
                video_output_dir = output_dir / video_file.stem
                video_output_dir.mkdir(exist_ok=True)
                
                # Extract frames
                cap = cv2.VideoCapture(str(video_file))
                frame_count = 0
                
                while cap.isOpened() and frame_count < max_frames_per_video:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Save frame
                    frame_path = video_output_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_count += 1
                
                cap.release()
                extracted_count += frame_count
                logger.info(f"Extracted {frame_count} frames from {video_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing video {video_file}: {e}")
                continue
        
        logger.info(f"✅ Extracted {extracted_count} frames total")
        return extracted_count > 0
    
    def create_yolo_dataset_properly(self, soccernet_dir, output_dir):
        """Create YOLO dataset with proper format and validation"""
        logger.info("🔄 Creating proper YOLO dataset...")
        
        # Create output directory structure
        for split in ["train", "val"]:
            (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # Process each split
        total_images = 0
        total_labels = 0
        
        for split in ["train", "valid"]:
            split_dir = soccernet_dir / split
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue
            
            # Extract frames from videos
            frames_dir = output_dir / f"{split}_frames"
            if self.extract_frames_from_videos(split_dir, frames_dir):
                # Process labels and create YOLO format
                images, labels = self.process_split_properly(split_dir, frames_dir, output_dir / split)
                total_images += images
                total_labels += labels
        
        logger.info(f"✅ Created YOLO dataset: {total_images} images, {total_labels} labels")
        return total_images > 0
    
    def process_split_properly(self, split_dir, frames_dir, output_split_dir):
        """Process a split properly with validation"""
        images_processed = 0
        labels_processed = 0
        
        # Process each game
        for game_dir in split_dir.glob("*"):
            if not game_dir.is_dir():
                continue
            
            # Find labels file
            labels_file = game_dir / "Labels-v2.json"
            if not labels_file.exists():
                logger.warning(f"No labels file found for {game_dir.name}")
                continue
            
            try:
                # Load labels
                with open(labels_file, 'r') as f:
                    labels_data = json.load(f)
                
                # Process annotations
                for annotation in labels_data.get("annotations", []):
                    frame_id = annotation.get("id", 0)
                    bbox = annotation.get("bbox", [])
                    category_id = annotation.get("category_id", 0)
                    
                    # Validate bbox
                    if len(bbox) != 4 or any(x < 0 or x > 1 for x in bbox):
                        continue
                    
                    # Map category to our classes
                    class_id = self.map_category_properly(category_id)
                    if class_id is None:
                        continue
                    
                    # Find corresponding frame
                    frame_path = frames_dir / game_dir.name / f"frame_{frame_id:06d}.jpg"
                    if not frame_path.exists():
                        continue
                    
                    # Copy frame to output directory
                    output_frame_path = output_split_dir / "images" / f"{game_dir.name}_{frame_id:06d}.jpg"
                    shutil.copy2(frame_path, output_frame_path)
                    
                    # Create YOLO label
                    yolo_label = f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
                    label_path = output_split_dir / "labels" / f"{game_dir.name}_{frame_id:06d}.txt"
                    
                    with open(label_path, 'w') as f:
                        f.write(yolo_label)
                    
                    images_processed += 1
                    labels_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing game {game_dir.name}: {e}")
                continue
        
        return images_processed, labels_processed
    
    def map_category_properly(self, category_id):
        """Map SoccerNet categories to our classes properly"""
        # SoccerNet category mapping
        mapping = {
            1: 0,  # Player -> team_a_player
            2: 1,  # Player -> team_b_player  
            3: 4,  # Ball -> ball
            4: 5,  # Referee -> referee
            5: 6,  # Assistant referee -> assistant_referee
            6: 2,  # Goalkeeper -> team_a_goalkeeper
            7: 3,  # Goalkeeper -> team_b_goalkeeper
        }
        return mapping.get(category_id)
    
    def create_dataset_yaml_properly(self, dataset_dir):
        """Create dataset.yaml with proper validation"""
        yaml_content = f"""
path: {dataset_dir}
train: train/images
val: val/images

nc: {len(self.classes)}
names: {self.classes}
"""
        
        yaml_file = dataset_dir / "dataset.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        # Validate dataset
        if self.validate_dataset(dataset_dir):
            logger.info(f"✅ Created and validated dataset.yaml: {yaml_file}")
            return yaml_file
        else:
            logger.error("❌ Dataset validation failed")
            return None
    
    def validate_dataset(self, dataset_dir):
        """Validate dataset structure and content"""
        logger.info("🔍 Validating dataset...")
        
        # Check directory structure
        required_dirs = ["train/images", "train/labels", "val/images", "val/labels"]
        for dir_path in required_dirs:
            if not (dataset_dir / dir_path).exists():
                logger.error(f"Missing directory: {dir_path}")
                return False
        
        # Check for images and labels
        train_images = list((dataset_dir / "train/images").glob("*.jpg"))
        train_labels = list((dataset_dir / "train/labels").glob("*.txt"))
        val_images = list((dataset_dir / "val/images").glob("*.jpg"))
        val_labels = list((dataset_dir / "val/labels").glob("*.txt"))
        
        if len(train_images) == 0 or len(val_images) == 0:
            logger.error("No images found in dataset")
            return False
        
        if len(train_labels) == 0 or len(val_labels) == 0:
            logger.error("No labels found in dataset")
            return False
        
        # Check label format
        for label_file in train_labels[:5]:  # Check first 5 labels
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            logger.error(f"Invalid label format in {label_file}")
                            return False
                        class_id = int(parts[0])
                        if class_id < 0 or class_id >= len(self.classes):
                            logger.error(f"Invalid class ID {class_id} in {label_file}")
                            return False
            except Exception as e:
                logger.error(f"Error reading label file {label_file}: {e}")
                return False
        
        logger.info(f"✅ Dataset validation passed: {len(train_images)} train, {len(val_images)} val")
        return True
    
    def train_model_properly(self, dataset_dir):
        """Train model with proper configuration and validation"""
        logger.info("🏋️ Starting proper model training...")
        
        # Initialize model with correct configuration
        model = YOLO('yolov8n.pt')  # Start with nano for testing
        
        # Proper training configuration
        results = model.train(
            data=str(dataset_dir / "dataset.yaml"),
            epochs=50,  # Reasonable number for testing
            imgsz=640,
            batch=8,  # Smaller batch for stability
            device=self.device,
            workers=2,  # Reduced workers for stability
            patience=10,
            save=True,
            save_period=5,
            cache=False,  # Disable cache to avoid memory issues
            augment=True,
            mixup=0.1,
            copy_paste=0.1,
            mosaic=0.5,
            degrees=5.0,
            translate=0.05,
            scale=0.3,
            shear=1.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            bgr=0.0,
            hsv_h=0.01,
            hsv_s=0.5,
            hsv_v=0.3,
            project="fixed_training",
            name="robust_football_model"
        )
        
        logger.info("✅ Proper model training completed!")
        return results
    
    def evaluate_model_properly(self, model_path):
        """Evaluate model with proper metrics"""
        logger.info("📊 Starting proper model evaluation...")
        
        try:
            # Load trained model
            model = YOLO(model_path)
            
            # Run validation
            results = model.val()
            
            # Extract metrics properly
            metrics = {
                'mAP50': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
                'mAP50-95': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
                'precision': float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
                'recall': float(results.box.mr) if hasattr(results.box, 'mr') else 0.0,
            }
            
            # Calculate F1 score
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1'] = 0.0
            
            logger.info("📊 Evaluation Results:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return None
    
    def save_results_properly(self, results, metrics):
        """Save results with proper validation"""
        results_dir = Path("results/fixed_training")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = results_dir / "fixed_football_model.pt"
        try:
            if hasattr(results, 'save'):
                results.save(str(model_path))
            else:
                # Copy from training directory
                source_model = Path("fixed_training/robust_football_model/weights/best.pt")
                if source_model.exists():
                    shutil.copy2(source_model, model_path)
                else:
                    logger.warning("No model file found to save")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
        
        # Save training info
        info = {
            "training_time": time.time() - self.start_time,
            "device": str(self.device),
            "classes": self.classes,
            "epochs": 50,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "model_path": str(model_path),
            "status": "completed" if metrics else "failed"
        }
        
        with open(results_dir / "training_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"📁 Results saved to: {results_dir}")
        return results_dir
    
    def run_fixed_training(self):
        """Main fixed training pipeline"""
        logger.info("🚀 Godseye AI - Fixed Training Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Download dataset safely
            logger.info("📥 Step 1: Downloading dataset safely...")
            if not self.download_soccernet_safely():
                logger.error("❌ Dataset download failed")
                return False
            
            # Step 2: Create proper YOLO dataset
            logger.info("🔄 Step 2: Creating proper YOLO dataset...")
            soccernet_dir = Path("/Users/user/Football_analytics/data/SoccerNet")
            dataset_dir = Path("/Users/user/Football_analytics/data/yolo_dataset")
            
            if not self.create_yolo_dataset_properly(soccernet_dir, dataset_dir):
                logger.error("❌ Dataset creation failed")
                return False
            
            # Step 3: Create and validate dataset.yaml
            logger.info("📝 Step 3: Creating dataset.yaml...")
            yaml_file = self.create_dataset_yaml_properly(dataset_dir)
            if not yaml_file:
                logger.error("❌ Dataset.yaml creation failed")
                return False
            
            # Step 4: Train model properly
            logger.info("🏋️ Step 4: Training model properly...")
            results = self.train_model_properly(dataset_dir)
            
            # Step 5: Evaluate model properly
            logger.info("📊 Step 5: Evaluating model properly...")
            model_path = "fixed_training/robust_football_model/weights/best.pt"
            metrics = self.evaluate_model_properly(model_path)
            
            # Step 6: Save results properly
            logger.info("💾 Step 6: Saving results properly...")
            results_dir = self.save_results_properly(results, metrics)
            
            # Final time check
            total_time = time.time() - self.start_time
            logger.info(f"⏱️ Total runtime: {total_time/3600:.1f} hours")
            
            if metrics and metrics['mAP50'] > 0.1:  # Basic success threshold
                logger.info("=" * 60)
                logger.info("✅ Fixed training completed successfully!")
                logger.info(f"🎯 Model saved to: {results_dir}")
                logger.info("🚀 Ready for testing!")
                logger.info("=" * 60)
                return True
            else:
                logger.error("❌ Training failed - poor model performance")
                return False
                
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            return False

def main():
    """Main function for fixed training"""
    trainer = RobustFixedTrainer()
    success = trainer.run_fixed_training()
    
    if success:
        print("🎉 SUCCESS! Fixed training completed!")
        print("🎯 Model ready for testing!")
    else:
        print("❌ Training failed - check logs for details")

if __name__ == "__main__":
    main()
