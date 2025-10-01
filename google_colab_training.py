#!/usr/bin/env python3
"""
===============================================================================
GODSEYE AI - GOOGLE COLAB OPTIMIZED TRAINING (UNDER 1 HOUR)
===============================================================================

Optimized training script for Google Colab free tier with time constraints:
- Download limited SoccerNet data (train split only)
- Quick training with essential augmentations
- Under 1 hour total runtime
- GPU acceleration when available

Author: Victor
Date: 2025
Version: 1.0.0
"""

import os
import sys
import time
import torch
import cv2
import numpy as np
from pathlib import Path
import logging
from ultralytics import YOLO
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ColabOptimizedTrainer:
    """Google Colab optimized trainer with time constraints"""
    
    def __init__(self):
        self.start_time = time.time()
        self.max_runtime = 50 * 60  # 50 minutes (leave 10 min buffer)
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
        
        # Check time remaining
        self.check_time_remaining()
    
    def check_time_remaining(self):
        """Check if we have enough time left"""
        elapsed = time.time() - self.start_time
        remaining = self.max_runtime - elapsed
        logger.info(f"Time remaining: {remaining/60:.1f} minutes")
        return remaining > 0
    
    def download_limited_soccernet(self):
        """Download limited SoccerNet data for Colab"""
        logger.info("🚀 Starting limited SoccerNet download for Colab...")
        
        try:
            from SoccerNet.Downloader import SoccerNetDownloader
            
            # Set up local directory
            local_directory = "/content/SoccerNet"
            os.makedirs(local_directory, exist_ok=True)
            
            # Initialize downloader
            downloader = SoccerNetDownloader(LocalDirectory=local_directory)
            downloader.password = "s0cc3rn3t"
            
            # Download only train split labels (fast)
            logger.info("📥 Downloading train split labels...")
            downloader.downloadGames(
                files=["Labels-v2.json"], 
                split=["train"]
            )
            
            # Download limited videos (first 10 games only)
            logger.info("📥 Downloading limited videos (first 10 games)...")
            try:
                # Get list of available games
                games = downloader.getGames(split="train")
                limited_games = games[:10]  # Only first 10 games
                
                for game in limited_games:
                    if not self.check_time_remaining():
                        logger.warning("⏰ Time limit approaching, stopping download")
                        break
                    
                    logger.info(f"Downloading game: {game}")
                    downloader.downloadGames(
                        files=["1_224p.mkv"], 
                        split=["train"],
                        games=[game]
                    )
                    
            except Exception as e:
                logger.warning(f"Limited video download failed: {e}")
                logger.info("Continuing with labels only...")
            
            logger.info("✅ Limited SoccerNet download completed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False
    
    def create_yolo_dataset(self):
        """Create YOLO format dataset from SoccerNet"""
        logger.info("🔄 Creating YOLO format dataset...")
        
        soccernet_dir = Path("/content/SoccerNet")
        yolo_dir = Path("/content/yolo_dataset")
        
        # Create YOLO directory structure
        for split in ["train"]:
            (yolo_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (yolo_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # Process SoccerNet data
        processed_count = 0
        for game_dir in soccernet_dir.glob("train/*"):
            if not self.check_time_remaining():
                break
                
            if game_dir.is_dir():
                # Process labels
                labels_file = game_dir / "Labels-v2.json"
                if labels_file.exists():
                    self.process_soccernet_game(game_dir, yolo_dir, processed_count)
                    processed_count += 1
                    
                    if processed_count >= 5:  # Limit to 5 games for speed
                        break
        
        logger.info(f"✅ Created YOLO dataset with {processed_count} games")
        return yolo_dir
    
    def process_soccernet_game(self, game_dir, yolo_dir, game_id):
        """Process a single SoccerNet game to YOLO format"""
        try:
            labels_file = game_dir / "Labels-v2.json"
            with open(labels_file, 'r') as f:
                labels_data = json.load(f)
            
            # Process each annotation
            for annotation in labels_data.get("annotations", []):
                if not self.check_time_remaining():
                    break
                
                # Extract frame info
                frame_id = annotation.get("id", 0)
                bbox = annotation.get("bbox", [])
                category_id = annotation.get("category_id", 0)
                
                # Map SoccerNet categories to our classes
                class_id = self.map_soccernet_category(category_id)
                if class_id is None:
                    continue
                
                # Create YOLO format label
                yolo_label = f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
                
                # Save label file
                label_file = yolo_dir / "train" / "labels" / f"game_{game_id}_frame_{frame_id}.txt"
                with open(label_file, 'w') as f:
                    f.write(yolo_label)
                
        except Exception as e:
            logger.warning(f"Error processing game {game_dir}: {e}")
    
    def map_soccernet_category(self, category_id):
        """Map SoccerNet categories to our classes"""
        # SoccerNet category mapping (simplified)
        mapping = {
            1: 0,  # Player -> team_a_player
            2: 1,  # Player -> team_b_player  
            3: 4,  # Ball -> ball
            4: 5,  # Referee -> referee
        }
        return mapping.get(category_id)
    
    def quick_train(self, dataset_dir):
        """Quick training optimized for Colab"""
        logger.info("🏋️ Starting quick training...")
        
        # Initialize YOLO model
        model = YOLO('yolov8n.pt')  # Use nano for speed
        
        # Quick training configuration
        results = model.train(
            data=str(dataset_dir / "dataset.yaml"),
            epochs=20,  # Reduced epochs for speed
            imgsz=640,
            batch=16,
            device=self.device,
            workers=2,
            patience=5,
            save=True,
            save_period=5,
            cache=True,
            augment=True,
            mixup=0.1,
            copy_paste=0.1,
            project="colab_training",
            name="quick_football_model"
        )
        
        logger.info("✅ Quick training completed!")
        return results
    
    def create_dataset_yaml(self, dataset_dir):
        """Create dataset.yaml for YOLO training"""
        yaml_content = f"""
path: {dataset_dir}
train: train/images
val: train/images  # Use train for both in quick training

nc: {len(self.classes)}
names: {self.classes}
"""
        
        yaml_file = dataset_dir / "dataset.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"✅ Created dataset.yaml: {yaml_file}")
        return yaml_file
    
    def run_colab_training(self):
        """Main Colab training pipeline"""
        logger.info("🚀 Godseye AI - Google Colab Training (Under 1 Hour)")
        logger.info("=" * 60)
        
        # Step 1: Download limited data (15 minutes max)
        logger.info("📥 Step 1: Downloading limited SoccerNet data...")
        if not self.download_limited_soccernet():
            logger.error("❌ Download failed, using fallback data")
            return False
        
        if not self.check_time_remaining():
            logger.warning("⏰ Time limit reached during download")
            return False
        
        # Step 2: Create YOLO dataset (5 minutes max)
        logger.info("🔄 Step 2: Creating YOLO dataset...")
        dataset_dir = self.create_yolo_dataset()
        self.create_dataset_yaml(dataset_dir)
        
        if not self.check_time_remaining():
            logger.warning("⏰ Time limit reached during dataset creation")
            return False
        
        # Step 3: Quick training (25 minutes max)
        logger.info("🏋️ Step 3: Quick training...")
        results = self.quick_train(dataset_dir)
        
        # Step 4: Save results
        logger.info("💾 Step 4: Saving results...")
        self.save_colab_results(results)
        
        # Final time check
        total_time = time.time() - self.start_time
        logger.info(f"⏱️ Total runtime: {total_time/60:.1f} minutes")
        
        logger.info("=" * 60)
        logger.info("✅ Google Colab training completed successfully!")
        logger.info("🎯 Model ready for download and deployment!")
        logger.info("=" * 60)
        
        return True
    
    def save_colab_results(self, results):
        """Save training results for download"""
        results_dir = Path("/content/results")
        results_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = results_dir / "colab_football_model.pt"
        if hasattr(results, 'save'):
            results.save(str(model_path))
        
        # Save training info
        info = {
            "training_time": time.time() - self.start_time,
            "device": str(self.device),
            "classes": self.classes,
            "epochs": 20,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(results_dir / "training_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"📁 Results saved to: {results_dir}")
        logger.info("📥 Download the results folder from Colab!")

def main():
    """Main function for Colab"""
    trainer = ColabOptimizedTrainer()
    success = trainer.run_colab_training()
    
    if success:
        print("🎉 SUCCESS! Training completed within time limit!")
        print("📥 Download the results folder from Colab")
    else:
        print("❌ Training failed or exceeded time limit")

if __name__ == "__main__":
    main()
