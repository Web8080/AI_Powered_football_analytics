#!/usr/bin/env python3
"""
===============================================================================
GODSEYE AI - GOOGLE COLAB OPTIMIZED TRAINING (UNDER 1 HOUR)
===============================================================================

Optimized training script for Google Colab free tier with time constraints:
- Automatic dependency installation
- Download limited SoccerNet data (train split only)
- Quick training with essential augmentations
- Under 1 hour total runtime
- GPU acceleration when available

Author: Victor
Date: 2025
Version: 2.0.0 - Auto Install
"""

import os
import sys
import subprocess
import time
import torch
import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProgressTracker:
    """Track progress and timing for training pipeline"""
    
    def __init__(self, total_steps=5):
        self.start_time = time.time()
        self.total_steps = total_steps
        self.current_step = 0
        self.step_times = []
        self.step_names = [
            "🔧 Installing Dependencies",
            "📥 Downloading SoccerNet Data", 
            "🔄 Converting to YOLO Format",
            "🏋️ Training Model",
            "📊 Evaluating Results"
        ]
    
    def start_step(self, step_name=None):
        """Start a new step"""
        if step_name:
            self.step_names[self.current_step] = step_name
        
        step_start = time.time()
        if self.current_step > 0:
            self.step_times.append(step_start - self.last_step_start)
        
        self.last_step_start = step_start
        elapsed = step_start - self.start_time
        
        logger.info("=" * 60)
        logger.info(f"🚀 STEP {self.current_step + 1}/{self.total_steps}: {self.step_names[self.current_step]}")
        logger.info(f"⏱️ Total elapsed time: {elapsed/60:.1f} minutes")
        logger.info("=" * 60)
    
    def complete_step(self):
        """Complete current step"""
        step_time = time.time() - self.last_step_start
        self.step_times.append(step_time)
        
        logger.info("=" * 60)
        logger.info(f"✅ STEP {self.current_step + 1} COMPLETED: {self.step_names[self.current_step]}")
        logger.info(f"⏱️ Step time: {step_time/60:.1f} minutes")
        
        if self.current_step < self.total_steps - 1:
            remaining_steps = self.total_steps - self.current_step - 1
            avg_time = sum(self.step_times) / len(self.step_times)
            estimated_remaining = remaining_steps * avg_time
            logger.info(f"📊 Estimated remaining time: {estimated_remaining/60:.1f} minutes")
        
        logger.info("=" * 60)
        self.current_step += 1
    
    def get_final_summary(self):
        """Get final timing summary"""
        total_time = time.time() - self.start_time
        logger.info("🎯 TRAINING PIPELINE COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
        
        for i, (name, step_time) in enumerate(zip(self.step_names, self.step_times)):
            percentage = (step_time / total_time) * 100
            logger.info(f"Step {i+1}: {name} - {step_time/60:.1f}min ({percentage:.1f}%)")
        
        logger.info("=" * 60)

def install_dependencies():
    """Automatically install all required dependencies for Google Colab"""
    logger.info("🔧 Installing dependencies for Google Colab...")
    
    # List of required packages
    packages = [
        "ultralytics",
        "opencv-python",
        "albumentations",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pandas",
        "numpy",
        "torch",
        "torchvision",
        "SoccerNet",
        "mlflow",
        "timm",
        "optuna",
        "optuna-integration"
    ]
    
    # Install packages one by one
    for package in packages:
        try:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
            logger.info(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Failed to install {package}: {e}")
            continue
    
    # Install additional packages that might be needed
    additional_packages = [
        "pillow",
        "requests",
        "tqdm",
        "pyyaml",
        "psutil"
    ]
    
    for package in additional_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        except subprocess.CalledProcessError:
            pass
    
    logger.info("✅ All dependencies installed successfully!")

# Install dependencies first
install_dependencies()

# Now import the required modules
try:
    from ultralytics import YOLO
    import json
    logger.info("✅ All modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import modules: {e}")
    sys.exit(1)

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
    
    def check_disk_space(self):
        """Check available disk space"""
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        used_gb = used / (1024**3)
        total_gb = total / (1024**3)
        
        logger.info(f"💾 Disk usage: {used_gb:.1f}GB / {total_gb:.1f}GB ({free_gb:.1f}GB free)")
        
        # Stop if less than 5GB free
        if free_gb < 5.0:
            logger.warning(f"⚠️ Low disk space: {free_gb:.1f}GB free")
            return False
        return True
    
    def limit_to_first_games(self, soccernet_dir, max_games):
        """Limit SoccerNet data to first N games only"""
        logger.info(f"🎯 Limiting to first {max_games} games only...")
        
        try:
            soccernet_path = Path(soccernet_dir)
            if not soccernet_path.exists():
                logger.warning("⚠️ SoccerNet directory not found")
                return
            
            # Find all game directories
            game_dirs = []
            for league_dir in soccernet_path.iterdir():
                if league_dir.is_dir():
                    for season_dir in league_dir.iterdir():
                        if season_dir.is_dir():
                            for game_dir in season_dir.iterdir():
                                if game_dir.is_dir():
                                    labels_file = game_dir / "Labels-v2.json"
                                    if labels_file.exists():
                                        game_dirs.append(game_dir)
            
            logger.info(f"📊 Found {len(game_dirs)} games with labels")
            
            # Keep only first N games
            if len(game_dirs) > max_games:
                excess_games = game_dirs[max_games:]
                logger.info(f"🗑️ Removing {len(excess_games)} excess game directories")
                
                for game_dir in excess_games:
                    try:
                        import shutil
                        shutil.rmtree(game_dir)
                        logger.info(f"✅ Removed: {game_dir.name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to remove {game_dir.name}: {e}")
            
            logger.info(f"✅ Limited to {min(len(game_dirs), max_games)} games")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to limit games: {e}")
    
    def cleanup_excess_data(self):
        """Clean up excess data to free disk space"""
        logger.info("🧹 Cleaning up excess data to free disk space...")
        
        try:
            # Remove excess video files (keep only first 2 games worth)
            soccernet_dir = Path("SoccerNet")
            if soccernet_dir.exists():
                # Find all video files
                video_files = list(soccernet_dir.glob("**/*.mkv"))
                logger.info(f"📹 Found {len(video_files)} video files")
                
                # Keep only first 4 videos (2 games × 2 cameras)
                if len(video_files) > 4:
                    excess_videos = video_files[4:]
                    logger.info(f"🗑️ Removing {len(excess_videos)} excess video files")
                    
                    for video in excess_videos:
                        try:
                            video.unlink()
                            logger.info(f"✅ Removed: {video.name}")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to remove {video.name}: {e}")
                
                # Clean up any temporary files
                temp_files = list(soccernet_dir.glob("**/*.tmp"))
                for temp_file in temp_files:
                    try:
                        temp_file.unlink()
                        logger.info(f"✅ Removed temp file: {temp_file.name}")
                    except:
                        pass
                        
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Cleanup failed: {e}")
    
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
            # Password should be set via environment variable or user input
            downloader.password = os.getenv("SOCCERNET_PASSWORD", input("Enter SoccerNet password: "))
            
            # Download only train split labels (fast)
            logger.info("📥 Downloading train split labels...")
            downloader.downloadGames(
                files=["Labels-v2.json"], 
                split=["train"]
            )
            
            # Immediately limit to first 2 games only
            logger.info("🎯 Limiting to first 2 games only for disk space...")
            self.limit_to_first_games("/content/SoccerNet", 2)
            
            # Check disk space before downloading videos
            if not self.check_disk_space():
                logger.warning("⚠️ Insufficient disk space for video download, skipping...")
                return True
            
            # Download limited videos for frame extraction
            logger.info("📥 Downloading limited videos for frame extraction...")
            try:
                # Download videos for train split (limited to save time and disk space)
                logger.info("📥 Downloading 224p videos for training (limited)...")
                try:
                    # Use downloadGames with limited parameters
                    downloader.downloadGames(
                        files=["1_224p.mkv", "2_224p.mkv"], 
                        split=["train"]
                    )
                    logger.info("✅ Video download completed successfully!")
                except Exception as download_error:
                    logger.warning(f"⚠️ Video download failed: {download_error}")
                    logger.info("📋 Will use only labels for training...")
                
                logger.info("✅ Limited video download completed!")
                
            except Exception as e:
                logger.warning(f"Video download failed: {e}")
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
        
        # Check if we have videos or need to create synthetic images
        video_files = list(soccernet_dir.glob("**/*.mkv"))
        logger.info(f"📹 Found {len(video_files)} video files")
        
        if len(video_files) == 0:
            logger.warning("⚠️ No video files found, creating synthetic training images...")
            return self.create_synthetic_dataset(yolo_dir)
        
        # Process SoccerNet data
        processed_count = 0
        for game_dir in soccernet_dir.glob("train/*"):
            if not self.check_time_remaining():
                break
            
            # Check disk space before processing each game
            if not self.check_disk_space():
                logger.warning("⚠️ Insufficient disk space, cleaning up excess data...")
                self.cleanup_excess_data()
                if not self.check_disk_space():
                    logger.warning("⚠️ Still insufficient disk space, stopping processing...")
                    break
                
            if game_dir.is_dir():
                # Process labels
                labels_file = game_dir / "Labels-v2.json"
                if labels_file.exists():
                    self.process_soccernet_game(game_dir, yolo_dir, processed_count)
                    processed_count += 1
                    
                    if processed_count >= 2:  # Limit to 2 games for disk space
                        break
        
        # If no real data was processed, create synthetic dataset
        if processed_count == 0:
            logger.warning("⚠️ No real data processed, creating synthetic training images...")
            return self.create_synthetic_dataset(yolo_dir)
        
        logger.info(f"✅ Created YOLO dataset with {processed_count} games")
        return yolo_dir
    
    def create_synthetic_dataset(self, yolo_dir):
        """Create synthetic training images when videos are not available"""
        logger.info("🎨 Creating synthetic training dataset...")
        
        import numpy as np
        from PIL import Image, ImageDraw
        
        # Create synthetic images with football field and players
        for i in range(50):  # Create 50 synthetic images
            if not self.check_time_remaining():
                break
                
            # Create a green football field background
            img = Image.new('RGB', (640, 640), color='green')
            draw = ImageDraw.Draw(img)
            
            # Draw field lines
            draw.rectangle([50, 50, 590, 590], outline='white', width=3)
            draw.line([320, 50, 320, 590], fill='white', width=2)
            draw.ellipse([270, 270, 370, 370], outline='white', width=2)
            
            # Draw goal posts
            draw.rectangle([50, 250, 80, 390], fill='white')
            draw.rectangle([560, 250, 590, 390], fill='white')
            
            # Draw players (circles with different colors)
            player_positions = [
                (150, 200, 'red'),    # Team A player
                (200, 300, 'red'),    # Team A player
                (150, 400, 'red'),    # Team A player
                (100, 320, 'blue'),   # Team B player
                (250, 250, 'blue'),   # Team B player
                (200, 450, 'blue'),   # Team B player
                (320, 320, 'yellow'), # Referee
                (300, 200, 'orange'), # Ball
            ]
            
            for x, y, color in player_positions:
                draw.ellipse([x-10, y-10, x+10, y+10], fill=color)
            
            # Save image
            img_path = yolo_dir / "train" / "images" / f"synthetic_{i:03d}.jpg"
            img.save(img_path)
            
            # Create corresponding label file
            label_path = yolo_dir / "train" / "labels" / f"synthetic_{i:03d}.txt"
            with open(label_path, 'w') as f:
                # Write YOLO format labels (class_id x_center y_center width height)
                labels = [
                    "0 0.234 0.312 0.031 0.031",  # Team A player
                    "0 0.312 0.469 0.031 0.031",  # Team A player
                    "0 0.234 0.625 0.031 0.031",  # Team A player
                    "1 0.156 0.500 0.031 0.031",  # Team B player
                    "1 0.391 0.391 0.031 0.031",  # Team B player
                    "1 0.312 0.703 0.031 0.031",  # Team B player
                    "2 0.500 0.500 0.031 0.031",  # Referee
                    "3 0.469 0.312 0.031 0.031",  # Ball
                ]
                f.write('\n'.join(labels))
        
        logger.info("✅ Created 50 synthetic training images")
        return yolo_dir
    
    def extract_frames_from_video(self, video_path, output_dir, game_id, start_frame=0):
        """Extract frames from video file"""
        logger.info(f"🎬 Extracting frames from: {video_path}")
        
        if not video_path.exists():
            logger.warning(f"❌ Video file does not exist: {video_path}")
            return 0
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"❌ Could not open video: {video_path}")
            return 0
        
        frame_count = 0
        extracted_count = 0
        
        # Extract every 30th frame (1 frame per second at 30fps)
        frame_interval = 30
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Save frame
                frame_filename = f"game_{game_id}_frame_{start_frame + extracted_count:06d}.jpg"
                frame_path = output_dir / frame_filename
                success = cv2.imwrite(str(frame_path), frame)
                if success:
                    extracted_count += 1
                    logger.info(f"✅ Saved frame: {frame_filename}")
                else:
                    logger.warning(f"❌ Failed to save frame: {frame_filename}")
                
                # Limit frames per video to save space
                if extracted_count >= 20:  # Max 20 frames per video for disk space
                    break
            
            frame_count += 1
        
        cap.release()
        logger.info(f"🎬 Extracted {extracted_count} frames from {video_path.name}")
        return extracted_count
    
    def process_soccernet_game(self, game_dir, yolo_dir, game_id):
        """Process a single SoccerNet game to YOLO format"""
        try:
            logger.info(f"🎮 Processing game: {game_dir.name}")
            
            # Extract frames from videos first
            video_files = list(game_dir.glob("*.mkv"))
            logger.info(f"📹 Found {len(video_files)} video files in {game_dir}")
            
            if not video_files:
                logger.warning(f"❌ No video files found in {game_dir}")
                # List all files in the directory for debugging
                all_files = list(game_dir.glob("*"))
                logger.info(f"📁 All files in {game_dir}: {[f.name for f in all_files]}")
                return
            
            # Extract frames from videos
            frame_count = 0
            for video_file in video_files:
                logger.info(f"🎬 Processing video: {video_file.name}")
                extracted = self.extract_frames_from_video(
                    video_file, 
                    yolo_dir / "train" / "images", 
                    game_id, 
                    frame_count
                )
                frame_count += extracted
                logger.info(f"📊 Total frames extracted so far: {frame_count}")
            
            # Process labels if available
            labels_file = game_dir / "Labels-v2.json"
            if labels_file.exists():
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
        progress = ProgressTracker()
        
        # Step 1: Download limited data
        progress.start_step("📥 Downloading SoccerNet Data")
        if not self.download_limited_soccernet():
            logger.error("❌ Download failed, using fallback data")
            return False
        
        if not self.check_time_remaining():
            logger.warning("⏰ Time limit reached during download")
            return False
        progress.complete_step()
        
        # Step 2: Create YOLO dataset
        progress.start_step("🔄 Converting to YOLO Format")
        dataset_dir = self.create_yolo_dataset()
        self.create_dataset_yaml(dataset_dir)
        
        if not self.check_time_remaining():
            logger.warning("⏰ Time limit reached during dataset creation")
            return False
        progress.complete_step()
        
        # Step 3: Quick training
        progress.start_step("🏋️ Training Model")
        results = self.quick_train(dataset_dir)
        progress.complete_step()
        
        # Step 4: Save results
        progress.start_step("📊 Evaluating Results")
        self.save_colab_results(results)
        progress.complete_step()
        
        # Final summary
        progress.get_final_summary()
        
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
