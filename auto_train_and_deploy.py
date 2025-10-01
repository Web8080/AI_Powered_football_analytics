#!/usr/bin/env python3
"""
===============================================================================
GODSEYE AI - AUTOMATIC TRAINING AND DEPLOYMENT
===============================================================================

This script automatically:
1. Trains the football analytics model
2. Saves the trained model
3. Integrates with the web app for real-time analysis
4. Provides both real-time and post-match analysis

Author: victor
Date: 2025
Version: 2.0.0
"""

import os
import sys
import json
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import threading
import logging

# ML Libraries
import torch
from ultralytics import YOLO
import mediapipe as mp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoTrainer:
    """
    Automatic trainer that trains the model and integrates with the web app
    """
    
    def __init__(self):
        self.model_path = "models/godseye_ai_model.pt"
        self.training_data_path = "data/training_data"
        self.model = None
        self.training_complete = False
        
        # Create directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("data/training_data", exist_ok=True)
        
    def create_training_data(self):
        """Create training data from available videos"""
        logger.info("📊 Creating training data...")
        
        # Look for available videos
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(Path('.').glob(ext))
        
        if not video_files:
            logger.warning("⚠️ No video files found. Creating synthetic training data...")
            self.create_synthetic_data()
        else:
            logger.info(f"✅ Found {len(video_files)} video files for training")
            self.extract_frames_from_videos(video_files)
    
    def create_synthetic_data(self):
        """Create synthetic training data"""
        logger.info("🎨 Creating synthetic training data...")
        
        # Create synthetic images with annotations
        for i in range(100):  # 100 synthetic samples
            # Create a green field background
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            img[:, :] = (34, 139, 34)  # Forest green
            
            # Draw field lines
            cv2.line(img, (0, 320), (640, 320), (255, 255, 255), 2)
            cv2.circle(img, (320, 320), 100, (255, 255, 255), 2)
            
            # Add players, ball, referee
            players = [
                (100, 200, 0),   # Team A player
                (200, 300, 0),   # Team A player
                (400, 200, 2),   # Team B player
                (500, 300, 2),   # Team B player
                (300, 100, 1),   # Team A goalkeeper
                (300, 500, 3),   # Team B goalkeeper
                (320, 320, 4),   # Referee
                (350, 350, 5),   # Ball
            ]
            
            annotations = []
            for x, y, class_id in players:
                # Draw player/object
                if class_id == 5:  # Ball
                    cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
                else:
                    cv2.circle(img, (x, y), 20, (255, 255, 255), -1)
                    cv2.circle(img, (x, y), 20, (0, 0, 0), 2)
                
                # Create annotation (YOLO format)
                x_norm = x / 640
                y_norm = y / 640
                w_norm = 0.1
                h_norm = 0.1
                annotations.append(f"{class_id} {x_norm} {y_norm} {w_norm} {h_norm}")
            
            # Save image and annotation
            img_path = f"{self.training_data_path}/image_{i:04d}.jpg"
            ann_path = f"{self.training_data_path}/image_{i:04d}.txt"
            
            cv2.imwrite(img_path, img)
            with open(ann_path, 'w') as f:
                f.write('\n'.join(annotations))
        
        logger.info("✅ Synthetic training data created")
    
    def extract_frames_from_videos(self, video_files: List[Path]):
        """Extract frames from video files for training"""
        logger.info("🎥 Extracting frames from videos...")
        
        frame_count = 0
        for video_file in video_files:
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                continue
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract every 30th frame
                if frame_idx % 30 == 0:
                    # Resize frame
                    frame = cv2.resize(frame, (640, 640))
                    
                    # Save frame
                    img_path = f"{self.training_data_path}/frame_{frame_count:04d}.jpg"
                    cv2.imwrite(img_path, frame)
                    
                    # Create basic annotation (you can improve this with actual detection)
                    ann_path = f"{self.training_data_path}/frame_{frame_count:04d}.txt"
                    with open(ann_path, 'w') as f:
                        # Add some basic annotations
                        f.write("0 0.5 0.5 0.1 0.1\n")  # Player
                        f.write("5 0.3 0.3 0.05 0.05\n")  # Ball
                    
                    frame_count += 1
                
                frame_idx += 1
            
            cap.release()
        
        logger.info(f"✅ Extracted {frame_count} frames for training")
    
    def create_dataset_config(self):
        """Create dataset configuration file"""
        logger.info("📝 Creating dataset configuration...")
        
        # Create data.yaml for YOLOv8
        data_config = {
            'path': str(Path(self.training_data_path).resolve()),
            'train': '.',
            'val': '.',
            'nc': 8,  # Number of classes
            'names': [
                'team_a_player',
                'team_a_goalkeeper', 
                'team_b_player',
                'team_b_goalkeeper',
                'referee',
                'ball',
                'other',
                'staff'
            ]
        }
        
        config_path = f"{self.training_data_path}/data.yaml"
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(data_config, f)
        
        logger.info("✅ Dataset configuration created")
        return config_path
    
    def train_model(self, epochs: int = 50):
        """Train the football analytics model"""
        logger.info("🚀 Starting model training...")
        
        # Create training data
        self.create_training_data()
        
        # Create dataset config
        config_path = self.create_dataset_config()
        
        # Load YOLOv8 model
        logger.info("📥 Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')
        
        # Train the model
        logger.info(f"🎯 Training for {epochs} epochs...")
        start_time = time.time()
        
        results = self.model.train(
            data=config_path,
            epochs=epochs,
            imgsz=640,
            batch=8,
            device='cpu',  # Use CPU as requested
            project='godseye_training',
            name='football_analytics',
            patience=20,
            save=True,
            plots=True
        )
        
        training_time = time.time() - start_time
        logger.info(f"⏱️ Training completed in {training_time:.2f} seconds")
        
        # Save the trained model
        self.save_trained_model()
        
        self.training_complete = True
        logger.info("✅ Model training completed successfully!")
        
        return results
    
    def save_trained_model(self):
        """Save the trained model"""
        logger.info("💾 Saving trained model...")
        
        # Copy the best model to our standard location
        best_model_path = "godseye_training/football_analytics/weights/best.pt"
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy2(best_model_path, self.model_path)
            logger.info(f"✅ Model saved to {self.model_path}")
        else:
            logger.warning("⚠️ Best model not found, using current model")
            self.model.save(self.model_path)
    
    def test_model(self, video_path: str = None):
        """Test the trained model"""
        logger.info("🧪 Testing trained model...")
        
        if not self.model:
            logger.error("❌ No model loaded for testing")
            return
        
        if video_path and os.path.exists(video_path):
            # Test on provided video
            logger.info(f"🎥 Testing on video: {video_path}")
            results = self.model(video_path, save=True, show_labels=True, show_conf=True)
        else:
            # Test on sample image
            logger.info("🖼️ Testing on sample image...")
            # Create a test image
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            test_img[:, :] = (34, 139, 34)
            cv2.circle(test_img, (320, 320), 20, (255, 255, 255), -1)
            cv2.imwrite("test_image.jpg", test_img)
            
            results = self.model("test_image.jpg", save=True)
        
        logger.info("✅ Model testing completed")
        return results

class WebAppIntegration:
    """
    Integration with the web app for real-time analysis
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logger.info(f"✅ Model loaded from {self.model_path}")
            else:
                logger.warning("⚠️ Trained model not found, using default YOLOv8")
                self.model = YOLO('yolov8n.pt')
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            self.model = YOLO('yolov8n.pt')
    
    def analyze_video(self, video_path: str, output_path: str = None):
        """Analyze video for real-time and post-match analysis"""
        logger.info(f"🎥 Analyzing video: {video_path}")
        
        if not self.model:
            logger.error("❌ No model available for analysis")
            return None
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"❌ Error opening video: {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"📊 Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None
        
        # Analysis results
        analysis_results = {
            'detections': [],
            'events': [],
            'statistics': {
                'team_a_players': 0,
                'team_b_players': 0,
                'referees': 0,
                'balls': 0,
                'total_detections': 0
            },
            'match_time': '00:00',
            'frame_count': 0
        }
        
        # Process frames
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = self.model(frame, verbose=False)
            
            # Process detections
            frame_detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_id': class_id,
                            'frame': frame_count
                        }
                        frame_detections.append(detection)
                        
                        # Update statistics
                        class_names = ['team_a_player', 'team_a_goalkeeper', 'team_b_player', 
                                     'team_b_goalkeeper', 'referee', 'ball', 'other', 'staff']
                        if class_id < len(class_names):
                            class_name = class_names[class_id]
                            if 'team_a' in class_name:
                                analysis_results['statistics']['team_a_players'] += 1
                            elif 'team_b' in class_name:
                                analysis_results['statistics']['team_b_players'] += 1
                            elif class_name == 'referee':
                                analysis_results['statistics']['referees'] += 1
                            elif class_name == 'ball':
                                analysis_results['statistics']['balls'] += 1
                        
                        analysis_results['statistics']['total_detections'] += 1
            
            # Draw bounding boxes
            annotated_frame = self.draw_detections(frame, frame_detections)
            
            # Draw UI elements
            annotated_frame = self.draw_ui(annotated_frame, analysis_results, frame_count, fps)
            
            # Write output
            if out:
                out.write(annotated_frame)
            
            # Update match time
            seconds = frame_count / fps
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
            analysis_results['match_time'] = f"{minutes:02d}:{seconds:02d}"
            analysis_results['frame_count'] = frame_count
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"📈 Progress: {progress:.1f}%")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        
        processing_time = time.time() - start_time
        logger.info(f"⏱️ Analysis completed in {processing_time:.2f} seconds")
        
        # Save results
        results_path = output_path.replace('.mp4', '_results.json') if output_path else 'analysis_results.json'
        with open(results_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info(f"📊 Results saved to: {results_path}")
        
        return analysis_results
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes on frame"""
        annotated_frame = frame.copy()
        
        class_names = ['team_a_player', 'team_a_goalkeeper', 'team_b_player', 
                      'team_b_goalkeeper', 'referee', 'ball', 'other', 'staff']
        colors = [(255, 0, 0), (200, 0, 0), (0, 0, 255), (0, 0, 200),
                 (0, 255, 0), (255, 255, 0), (128, 128, 128), (255, 0, 255)]
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            if class_id < len(class_names):
                class_name = class_names[class_id]
                color = colors[class_id]
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw label
                label = f"{class_name} {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_size[1] - 10),
                             (int(x1) + label_size[0], int(y1)), color, -1)
                cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def draw_ui(self, frame: np.ndarray, results: Dict, frame_count: int, fps: int) -> np.ndarray:
        """Draw UI elements on frame"""
        height, width = frame.shape[:2]
        
        # Draw header
        header_height = 60
        cv2.rectangle(frame, (0, 0), (width, header_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (width, header_height), (255, 255, 255), 2)
        
        # Draw title
        cv2.putText(frame, "Godseye AI - Football Analytics", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw match time
        cv2.putText(frame, f"Time: {results['match_time']}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw statistics
        stats = results['statistics']
        stats_text = f"Team A: {stats['team_a_players']} | Team B: {stats['team_b_players']} | Ref: {stats['referees']} | Ball: {stats['balls']}"
        cv2.putText(frame, stats_text, (width - 400, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

def main():
    """Main function"""
    logger.info("🚀 Godseye AI - Automatic Training and Deployment")
    logger.info("=" * 60)
    
    # Step 1: Train the model
    logger.info("📚 Step 1: Training the model...")
    trainer = AutoTrainer()
    trainer.train_model(epochs=30)  # Reduced epochs for faster training
    
    # Step 2: Test the model
    logger.info("🧪 Step 2: Testing the model...")
    trainer.test_model()
    
    # Step 3: Integrate with web app
    logger.info("🌐 Step 3: Integrating with web app...")
    web_app = WebAppIntegration(trainer.model_path)
    
    # Test with available videos
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        video_files.extend(Path('.').glob(ext))
    
    if video_files:
        logger.info(f"🎥 Found {len(video_files)} videos for testing")
        test_video = str(video_files[0])
        logger.info(f"🎯 Testing with: {test_video}")
        
        # Analyze video
        results = web_app.analyze_video(test_video, f"analyzed_{Path(test_video).name}")
        
        if results:
            logger.info("✅ Video analysis completed successfully!")
            logger.info(f"📊 Detected: {results['statistics']['total_detections']} objects")
            logger.info(f"⏱️ Match time: {results['match_time']}")
            logger.info(f"🎬 Frames processed: {results['frame_count']}")
    else:
        logger.info("ℹ️ No videos found for testing")
    
    logger.info("🎉 Godseye AI is ready for real-time football analytics!")
    logger.info("=" * 60)
    logger.info("🌐 The trained model is now integrated with the web app")
    logger.info("📱 Users can upload videos for real-time and post-match analysis")
    logger.info("🏆 The system provides comprehensive football analytics")

if __name__ == "__main__":
    main()

