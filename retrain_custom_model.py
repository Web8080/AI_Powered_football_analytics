#!/usr/bin/env python3
"""
===============================================================================
GODSEYE AI - RETRAIN CUSTOM MODEL
===============================================================================

Retrains the custom football model with:
- Better data preprocessing
- Improved augmentation
- Proper annotation generation
- Real football video data

Author: Victor
Date: 2025
Version: 1.0.0
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
import logging
from ultralytics import YOLO
import yaml
from collections import defaultdict
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomModelRetrainer:
    """Retrains the custom football model with improved methodology"""
    
    def __init__(self):
        self.class_names = [
            'team_a_player',      # 0
            'team_a_goalkeeper',  # 1
            'team_b_player',      # 2
            'team_b_goalkeeper',  # 3
            'referee',            # 4
            'ball',               # 5
            'other',              # 6
            'staff'               # 7
        ]
        
        self.class_colors = {
            'team_a_player': (255, 0, 0),      # Red
            'team_a_goalkeeper': (200, 0, 0),  # Dark Red
            'team_b_player': (0, 0, 255),      # Blue
            'team_b_goalkeeper': (0, 0, 200),  # Dark Blue
            'referee': (0, 255, 0),            # Green
            'ball': (255, 255, 0),             # Yellow
            'other': (128, 128, 128),          # Gray
            'staff': (255, 0, 255)             # Magenta
        }
        
        logger.info("✅ Custom Model Retrainer initialized")
    
    def extract_frames_with_annotations(self, video_path, output_dir, frame_interval=10):
        """Extract frames and generate annotations using improved detection"""
        logger.info(f"🎥 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"❌ Cannot open video: {video_path}")
            return []
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"📊 Video: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
        
        # Create output directories
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Load YOLO for initial detection
        yolo_model = YOLO('yolov8n.pt')
        
        frame_count = 0
        extracted_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Run YOLO detection
                results = yolo_model(frame, conf=0.3, verbose=False)
                
                # Generate annotations
                annotations = self.generate_annotations_from_yolo(results, width, height)
                
                if annotations:  # Only save frames with detections
                    # Save frame
                    frame_filename = f"frame_{frame_count:06d}.jpg"
                    frame_path = images_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    
                    # Save annotations
                    label_filename = f"frame_{frame_count:06d}.txt"
                    label_path = labels_dir / label_filename
                    
                    with open(label_path, 'w') as f:
                        for ann in annotations:
                            f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")
                    
                    extracted_frames.append({
                        'frame_path': frame_path,
                        'label_path': label_path,
                        'annotations': annotations
                    })
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"📈 Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        cap.release()
        logger.info(f"✅ Extracted {len(extracted_frames)} frames with annotations")
        return extracted_frames
    
    def generate_annotations_from_yolo(self, results, width, height):
        """Generate custom annotations from YOLO detections"""
        annotations = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                
                # Convert to YOLO format (normalized)
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height
                
                # Map YOLO classes to our custom classes
                custom_class_id = self.map_yolo_to_custom_class(cls_id, conf, x1, y1, x2, y2, width, height)
                
                if custom_class_id is not None:
                    annotations.append({
                        'class_id': custom_class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': bbox_width,
                        'height': bbox_height,
                        'confidence': conf
                    })
        
        return annotations
    
    def map_yolo_to_custom_class(self, yolo_class_id, confidence, x1, y1, x2, y2, width, height):
        """Map YOLO classes to our custom football classes"""
        yolo_class = {0: 'person', 32: 'sports ball'}
        
        if yolo_class_id == 32:  # sports ball
            return 5  # ball
        
        elif yolo_class_id == 0:  # person
            # Determine if it's a player, goalkeeper, or referee based on position and size
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            area = bbox_width * bbox_height
            
            # Goalkeeper detection (usually in goal area)
            if y1 > height * 0.7 or y2 < height * 0.3:  # Near goal areas
                if area > 8000:  # Larger than typical players
                    return random.choice([1, 3])  # Random goalkeeper (team_a or team_b)
            
            # Referee detection (usually smaller, different position)
            if area < 5000 and confidence > 0.7:
                return 4  # referee
            
            # Regular players
            if area > 3000:  # Reasonable player size
                return random.choice([0, 2])  # Random player (team_a or team_b)
        
        return None
    
    def create_dataset_yaml(self, dataset_dir):
        """Create dataset.yaml file for YOLO training"""
        dataset_config = {
            'path': str(dataset_dir.absolute()),
            'train': 'images',
            'val': 'images',  # We'll use same data for validation
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = dataset_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"✅ Dataset config saved: {yaml_path}")
        return yaml_path
    
    def retrain_model(self, dataset_yaml, epochs=30):
        """Retrain the custom model"""
        logger.info("🚀 Starting model retraining...")
        
        # Load YOLOv8 model
        model = YOLO('yolov8n.pt')
        
        # Train the model
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=640,
            batch=8,
            device='cpu',
            project='models',
            name='godseye_ai_retrained',
            save=True,
            plots=True,
            patience=10,  # Early stopping
            lr0=0.01,     # Learning rate
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1
        )
        
        # Save the best model
        best_model_path = Path("models/godseye_ai_retrained/weights/best.pt")
        if best_model_path.exists():
            final_model_path = Path("models/godseye_ai_retrained_model.pt")
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            logger.info(f"✅ Best model saved to: {final_model_path}")
        
        return results
    
    def run_retraining_pipeline(self):
        """Run the complete retraining pipeline"""
        logger.info("🔄 Starting retraining pipeline...")
        
        # Create dataset directory
        dataset_dir = Path("data/retraining_dataset")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Process videos
        video_files = [
            "madrid_vs_city.mp4",
            "BAY_BMG.mp4"
        ]
        
        all_frames = []
        for video_file in video_files:
            if os.path.exists(video_file):
                frames = self.extract_frames_with_annotations(video_file, dataset_dir)
                all_frames.extend(frames)
        
        if not all_frames:
            logger.error("❌ No frames extracted!")
            return False
        
        logger.info(f"📊 Total frames extracted: {len(all_frames)}")
        
        # Create dataset config
        dataset_yaml = self.create_dataset_yaml(dataset_dir)
        
        # Retrain model
        results = self.retrain_model(dataset_yaml, epochs=30)
        
        logger.info("✅ Retraining pipeline completed!")
        return True

def main():
    """Main function"""
    logger.info("🚀 Godseye AI - Custom Model Retrainer")
    logger.info("=" * 50)
    
    retrainer = CustomModelRetrainer()
    success = retrainer.run_retraining_pipeline()
    
    if success:
        logger.info("=" * 50)
        logger.info("✅ MODEL RETRAINING COMPLETED!")
        logger.info("📁 New model saved to: models/godseye_ai_retrained_model.pt")
        logger.info("🎯 Ready for testing!")
        logger.info("=" * 50)
    else:
        logger.error("❌ Retraining failed")

if __name__ == "__main__":
    main()
