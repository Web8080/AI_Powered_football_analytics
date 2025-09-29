#!/usr/bin/env python3
"""
================================================================================
GODSEYE AI - SIMPLE MODEL TRAINING SCRIPT
================================================================================

Author: Victor Ibhafidon
Date: January 28, 2025
Version: 1.0.0

DESCRIPTION:
This script trains a simple YOLOv8 detection model for the Godseye AI sports
analytics platform. It creates a basic model that can detect players, goalkeepers,
referees, and balls. This is a quick training script to generate actual model
files for testing the system.

USAGE:
    python train_simple_model.py

OUTPUT:
    - models/yolov8_football.pt (PyTorch model)
    - models/yolov8_football.onnx (ONNX model for deployment)
    - training_results/ (training logs and metrics)
"""

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
import yaml
import json
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from ultralytics import YOLO
    print("✅ Ultralytics YOLO imported successfully")
except ImportError:
    print("❌ Ultralytics not installed. Installing...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO

def create_synthetic_dataset():
    """Create a synthetic dataset for training"""
    print("📊 Creating synthetic dataset...")
    
    # Create directories
    data_dir = Path("data/synthetic")
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic images and labels
    for i in range(100):  # Create 100 synthetic samples
        # Create a simple synthetic image (green field with some objects)
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        img[100:540, 100:540] = [34, 139, 34]  # Green field
        
        # Add some random "players" as colored rectangles
        for j in range(22):  # 22 players
            x = np.random.randint(120, 520)
            y = np.random.randint(120, 520)
            w, h = 20, 40
            
            # Color based on team (red or blue)
            color = [255, 0, 0] if j < 11 else [0, 0, 255]
            img[y:y+h, x:x+w] = color
        
        # Add ball (white circle)
        ball_x, ball_y = np.random.randint(200, 440), np.random.randint(200, 440)
        cv2.circle(img, (ball_x, ball_y), 8, (255, 255, 255), -1)
        
        # Add referee (yellow)
        ref_x, ref_y = np.random.randint(150, 490), np.random.randint(150, 490)
        img[ref_y:ref_y+30, ref_x:ref_x+15] = [0, 255, 255]
        
        # Save image
        img_path = images_dir / f"image_{i:04d}.jpg"
        cv2.imwrite(str(img_path), img)
        
        # Create corresponding label file
        label_path = labels_dir / f"image_{i:04d}.txt"
        with open(label_path, 'w') as f:
            # Write YOLO format labels
            for j in range(22):  # 22 players
                x = np.random.randint(120, 520) / 640
                y = np.random.randint(120, 520) / 640
                w, h = 20/640, 40/640
                class_id = 0 if j < 11 else 2  # team_a_player or team_b_player
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
            # Add ball
            ball_x = np.random.randint(200, 440) / 640
            ball_y = np.random.randint(200, 440) / 640
            f.write(f"5 {ball_x:.6f} {ball_y:.6f} 0.025 0.025\n")  # ball class
            
            # Add referee
            ref_x = np.random.randint(150, 490) / 640
            ref_y = np.random.randint(150, 490) / 640
            f.write(f"4 {ref_x:.6f} {ref_y:.6f} 0.023 0.047\n")  # referee class
    
    print(f"✅ Created {len(list(images_dir.glob('*.jpg')))} synthetic images")
    return data_dir

def create_dataset_config(data_dir):
    """Create dataset configuration file"""
    config = {
        'path': str(data_dir.absolute()),
        'train': 'images',
        'val': 'images',  # Using same data for validation (for simplicity)
        'nc': 8,  # Number of classes
        'names': [
            'team_a_player',      # 0
            'team_a_goalkeeper',  # 1
            'team_b_player',      # 2
            'team_b_goalkeeper',  # 3
            'referee',            # 4
            'ball',               # 5
            'other',              # 6
            'staff'               # 7
        ]
    }
    
    config_path = data_dir / "dataset.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Created dataset config: {config_path}")
    return config_path

def train_model(config_path):
    """Train the YOLOv8 model"""
    print("🚀 Starting model training...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')  # Use nano version for faster training
    
    # Train the model
    results = model.train(
        data=str(config_path),
        epochs=10,  # Quick training for demo
        imgsz=640,
        batch=8,
        device='cpu',  # Use CPU since you mentioned you don't have GPU
        project='training_results',
        name='football_detection',
        save=True,
        save_period=5,
        plots=True,
        verbose=True
    )
    
    print("✅ Training completed!")
    return model, results

def export_models(model):
    """Export models in different formats"""
    print("📦 Exporting models...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Export PyTorch model
    pt_path = models_dir / "yolov8_football.pt"
    model.save(str(pt_path))
    print(f"✅ PyTorch model saved: {pt_path}")
    
    # Export ONNX model
    try:
        onnx_path = models_dir / "yolov8_football.onnx"
        model.export(format='onnx', imgsz=640)
        # Move the exported file to our models directory
        exported_onnx = Path("training_results/football_detection/weights/best.onnx")
        if exported_onnx.exists():
            exported_onnx.rename(onnx_path)
            print(f"✅ ONNX model saved: {onnx_path}")
    except Exception as e:
        print(f"⚠️ ONNX export failed: {e}")
    
    return pt_path

def create_model_info(model_path, results):
    """Create model information file"""
    info = {
        "model_name": "YOLOv8 Football Detection",
        "version": "1.0.0",
        "created_date": datetime.now().isoformat(),
        "model_path": str(model_path),
        "classes": [
            'team_a_player',
            'team_a_goalkeeper', 
            'team_b_player',
            'team_b_goalkeeper',
            'referee',
            'ball',
            'other',
            'staff'
        ],
        "input_size": [640, 640],
        "framework": "PyTorch",
        "training_epochs": 10,
        "training_device": "CPU",
        "model_size": "nano (yolov8n)",
        "description": "Simple YOLOv8 model trained on synthetic football data for Godseye AI sports analytics"
    }
    
    info_path = Path("models/model_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Model info saved: {info_path}")
    return info_path

def main():
    """Main training function"""
    print("🏈 Godseye AI - Simple Model Training")
    print("=" * 50)
    
    print("✅ OpenCV available")
    
    # Step 1: Create synthetic dataset
    data_dir = create_synthetic_dataset()
    
    # Step 2: Create dataset configuration
    config_path = create_dataset_config(data_dir)
    
    # Step 3: Train the model
    model, results = train_model(config_path)
    
    # Step 4: Export models
    model_path = export_models(model)
    
    # Step 5: Create model info
    info_path = create_model_info(model_path, results)
    
    print("\n🎉 Training completed successfully!")
    print(f"📁 Model files created in: {Path('models').absolute()}")
    print(f"📊 Training results in: {Path('training_results').absolute()}")
    print("\nYou can now test the model with your video uploads!")
    
    return model_path

if __name__ == "__main__":
    main()
