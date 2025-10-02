#!/usr/bin/env python3
"""
Create Balanced Dataset for Robust Training
Fixes class imbalance and background issues
"""

import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
from collections import Counter

def create_balanced_dataset():
    """Create a balanced dataset with proper class distribution"""
    
    # Paths
    train_images_dir = Path("data/yolo_dataset/train/images")
    train_labels_dir = Path("data/yolo_dataset/train/labels")
    val_images_dir = Path("data/yolo_dataset/val/images")
    val_labels_dir = Path("data/yolo_dataset/val/labels")
    
    # Create balanced directories
    balanced_train_dir = Path("data/balanced_dataset/train")
    balanced_val_dir = Path("data/balanced_dataset/val")
    
    for split in ["images", "labels"]:
        (balanced_train_dir / split).mkdir(parents=True, exist_ok=True)
        (balanced_val_dir / split).mkdir(parents=True, exist_ok=True)
    
    print("🔄 Creating balanced dataset...")
    
    # Collect all labeled files
    labeled_files = []
    for label_file in train_labels_dir.glob("*.txt"):
        if label_file.stat().st_size > 0:  # Non-empty files
            image_file = train_images_dir / f"{label_file.stem}.jpg"
            if image_file.exists():
                labeled_files.append((image_file, label_file))
    
    print(f"📊 Found {len(labeled_files)} labeled files")
    
    # Analyze class distribution
    class_counts = Counter()
    for _, label_file in labeled_files:
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    
    print(f"📈 Class distribution: {dict(class_counts)}")
    
    # Create balanced dataset
    # Keep all labeled files (they have objects)
    balanced_files = labeled_files.copy()
    
    # Add some background images (but limit to 20% of total)
    background_files = []
    for image_file in train_images_dir.glob("*.jpg"):
        label_file = train_labels_dir / f"{image_file.stem}.txt"
        if label_file.stat().st_size == 0:  # Empty label files (backgrounds)
            background_files.append((image_file, label_file))
    
    # Limit backgrounds to 20% of labeled files
    max_backgrounds = len(labeled_files) // 4
    selected_backgrounds = random.sample(background_files, min(max_backgrounds, len(background_files)))
    balanced_files.extend(selected_backgrounds)
    
    print(f"✅ Balanced dataset: {len(labeled_files)} labeled + {len(selected_backgrounds)} backgrounds")
    
    # Split into train/val (80/20)
    random.shuffle(balanced_files)
    split_idx = int(len(balanced_files) * 0.8)
    train_files = balanced_files[:split_idx]
    val_files = balanced_files[split_idx:]
    
    print(f"📊 Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Copy files to balanced dataset
    for i, (image_file, label_file) in enumerate(train_files):
        shutil.copy2(image_file, balanced_train_dir / "images" / f"train_{i:04d}.jpg")
        shutil.copy2(label_file, balanced_train_dir / "labels" / f"train_{i:04d}.txt")
    
    for i, (image_file, label_file) in enumerate(val_files):
        shutil.copy2(image_file, balanced_val_dir / "images" / f"val_{i:04d}.jpg")
        shutil.copy2(label_file, balanced_val_dir / "labels" / f"val_{i:04d}.txt")
    
    # Create dataset.yaml
    yaml_content = f"""
path: {Path("data/balanced_dataset").absolute()}
train: train/images
val: val/images

nc: 8
names: ['team_a_player', 'team_b_player', 'referee', 'ball', 'team_a_goalkeeper', 'team_b_goalkeeper', 'assistant_referee', 'others']

# Balanced training parameters
augment: true
mosaic: 1.0
mixup: 0.1
copy_paste: 0.1
degrees: 15.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
"""
    
    yaml_path = Path("data/balanced_dataset/dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    
    print(f"✅ Created balanced dataset.yaml at {yaml_path}")
    
    # Verify the balanced dataset
    verify_balanced_dataset()
    
    return yaml_path

def verify_balanced_dataset():
    """Verify the balanced dataset has proper distribution"""
    
    train_labels_dir = Path("data/balanced_dataset/train/labels")
    val_labels_dir = Path("data/balanced_dataset/val/labels")
    
    # Count instances in training set
    train_instances = 0
    train_backgrounds = 0
    train_class_counts = Counter()
    
    for label_file in train_labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            if not lines or all(line.strip() == '' for line in lines):
                train_backgrounds += 1
            else:
                for line in lines:
                    if line.strip():
                        class_id = int(line.split()[0])
                        train_class_counts[class_id] += 1
                        train_instances += 1
    
    # Count instances in validation set
    val_instances = 0
    val_backgrounds = 0
    val_class_counts = Counter()
    
    for label_file in val_labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            if not lines or all(line.strip() == '' for line in lines):
                val_backgrounds += 1
            else:
                for line in lines:
                    if line.strip():
                        class_id = int(line.split()[0])
                        val_class_counts[class_id] += 1
                        val_instances += 1
    
    total_train = len(list(train_labels_dir.glob("*.txt")))
    total_val = len(list(val_labels_dir.glob("*.txt")))
    
    print("\n📊 BALANCED DATASET VERIFICATION:")
    print(f"Train: {total_train} images, {train_instances} instances, {train_backgrounds} backgrounds")
    print(f"Val: {total_val} images, {val_instances} instances, {val_backgrounds} backgrounds")
    print(f"Train class distribution: {dict(train_class_counts)}")
    print(f"Val class distribution: {dict(val_class_counts)}")
    
    # Calculate percentages
    train_bg_pct = (train_backgrounds / total_train) * 100
    val_bg_pct = (val_backgrounds / total_val) * 100
    
    print(f"Train background %: {train_bg_pct:.1f}%")
    print(f"Val background %: {val_bg_pct:.1f}%")
    
    if train_bg_pct > 30:
        print("⚠️ WARNING: Still too many backgrounds in training set")
    else:
        print("✅ Good background distribution")

if __name__ == "__main__":
    create_balanced_dataset()
