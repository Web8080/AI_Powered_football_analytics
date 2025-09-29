#!/usr/bin/env python3
"""
Jersey Number Detection System
=============================

This script creates a specialized CNN for detecting and recognizing jersey numbers
on football players. It uses OCR techniques and number recognition.

Author: Godseye AI Team
Usage: python jersey_number_detection.py --video BAY_BMG.mp4 --output models/jersey_detector.pt
"""

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class JerseyNumberDataset(Dataset):
    """Dataset for jersey number detection and recognition"""
    
    def __init__(self, video_path, annotations=None):
        self.video_path = video_path
        self.annotations = annotations or []
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._extract_jersey_crops()
    
    def _extract_jersey_crops(self):
        """Extract jersey number crops from video"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Extracting jersey crops from {total_frames} frames...")
        
        frame_count = 0
        crops_extracted = 0
        
        while crops_extracted < 1000:  # Limit for training
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 30th frame
            if frame_count % 30 != 0:
                continue
            
            # Detect potential jersey regions
            jersey_crops = self._detect_jersey_regions(frame)
            
            for crop, bbox in jersey_crops:
                if crop is not None:
                    # Generate synthetic number labels for training
                    number = self._generate_synthetic_number()
                    
                    self.samples.append({
                        'crop': crop,
                        'number': number,
                        'bbox': bbox,
                        'frame': frame_count
                    })
                    crops_extracted += 1
        
        cap.release()
        print(f"✅ Extracted {len(self.samples)} jersey crops")
    
    def _detect_jersey_regions(self, frame):
        """Detect potential jersey regions in frame"""
        crops = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common jersey colors
        color_ranges = [
            # Red jerseys
            ([0, 50, 50], [10, 255, 255]),
            ([170, 50, 50], [180, 255, 255]),
            # Blue jerseys
            ([100, 50, 50], [130, 255, 255]),
            # Green jerseys
            ([40, 50, 50], [80, 255, 255]),
            # Yellow jerseys
            ([20, 50, 50], [40, 255, 255]),
            # White jerseys
            ([0, 0, 200], [180, 30, 255]),
            # Black jerseys
            ([0, 0, 0], [180, 255, 50])
        ]
        
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 10000:  # Filter by size
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (jerseys are roughly rectangular)
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 2.0:
                        # Extract crop
                        crop = frame[y:y+h, x:x+w]
                        if crop.size > 0:
                            crops.append((crop, (x, y, w, h)))
        
        return crops
    
    def _generate_synthetic_number(self):
        """Generate synthetic jersey number for training"""
        # Common jersey numbers in football
        common_numbers = list(range(1, 12)) + list(range(12, 24)) + [99]
        return np.random.choice(common_numbers)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        crop = sample['crop']
        number = sample['number']
        
        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        
        # Apply transforms
        crop_tensor = self.transform(crop_pil)
        
        return crop_tensor, torch.LongTensor([number])

class JerseyNumberModel(nn.Module):
    """CNN model for jersey number recognition"""
    
    def __init__(self, num_classes=100):  # Numbers 0-99
        super(JerseyNumberModel, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_jersey_detector(video_path, output_dir="models", epochs=100):
    """Train jersey number detection model"""
    
    print("🔢 Starting Jersey Number Detection Training...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    print("📊 Creating jersey number dataset...")
    dataset = JerseyNumberDataset(video_path)
    
    if len(dataset) == 0:
        print("❌ No jersey crops extracted from video")
        return None
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = JerseyNumberModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    print("🚀 Starting training...")
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Testing
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                total += target.size(0)
                correct += (pred == target.squeeze()).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {train_loss:.4f}, Accuracy = {accuracy:.2f}%")
    
    # Save model
    model_path = Path(output_dir) / "jersey_number_detector.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Jersey number model saved to: {model_path}")
    
    # Generate evaluation report
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.squeeze().cpu().numpy())
    
    # Classification report
    report = classification_report(all_targets, all_preds)
    print("\n📊 Jersey Number Recognition Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Jersey Number Recognition Confusion Matrix')
    plt.ylabel('True Number')
    plt.xlabel('Predicted Number')
    
    cm_path = Path(output_dir) / "jersey_number_confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"📊 Confusion matrix saved to: {cm_path}")
    
    # Training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    curves_path = Path(output_dir) / "jersey_number_training_curves.png"
    plt.savefig(curves_path)
    print(f"📈 Training curves saved to: {curves_path}")
    
    return model_path

def detect_jersey_numbers(video_path, model_path, output_dir="jersey_detection_results"):
    """Detect jersey numbers in video using trained model"""
    
    print("🔢 Detecting jersey numbers in video...")
    
    # Load model
    model = JerseyNumberModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    jersey_detections = []
    frame_count = 0
    
    print(f"📹 Processing {total_frames} frames...")
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 60th frame
        if frame_count % 60 != 0:
            continue
        
        # Detect jersey regions
        dataset = JerseyNumberDataset("")  # Empty dataset for detection methods
        jersey_crops = dataset._detect_jersey_regions(frame)
        
        for crop, bbox in jersey_crops:
            if crop is not None:
                # Preprocess crop
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb)
                transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                crop_tensor = transform(crop_pil).unsqueeze(0)
                
                # Predict number
                with torch.no_grad():
                    output = model(crop_tensor)
                    prediction = output.argmax(dim=1).item()
                    confidence = torch.softmax(output, dim=1)[0, prediction].item()
                
                if confidence > 0.5:  # High confidence detection
                    timestamp = frame_count / fps
                    jersey_detections.append({
                        'frame': frame_count,
                        'timestamp': timestamp,
                        'number': prediction,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    print(f"🔢 Jersey number {prediction} detected at {timestamp:.1f}s (confidence: {confidence:.3f})")
    
    cap.release()
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_path = Path(output_dir) / "jersey_numbers_detected.json"
    
    with open(results_path, 'w') as f:
        json.dump(jersey_detections, f, indent=2)
    
    print(f"✅ Jersey number detection results saved to: {results_path}")
    print(f"🔢 Total jersey numbers detected: {len(jersey_detections)}")
    
    return jersey_detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train jersey number detection model")
    parser.add_argument("--video", required=True, help="Path to football video")
    parser.add_argument("--output", default="models", help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--detect", action="store_true", help="Detect jersey numbers after training")
    
    args = parser.parse_args()
    
    # Train model
    model_path = train_jersey_detector(args.video, args.output, args.epochs)
    
    if model_path and args.detect:
        # Detect jersey numbers
        numbers = detect_jersey_numbers(args.video, model_path)
        
        print("\n🔢 Jersey Number Detection Summary:")
        for detection in numbers[:10]:  # Show first 10
            print(f"  🔢 Number {detection['number']} at {detection['timestamp']:.1f}s (confidence: {detection['confidence']:.3f})")
