#!/usr/bin/env python3
"""
Goal Detection Training Script
=============================

This script trains a specialized model for goal detection in football videos.
It uses temporal analysis and event detection to identify when goals are scored.

Author: Godseye AI Team
Usage: python train_goal_detection.py --video BAY_BMG.mp4 --output models/goal_detector.pt
"""

import os
import sys
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from ultralytics import YOLO
    print("✅ Ultralytics YOLO imported successfully")
except ImportError:
    print("❌ Ultralytics not installed. Installing...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO

class GoalDetectionDataset(Dataset):
    """Dataset for goal detection training"""
    
    def __init__(self, video_path, annotations=None, sequence_length=30):
        self.video_path = video_path
        self.sequence_length = sequence_length
        self.annotations = annotations or []
        self.sequences = []
        self.labels = []
        
        self._extract_sequences()
    
    def _extract_sequences(self):
        """Extract sequences from video for goal detection"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Extracting sequences from {total_frames} frames at {fps} FPS")
        
        # Create goal annotations based on common goal indicators
        goal_sequences = self._detect_goal_sequences(cap, fps)
        
        # Extract positive and negative sequences
        positive_sequences = []
        negative_sequences = []
        
        for goal_time in goal_sequences:
            start_frame = int(goal_time * fps)
            end_frame = min(start_frame + self.sequence_length, total_frames)
            
            if end_frame - start_frame >= self.sequence_length:
                sequence = self._extract_frame_sequence(cap, start_frame, end_frame)
                if sequence is not None:
                    positive_sequences.append(sequence)
        
        # Extract negative sequences (non-goal moments)
        for i in range(0, total_frames - self.sequence_length, self.sequence_length * 2):
            # Skip if this sequence overlaps with a goal
            is_goal_sequence = any(
                abs(i - int(goal_time * fps)) < self.sequence_length 
                for goal_time in goal_sequences
            )
            
            if not is_goal_sequence:
                sequence = self._extract_frame_sequence(cap, i, i + self.sequence_length)
                if sequence is not None:
                    negative_sequences.append(sequence)
        
        cap.release()
        
        # Combine and label sequences
        self.sequences = positive_sequences + negative_sequences
        self.labels = [1] * len(positive_sequences) + [0] * len(negative_sequences)
        
        print(f"✅ Extracted {len(positive_sequences)} goal sequences and {len(negative_sequences)} non-goal sequences")
    
    def _detect_goal_sequences(self, cap, fps):
        """Detect potential goal sequences using heuristics"""
        goal_times = []
        
        # Method 1: Detect sudden crowd noise/celebration (simplified)
        # Method 2: Detect ball movement towards goal
        # Method 3: Detect player celebration patterns
        
        # For now, use manual annotations or simple heuristics
        # In a real system, you would use audio analysis, crowd detection, etc.
        
        # Example: Detect goals at specific timestamps (you can modify these)
        # These are example timestamps where goals might occur
        potential_goals = [
            1200,  # 20 minutes
            2400,  # 40 minutes  
            3600,  # 60 minutes
            4800,  # 80 minutes
        ]
        
        for timestamp in potential_goals:
            if timestamp < cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps:
                goal_times.append(timestamp)
        
        return goal_times
    
    def _extract_frame_sequence(self, cap, start_frame, end_frame):
        """Extract a sequence of frames"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for i in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                return None
            
            # Resize frame for consistency
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        
        return np.array(frames)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert to tensor and normalize
        sequence = torch.FloatTensor(sequence).permute(0, 3, 1, 2) / 255.0
        
        return sequence, torch.LongTensor([label])

class GoalDetectionModel(nn.Module):
    """CNN-LSTM model for goal detection"""
    
    def __init__(self, sequence_length=30, num_classes=2):
        super(GoalDetectionModel, self).__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # LSTM for temporal analysis
        self.lstm = nn.LSTM(128 * 7 * 7, 256, batch_first=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        
        # Reshape for CNN processing
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Extract features with CNN
        features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(features)
        
        # Use last output for classification
        output = self.classifier(lstm_out[:, -1, :])
        
        return output

def train_goal_detector(video_path, output_dir="models", epochs=50):
    """Train goal detection model"""
    
    print("🎯 Starting Goal Detection Training...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    print("📊 Creating dataset...")
    dataset = GoalDetectionDataset(video_path)
    
    if len(dataset) == 0:
        print("❌ No sequences extracted from video")
        return None
    
    # Split dataset
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(
        dataset.sequences, dataset.labels, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_sequences), 
        torch.LongTensor(train_labels)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_sequences), 
        torch.LongTensor(test_labels)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Initialize model
    model = GoalDetectionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
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
            loss = criterion(output, target)
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
                correct += (pred == target).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {train_loss:.4f}, Accuracy = {accuracy:.2f}%")
    
    # Save model
    model_path = Path(output_dir) / "goal_detector.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Generate evaluation report
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Classification report
    report = classification_report(all_targets, all_preds, target_names=['No Goal', 'Goal'])
    print("\n📊 Classification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Goal', 'Goal'], 
                yticklabels=['No Goal', 'Goal'])
    plt.title('Goal Detection Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = Path(output_dir) / "goal_detection_confusion_matrix.png"
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
    
    curves_path = Path(output_dir) / "goal_detection_training_curves.png"
    plt.savefig(curves_path)
    print(f"📈 Training curves saved to: {curves_path}")
    
    return model_path

def detect_goals_in_video(video_path, model_path, output_dir="goal_detection_results"):
    """Detect goals in a video using trained model"""
    
    print("🎯 Detecting goals in video...")
    
    # Load model
    model = GoalDetectionModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    goals_detected = []
    sequence_length = 30
    
    print(f"📹 Processing {total_frames} frames...")
    
    for i in range(0, total_frames - sequence_length, sequence_length):
        # Extract sequence
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        frames = []
        
        for j in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        
        if len(frames) == sequence_length:
            # Convert to tensor
            sequence = torch.FloatTensor(frames).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
            
            # Predict
            with torch.no_grad():
                output = model(sequence)
                prediction = output.argmax(dim=1).item()
                confidence = torch.softmax(output, dim=1)[0, 1].item()
            
            if prediction == 1 and confidence > 0.7:  # Goal detected with high confidence
                timestamp = i / fps
                goals_detected.append({
                    'timestamp': timestamp,
                    'frame': i,
                    'confidence': confidence
                })
                print(f"⚽ Goal detected at {timestamp:.1f}s (confidence: {confidence:.3f})")
    
    cap.release()
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_path = Path(output_dir) / "goals_detected.json"
    
    with open(results_path, 'w') as f:
        json.dump(goals_detected, f, indent=2)
    
    print(f"✅ Goal detection results saved to: {results_path}")
    print(f"🎯 Total goals detected: {len(goals_detected)}")
    
    return goals_detected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train goal detection model")
    parser.add_argument("--video", required=True, help="Path to football video")
    parser.add_argument("--output", default="models", help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--detect", action="store_true", help="Detect goals in video after training")
    
    args = parser.parse_args()
    
    # Train model
    model_path = train_goal_detector(args.video, args.output, args.epochs)
    
    if model_path and args.detect:
        # Detect goals
        goals = detect_goals_in_video(args.video, model_path)
        
        print("\n🎯 Goal Detection Summary:")
        for goal in goals:
            print(f"  ⚽ Goal at {goal['timestamp']:.1f}s (confidence: {goal['confidence']:.3f})")
