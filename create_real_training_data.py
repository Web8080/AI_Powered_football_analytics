#!/usr/bin/env python3
"""
================================================================================
GODSEYE AI - REAL TRAINING DATA CREATOR
================================================================================

Author: Victor Ibhafidon
Date: January 28, 2025
Version: 1.0.0

DESCRIPTION:
Creates training data from real football videos by extracting frames
and generating annotations. This addresses the gap between synthetic
and real football footage.

USAGE:
    python create_real_training_data.py
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import random

class RealTrainingDataCreator:
    """Creates training data from real football videos"""
    
    def __init__(self, video_path: str, output_dir: str = "data/real_football"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Class mapping (simplified for real data)
        self.classes = {
            'player': 0,
            'ball': 1,
            'referee': 2,
            'other': 3
        }
        
    def extract_frames(self, max_frames: int = 100):
        """Extract frames from the real video"""
        print(f"🎥 Extracting frames from {self.video_path}")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📊 Video info: {total_frames} frames, {fps} FPS")
        
        # Extract frames at regular intervals
        frame_interval = max(1, total_frames // max_frames)
        extracted_frames = 0
        
        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Save frame
            frame_path = self.images_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            
            # Create corresponding label file
            self.create_annotations_for_frame(frame, frame_idx)
            
            extracted_frames += 1
            
            if extracted_frames >= max_frames:
                break
        
        cap.release()
        print(f"✅ Extracted {extracted_frames} frames")
        return extracted_frames
    
    def create_annotations_for_frame(self, frame: np.ndarray, frame_idx: int):
        """Create annotations for a frame using computer vision techniques"""
        height, width = frame.shape[:2]
        label_lines = []
        
        # Use computer vision to detect potential players and objects
        detections = self.detect_objects_cv(frame)
        
        for detection in detections:
            class_id, bbox, confidence = detection
            
            # Convert to YOLO format
            x, y, w, h = bbox
            center_x = (x + w/2) / width
            center_y = (y + h/2) / height
            norm_width = w / width
            norm_height = h / height
            
            label_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
        
        # Save labels
        label_path = self.labels_dir / f"frame_{frame_idx:06d}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))
    
    def detect_objects_cv(self, frame: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Use computer vision techniques to detect objects in the frame"""
        detections = []
        
        # Convert to different color spaces for better detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect potential players using color segmentation
        player_detections = self.detect_players_by_color(frame, hsv)
        detections.extend(player_detections)
        
        # Detect ball using circular Hough transform
        ball_detections = self.detect_ball(frame, gray)
        detections.extend(ball_detections)
        
        # Detect referee (yellow shirt detection)
        referee_detections = self.detect_referee(frame, hsv)
        detections.extend(referee_detections)
        
        return detections
    
    def detect_players_by_color(self, frame: np.ndarray, hsv: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Detect players based on team colors"""
        detections = []
        
        # Define color ranges for different teams (adjust based on your video)
        # Team A (typically white/light colors)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Team B (typically dark colors)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 100])
        
        # Create masks
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Find contours
        for mask, class_id in [(mask_white, 0), (mask_dark, 0)]:  # Both are players
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter small objects
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter by aspect ratio (players are roughly rectangular)
                    aspect_ratio = h / w
                    if 1.5 < aspect_ratio < 4.0:  # Typical player aspect ratio
                        detections.append((class_id, (x, y, w, h), 0.8))
        
        return detections
    
    def detect_ball(self, frame: np.ndarray, gray: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Detect ball using circular Hough transform"""
        detections = []
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=5, maxRadius=25
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Create bounding box around circle
                bbox = (x - r, y - r, 2 * r, 2 * r)
                detections.append((1, bbox, 0.7))  # class_id 1 for ball
        
        return detections
    
    def detect_referee(self, frame: np.ndarray, hsv: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Detect referee by yellow shirt color"""
        detections = []
        
        # Yellow color range for referee shirts
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio
                aspect_ratio = h / w
                if 1.5 < aspect_ratio < 4.0:
                    detections.append((2, (x, y, w, h), 0.6))  # class_id 2 for referee
        
        return detections
    
    def create_dataset_config(self):
        """Create YOLO dataset configuration"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images',
            'val': 'images',
            'nc': len(self.classes),
            'names': list(self.classes.keys())
        }
        
        config_path = self.output_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Created dataset config: {config_path}")
        return config_path

def main():
    """Main function to create real training data"""
    print("🏈 Godseye AI - Real Training Data Creator")
    print("=" * 50)
    
    # Use the Madrid vs City video
    video_path = "madrid_vs_city.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    # Create training data
    creator = RealTrainingDataCreator(video_path)
    
    # Extract frames and create annotations
    frames_extracted = creator.extract_frames(max_frames=50)  # Limit for quick training
    
    # Create dataset configuration
    config_path = creator.create_dataset_config()
    
    print(f"\n🎉 Real training data created successfully!")
    print(f"📁 Output directory: {creator.output_dir}")
    print(f"🖼️ Frames extracted: {frames_extracted}")
    print(f"⚙️ Dataset config: {config_path}")
    
    return creator.output_dir

if __name__ == "__main__":
    main()
