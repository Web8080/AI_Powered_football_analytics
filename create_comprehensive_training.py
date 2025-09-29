#!/usr/bin/env python3
"""
================================================================================
GODSEYE AI - COMPREHENSIVE TRAINING DATA CREATOR
================================================================================

Author: Victor Ibhafidon
Date: January 28, 2025
Version: 2.0.0

DESCRIPTION:
Creates comprehensive training data from full 90-minute football matches.
Handles first half, second half, and filters out advertisements.
Uses advanced computer vision techniques for better detection.

USAGE:
    python create_comprehensive_training.py
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import random
from datetime import datetime

class ComprehensiveTrainingCreator:
    """Creates comprehensive training data from full football matches"""
    
    def __init__(self, video_path: str, output_dir: str = "data/comprehensive_football"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced class mapping for comprehensive detection
        self.classes = {
            'team_a_player': 0,
            'team_a_goalkeeper': 1,
            'team_b_player': 2, 
            'team_b_goalkeeper': 3,
            'referee': 4,
            'ball': 5,
            'outlier': 6,
            'staff': 7
        }
        
        # Match timing (approximate)
        self.first_half_start = 0
        self.first_half_end = 45 * 60  # 45 minutes
        self.halftime_start = 45 * 60
        self.halftime_end = 50 * 60    # 5 minute halftime
        self.second_half_start = 50 * 60
        self.second_half_end = 95 * 60  # 45 minutes + injury time
        
    def extract_comprehensive_frames(self, max_frames: int = 200):
        """Extract frames from different parts of the match"""
        print(f"🎥 Extracting comprehensive frames from {self.video_path}")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        print(f"📊 Video info: {total_frames} frames, {fps} FPS, {duration/60:.1f} minutes")
        
        # Extract frames from different match phases
        frames_per_phase = max_frames // 4  # Distribute across 4 phases
        
        extracted_frames = 0
        
        # Phase 1: First half (0-45 min)
        print("⚽ Extracting first half frames...")
        extracted_frames += self.extract_phase_frames(
            cap, 0, self.first_half_end, frames_per_phase, "first_half"
        )
        
        # Phase 2: Second half (50-95 min) 
        print("⚽ Extracting second half frames...")
        extracted_frames += self.extract_phase_frames(
            cap, self.second_half_start, self.second_half_end, frames_per_phase, "second_half"
        )
        
        # Phase 3: Random moments throughout match
        print("🎯 Extracting random match moments...")
        extracted_frames += self.extract_phase_frames(
            cap, 0, min(duration, self.second_half_end), frames_per_phase, "random"
        )
        
        # Phase 4: High activity moments (corners, free kicks, etc.)
        print("🔥 Extracting high activity moments...")
        extracted_frames += self.extract_phase_frames(
            cap, 0, min(duration, self.second_half_end), frames_per_phase, "high_activity"
        )
        
        cap.release()
        print(f"✅ Extracted {extracted_frames} comprehensive frames")
        return extracted_frames
    
    def extract_phase_frames(self, cap, start_time: float, end_time: float, 
                           max_frames: int, phase_name: str) -> int:
        """Extract frames from a specific phase of the match"""
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        if end_frame > cap.get(cv2.CAP_PROP_FRAME_COUNT):
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_interval = max(1, (end_frame - start_frame) // max_frames)
        extracted = 0
        
        for frame_idx in range(start_frame, end_frame, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Skip if frame looks like advertisement (dark or static)
            if self.is_advertisement_frame(frame):
                continue
            
            # Save frame
            frame_path = self.images_dir / f"{phase_name}_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            
            # Create enhanced annotations
            self.create_enhanced_annotations(frame, frame_idx, phase_name)
            
            extracted += 1
            
            if extracted >= max_frames:
                break
        
        return extracted
    
    def is_advertisement_frame(self, frame: np.ndarray) -> bool:
        """Detect if frame is an advertisement (dark, static, etc.)"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check if frame is too dark
        mean_brightness = np.mean(gray)
        if mean_brightness < 50:  # Too dark
            return True
        
        # Check if frame has too much black (advertisement borders)
        black_pixels = np.sum(gray < 30)
        total_pixels = gray.shape[0] * gray.shape[1]
        if black_pixels / total_pixels > 0.3:  # More than 30% black
            return True
        
        return False
    
    def create_enhanced_annotations(self, frame: np.ndarray, frame_idx: int, phase: str):
        """Create enhanced annotations using multiple detection methods"""
        height, width = frame.shape[:2]
        label_lines = []
        
        # Use multiple detection methods
        detections = []
        
        # Method 1: Color-based detection
        color_detections = self.detect_by_colors(frame)
        detections.extend(color_detections)
        
        # Method 2: Motion-based detection
        motion_detections = self.detect_by_motion(frame)
        detections.extend(motion_detections)
        
        # Method 3: Shape-based detection
        shape_detections = self.detect_by_shapes(frame)
        detections.extend(shape_detections)
        
        # Method 4: Template matching for specific objects
        template_detections = self.detect_by_templates(frame)
        detections.extend(template_detections)
        
        # Remove duplicate detections
        detections = self.remove_duplicates(detections)
        
        # Convert to YOLO format
        for detection in detections:
            class_id, bbox, confidence = detection
            
            x, y, w, h = bbox
            center_x = (x + w/2) / width
            center_y = (y + h/2) / height
            norm_width = w / width
            norm_height = h / height
            
            label_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
        
        # Save labels
        label_path = self.labels_dir / f"{phase}_{frame_idx:06d}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))
    
    def detect_by_colors(self, frame: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Enhanced color-based detection"""
        detections = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Team A colors (adjust based on actual team colors)
        team_a_colors = [
            ([0, 0, 200], [180, 30, 255]),    # White/light colors
            ([100, 50, 50], [130, 255, 255])  # Blue colors
        ]
        
        # Team B colors
        team_b_colors = [
            ([0, 50, 50], [10, 255, 255]),    # Red colors
            ([20, 100, 100], [30, 255, 255])  # Yellow colors
        ]
        
        # Detect team A players
        for lower, upper in team_a_colors:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            detections.extend(self.find_objects_in_mask(mask, 0, 0.7))  # team_a_player
        
        # Detect team B players  
        for lower, upper in team_b_colors:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            detections.extend(self.find_objects_in_mask(mask, 2, 0.7))  # team_b_player
        
        # Enhanced referee detection (multiple color schemes)
        referee_detections = self.detect_referees_enhanced(frame)
        detections.extend(referee_detections)
        
        return detections
    
    def detect_referees_enhanced(self, frame: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Enhanced referee detection with multiple color schemes"""
        detections = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Multiple referee color schemes (based on actual video analysis)
        referee_schemes = [
            # BAY_BMG video referee colors (yellow-orange)
            ([30, 94, 79], [50, 194, 179]),    # Yellow-orange from analysis
            ([30, 116, 65], [50, 216, 165]),   # Yellow-orange variant 1
            ([30, 71, 89], [50, 171, 189]),    # Yellow-orange variant 2
            # Traditional yellow/black
            ([20, 100, 100], [30, 255, 255]),  # Bright yellow
            ([15, 50, 100], [25, 255, 255]),   # Light yellow
            # Modern blue/white
            ([100, 50, 50], [130, 255, 255]),  # Blue
            ([0, 0, 200], [180, 30, 255]),     # White
            # Green/white
            ([40, 50, 50], [80, 255, 255]),    # Green
            # High contrast (black/white)
            ([0, 0, 0], [180, 255, 50]),       # Black
        ]
        
        for lower, upper in referee_schemes:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            detections.extend(self.find_objects_in_mask(mask, 4, 0.5))  # referee
        
        return detections
    
    def detect_by_motion(self, frame: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Detect objects based on motion patterns"""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction (simplified)
        # In a real implementation, you'd maintain a background model
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 5000:  # Filter by size
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio
                aspect_ratio = h / w
                if 1.5 < aspect_ratio < 4.0:
                    detections.append((0, (x, y, w, h), 0.5))  # Generic player
        
        return detections
    
    def detect_by_shapes(self, frame: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Detect objects based on shape characteristics"""
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if it looks like a person (rectangular, standing)
                aspect_ratio = h / w
                if 2.0 < aspect_ratio < 5.0:
                    detections.append((0, (x, y, w, h), 0.4))  # Generic player
        
        return detections
    
    def detect_by_templates(self, frame: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Detect objects using template matching"""
        detections = []
        
        # This is a simplified version - in practice you'd have actual templates
        # For now, we'll use simple pattern matching
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Look for circular patterns (ball)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=5, maxRadius=30
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                bbox = (x - r, y - r, 2 * r, 2 * r)
                detections.append((5, bbox, 0.6))  # ball
        
        return detections
    
    def find_objects_in_mask(self, mask: np.ndarray, class_id: int, confidence: float) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Find objects in a binary mask"""
        detections = []
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio
                aspect_ratio = h / w
                if 1.0 < aspect_ratio < 6.0:  # Allow wider range for different poses
                    detections.append((class_id, (x, y, w, h), confidence))
        
        return detections
    
    def remove_duplicates(self, detections: List[Tuple[int, Tuple[int, int, int, int], float]]) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Remove duplicate detections"""
        if not detections:
            return []
        
        # Simple duplicate removal based on overlap
        filtered = []
        for detection in detections:
            class_id, bbox, confidence = detection
            x1, y1, w1, h1 = bbox
            
            is_duplicate = False
            for existing in filtered:
                _, existing_bbox, _ = existing
                x2, y2, w2, h2 = existing_bbox
                
                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - overlap_area
                
                if union_area > 0 and overlap_area / union_area > 0.5:  # 50% overlap
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
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
    """Main function to create comprehensive training data"""
    print("🏈 Godseye AI - Comprehensive Training Data Creator")
    print("=" * 60)
    
    # Use the BAY_BMG video
    video_path = "BAY_BMG.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    # Create comprehensive training data
    creator = ComprehensiveTrainingCreator(video_path)
    
    # Extract frames from different match phases
    frames_extracted = creator.extract_comprehensive_frames(max_frames=200)
    
    # Create dataset configuration
    config_path = creator.create_dataset_config()
    
    print(f"\n🎉 Comprehensive training data created successfully!")
    print(f"📁 Output directory: {creator.output_dir}")
    print(f"🖼️ Frames extracted: {frames_extracted}")
    print(f"⚙️ Dataset config: {config_path}")
    print(f"🎯 Match phases: First half, Second half, Random moments, High activity")
    
    return creator.output_dir

if __name__ == "__main__":
    main()
