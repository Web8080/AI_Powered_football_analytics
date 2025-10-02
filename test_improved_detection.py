#!/usr/bin/env python3
"""
Improved Real-time Testing with Lower Confidence and Debugging
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import time

class ImprovedFootballAnalyzer:
    """Improved football analysis with debugging"""
    
    def __init__(self):
        # Load our trained practical model
        self.model_path = "practical_training/godseye_practical_model/weights/best.pt"
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found at {self.model_path}")
            # Try alternative paths
            alt_paths = [
                "models/godseye_practical_model.pt",
                "practical_training/godseye_practical_model/weights/last.pt",
                "runs/detect/train/weights/best.pt",
                "yolov8n.pt"  # Fallback to pretrained
            ]
            
            for path in alt_paths:
                if os.path.exists(path):
                    self.model_path = path
                    print(f"✅ Found model at: {path}")
                    break
        
        print(f"🚀 Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # Print model info
        print(f"📋 Model classes: {self.model.names}")
        
        # Class names - adapt to actual model
        if hasattr(self.model, 'names') and self.model.names:
            self.class_names = self.model.names
        else:
            self.class_names = {
                0: "person",
                1: "ball", 
                2: "goalpost",
                3: "field"
            }
        
        print(f"🏷️ Using classes: {self.class_names}")
        
        # Colors for each class (BGR format)
        self.class_colors = {
            0: (0, 255, 0),    # Green for person
            1: (0, 0, 255),    # Red for ball
            2: (255, 0, 0),    # Blue for goalpost
            3: (128, 128, 128), # Gray for field
            # Add more colors for COCO classes if using pretrained
            16: (0, 0, 255),   # Bird -> Ball (COCO class 16)
            32: (0, 0, 255),   # Sports ball (COCO class 32)
        }
        
        # Statistics
        self.frame_count = 0
        self.total_detections = 0
    
    def draw_detection(self, image, bbox, class_id, confidence):
        """Draw detection with debugging info"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get class info
        class_name = self.class_names.get(class_id, f"class_{class_id}")
        color = self.class_colors.get(class_id, (255, 255, 255))
        
        # Draw bounding box
        thickness = 3 if 'ball' in class_name.lower() or class_id == 32 else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"{class_name}: {confidence:.3f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Special highlighting for ball-like objects
        if 'ball' in class_name.lower() or class_id in [32, 1]:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(image, center, 20, (0, 0, 255), 3)
    
    def draw_info(self, image, detections_this_frame):
        """Draw information overlay"""
        # Background
        cv2.rectangle(image, (10, 10), (500, 120), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (500, 120), (255, 255, 255), 2)
        
        # Title
        cv2.putText(image, "🎯 GODSEYE AI - IMPROVED DETECTION", 
                   (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Stats
        cv2.putText(image, f"Frame: {self.frame_count} | Total Detections: {self.total_detections}", 
                   (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(image, f"This Frame: {detections_this_frame} detections", 
                   (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(image, f"Model: {os.path.basename(self.model_path)}", 
                   (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def process_video(self, video_path, max_frames=500):
        """Process video with improved detection"""
        print(f"🎥 Opening video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video Info: {width}x{height}, {fps} FPS, {total_frames} frames")
        print(f"🚀 Starting analysis (max {max_frames} frames)...")
        print(f"💡 Controls: SPACE=pause, Q=quit")
        
        paused = False
        start_time = time.time()
        
        # Try different confidence thresholds
        confidence_levels = [0.01, 0.05, 0.1, 0.2]  # Very low to catch anything
        current_conf_idx = 0
        current_conf = confidence_levels[current_conf_idx]
        
        print(f"🎯 Starting with confidence threshold: {current_conf}")
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or self.frame_count >= max_frames:
                    print("📹 Stopping analysis")
                    break
                
                self.frame_count += 1
                
                # Run inference with very low confidence
                results = self.model(frame, conf=current_conf, verbose=False)
                
                detections_this_frame = 0
                
                # Process detections
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    detections_this_frame = len(results[0].boxes)
                    self.total_detections += detections_this_frame
                    
                    print(f"Frame {self.frame_count}: {detections_this_frame} detections")
                    
                    for box in results[0].boxes:
                        bbox = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Debug print
                        class_name = self.class_names.get(class_id, f"class_{class_id}")
                        print(f"  - {class_name} (ID: {class_id}): {confidence:.3f}")
                        
                        # Draw detection
                        self.draw_detection(frame, bbox, class_id, confidence)
                
                # Draw info overlay
                self.draw_info(frame, detections_this_frame)
                
                # Show confidence level
                cv2.putText(frame, f"Confidence: {current_conf} (Press C to change)", 
                           (15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow('🎯 Godseye AI - Improved Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to pause/unpause
                paused = not paused
                print(f"⏸️ {'Paused' if paused else 'Resumed'}")
            elif key == ord('c'):  # Change confidence threshold
                current_conf_idx = (current_conf_idx + 1) % len(confidence_levels)
                current_conf = confidence_levels[current_conf_idx]
                print(f"🎯 Changed confidence to: {current_conf}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        elapsed_time = time.time() - start_time
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"⏱️ Processing time: {elapsed_time:.1f} seconds")
        print(f"📊 Frames processed: {self.frame_count}")
        print(f"🎯 Total detections: {self.total_detections}")
        print(f"📈 Average detections per frame: {self.total_detections/max(1, self.frame_count):.2f}")

def main():
    """Main function"""
    print("🎯 GODSEYE AI - IMPROVED DETECTION TESTING")
    print("=" * 50)
    
    analyzer = ImprovedFootballAnalyzer()
    
    # Look for video files
    video_paths = [
        "madrid_vs_city.mp4",
        "data/madrid_vs_city.mp4", 
        "BAY_BMG.mp4",
        "data/BAY_BMG.mp4"
    ]
    
    video_path = None
    for path in video_paths:
        if os.path.exists(path):
            video_path = path
            break
    
    if not video_path:
        print("❌ No test video found!")
        # Search for any video files
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(root, file)
                    print(f"✅ Found video: {video_path}")
                    break
            if video_path:
                break
    
    if not video_path:
        print("❌ No video files found!")
        return
    
    # Process video with debugging
    analyzer.process_video(video_path, max_frames=200)  # Limit frames for testing

if __name__ == "__main__":
    main()
