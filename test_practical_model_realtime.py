#!/usr/bin/env python3
"""
Real-time Testing of Practical Football Model
Test on Manchester City video with live bounding box visualization
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import time

class FootballRealtimeAnalyzer:
    """Real-time football analysis with practical model"""
    
    def __init__(self):
        # Load our trained practical model
        self.model_path = "practical_training/godseye_practical_model/weights/best.pt"
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found at {self.model_path}")
            print("🔍 Looking for alternative model paths...")
            
            # Check alternative paths
            alt_paths = [
                "models/godseye_practical_model.pt",
                "practical_training/godseye_practical_model/weights/last.pt",
                "runs/detect/train/weights/best.pt"
            ]
            
            for path in alt_paths:
                if os.path.exists(path):
                    self.model_path = path
                    print(f"✅ Found model at: {path}")
                    break
            else:
                print("❌ No trained model found. Using YOLOv8n pretrained.")
                self.model_path = "yolov8n.pt"
        
        print(f"🚀 Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # Class names for our practical model
        self.class_names = {
            0: "person",
            1: "ball", 
            2: "goalpost",
            3: "field"
        }
        
        # Colors for each class (BGR format)
        self.class_colors = {
            0: (0, 255, 0),    # Green for person
            1: (0, 0, 255),    # Red for ball
            2: (255, 0, 0),    # Blue for goalpost
            3: (128, 128, 128) # Gray for field
        }
        
        # Team classification colors (for persons)
        self.team_colors = {
            'team_a': (0, 0, 255),      # Red
            'team_b': (255, 0, 0),      # Blue  
            'referee': (0, 255, 255),   # Yellow
            'unknown': (0, 255, 0)      # Green
        }
        
        # Statistics
        self.frame_count = 0
        self.detection_stats = {
            'person': 0,
            'ball': 0,
            'goalpost': 0,
            'field': 0
        }
    
    def classify_team_by_color(self, image, bbox):
        """Classify team based on jersey color in bounding box"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Extract jersey region (upper part of person)
            jersey_height = max(1, int((y2 - y1) * 0.4))  # Top 40% of person
            jersey_region = image[y1:y1+jersey_height, x1:x2]
            
            if jersey_region.size == 0:
                return 'unknown'
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
            
            # Calculate dominant color
            pixels = hsv.reshape(-1, 3)
            dominant_color = np.mean(pixels, axis=0)
            
            # Classify based on color ranges
            h, s, v = dominant_color
            
            # Color classification logic
            if 0 <= h <= 15 or 170 <= h <= 180:  # Red range
                return 'team_a'
            elif 100 <= h <= 130:  # Blue range
                return 'team_b'
            elif 20 <= h <= 35:  # Yellow range
                return 'referee'
            else:
                return 'unknown'
                
        except Exception as e:
            return 'unknown'
    
    def draw_enhanced_bbox(self, image, bbox, class_id, confidence, team=None):
        """Draw enhanced bounding box with team classification"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get class info
        class_name = self.class_names.get(class_id, f"class_{class_id}")
        
        # Choose color based on class and team
        if class_id == 0 and team:  # Person with team classification
            color = self.team_colors.get(team, (0, 255, 0))
            label = f"{team}: {confidence:.2f}"
        else:
            color = self.class_colors.get(class_id, (255, 255, 255))
            label = f"{class_name}: {confidence:.2f}"
        
        # Draw bounding box
        thickness = 3 if class_id == 1 else 2  # Thicker for ball
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Special highlighting for ball
        if class_id == 1:  # Ball
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(image, center, 15, (0, 0, 255), 3)
            cv2.putText(image, "⚽", (center[0] - 10, center[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    def draw_statistics(self, image):
        """Draw real-time statistics on image"""
        # Background for stats
        cv2.rectangle(image, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (400, 150), (255, 255, 255), 2)
        
        # Title
        cv2.putText(image, "🎯 GODSEYE AI - REAL-TIME ANALYSIS", 
                   (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Statistics
        y_offset = 55
        for class_name, count in self.detection_stats.items():
            color = self.class_colors.get(list(self.class_names.values()).index(class_name), (255, 255, 255))
            cv2.putText(image, f"{class_name.capitalize()}: {count}", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
        
        # Frame info
        cv2.putText(image, f"Frame: {self.frame_count}", 
                   (15, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_video(self, video_path):
        """Process video with real-time analysis"""
        print(f"🎥 Opening video: {video_path}")
        
        # Open video
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
        print(f"🚀 Starting real-time analysis...")
        print(f"💡 Controls: SPACE=pause, Q=quit, R=reset stats")
        
        paused = False
        start_time = time.time()
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("📹 End of video reached")
                    break
                
                self.frame_count += 1
                
                # Run inference
                results = self.model(frame, conf=0.25, verbose=False)
                
                # Reset frame stats
                frame_detections = {
                    'person': 0,
                    'ball': 0, 
                    'goalpost': 0,
                    'field': 0
                }
                
                # Process detections
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        # Get detection info
                        bbox = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Update statistics
                        class_name = self.class_names.get(class_id, 'unknown')
                        if class_name in frame_detections:
                            frame_detections[class_name] += 1
                            self.detection_stats[class_name] += 1
                        
                        # Team classification for persons
                        team = None
                        if class_id == 0:  # Person
                            team = self.classify_team_by_color(frame, bbox)
                        
                        # Draw bounding box
                        self.draw_enhanced_bbox(frame, bbox, class_id, confidence, team)
                
                # Draw statistics overlay
                self.draw_statistics(frame)
                
                # Show current detections
                detection_text = " | ".join([f"{k}: {v}" for k, v in frame_detections.items() if v > 0])
                if detection_text:
                    cv2.putText(frame, f"Current: {detection_text}", 
                               (15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('🎯 Godseye AI - Football Analysis', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to pause/unpause
                paused = not paused
                print(f"⏸️ {'Paused' if paused else 'Resumed'}")
            elif key == ord('r'):  # Reset statistics
                self.detection_stats = {k: 0 for k in self.detection_stats}
                print("🔄 Statistics reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        elapsed_time = time.time() - start_time
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"⏱️ Processing time: {elapsed_time:.1f} seconds")
        print(f"📊 Frames processed: {self.frame_count}")
        print(f"🎯 Final Detection Stats:")
        for class_name, count in self.detection_stats.items():
            print(f"   {class_name.capitalize()}: {count} detections")

def main():
    """Main function to test the practical model"""
    print("🎯 GODSEYE AI - PRACTICAL MODEL TESTING")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = FootballRealtimeAnalyzer()
    
    # Look for Manchester City video
    video_paths = [
        "madrid_vs_city.mp4",
        "data/madrid_vs_city.mp4", 
        "videos/madrid_vs_city.mp4",
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
        print("🔍 Looking for any .mp4 files...")
        
        # Search for any .mp4 files
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(root, file)
                    print(f"✅ Found video: {video_path}")
                    break
            if video_path:
                break
    
    if not video_path:
        print("❌ No video files found for testing!")
        print("💡 Please place a football video (madrid_vs_city.mp4) in the current directory")
        return
    
    # Process video
    analyzer.process_video(video_path)

if __name__ == "__main__":
    main()
