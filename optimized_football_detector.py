#!/usr/bin/env python3
"""
Optimized Football Detector using Best Pre-trained Model
Ready for production use with excellent accuracy
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

class OptimizedFootballDetector:
    """Production-ready football detector with best pre-trained model"""
    
    def __init__(self):
        print("🚀 Loading optimized football detection model...")
        
        # Load the best available model
        model_path = "pretrained_models/yolov8l_sports.pt"
        self.model = YOLO(model_path)
        
        print(f"✅ Model loaded: {model_path}")
        print(f"📊 Model classes: {self.model.names}")
        
        # Football-specific configuration
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        
        # Team colors for classification
        self.team_colors = {
            'team_a': (0, 0, 255),      # Red
            'team_b': (255, 0, 0),      # Blue
            'referee': (0, 255, 255),   # Yellow
            'unknown': (128, 128, 128)  # Gray
        }
        
        self.stats = {
            'team_a': 0,
            'team_b': 0, 
            'referee': 0,
            'ball': 0,
            'total_frames': 0
        }
    
    def classify_team_by_jersey(self, image, bbox):
        """Advanced team classification by jersey color"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Extract jersey region (upper portion)
            jersey_height = max(1, int((y2 - y1) * 0.4))
            jersey_region = image[y1:y1+jersey_height, x1:x2]
            
            if jersey_region.size == 0:
                return 'unknown'
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
            
            # Calculate average color
            avg_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
            h, s, v = avg_hsv
            
            # Team classification logic
            if s < 50 and v > 150:  # White/light colors
                return 'team_a'  # Assume white team
            elif 90 <= h <= 130 and s > 40:  # Blue range
                return 'team_b'
            elif (0 <= h <= 15 or 165 <= h <= 180) and s > 40:  # Red range
                return 'team_a'
            elif 15 <= h <= 60 and s > 50:  # Yellow/green (referee)
                return 'referee'
            elif v < 70:  # Dark colors (referee)
                return 'referee'
            else:
                return 'unknown'
                
        except Exception:
            return 'unknown'
    
    def detect_and_classify(self, frame):
        """Detect and classify all objects in frame"""
        # Run YOLO detection
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        detections = []
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                bbox = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Focus on persons and sports balls
                if class_id == 0:  # Person
                    team = self.classify_team_by_jersey(frame, bbox)
                    detections.append({
                        'type': 'person',
                        'team': team,
                        'bbox': bbox,
                        'confidence': confidence
                    })
                    self.stats[team] += 1
                    
                elif class_id == 32:  # Sports ball
                    detections.append({
                        'type': 'ball',
                        'team': 'none',
                        'bbox': bbox,
                        'confidence': confidence
                    })
                    self.stats['ball'] += 1
        
        self.stats['total_frames'] += 1
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw all detections with team colors"""
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            if detection['type'] == 'person':
                color = self.team_colors[detection['team']]
                label = f"{detection['team'].upper()}: {detection['confidence']:.2f}"
                thickness = 3
                
            elif detection['type'] == 'ball':
                color = (0, 0, 255)  # Red for ball
                label = f"BALL: {detection['confidence']:.2f}"
                thickness = 4
                
                # Special ball highlighting
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.circle(frame, center, 25, color, 3)
                cv2.putText(frame, "⚽", (center[0] - 15, center[1] + 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 15), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            
            text_color = (0, 0, 0) if detection['team'] == 'team_a' else (255, 255, 255)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    def draw_statistics(self, frame):
        """Draw real-time statistics"""
        height, width = frame.shape[:2]
        
        # Background
        cv2.rectangle(frame, (10, 10), (600, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (600, 150), (255, 255, 255), 3)
        
        # Title
        cv2.putText(frame, "🎯 GODSEYE AI - OPTIMIZED DETECTION", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Statistics
        y_offset = 70
        cv2.putText(frame, f"Frame: {self.stats['total_frames']}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        stats_text = f"Team A: {self.stats['team_a']} | Team B: {self.stats['team_b']} | Referee: {self.stats['referee']} | Ball: {self.stats['ball']}"
        cv2.putText(frame, stats_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Controls
        cv2.putText(frame, "Controls: SPACE=pause, Q=quit, R=reset", 
                   (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def process_video(self, video_path):
        """Process video with optimized detection"""
        print(f"🎥 Processing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Cannot open video")
            return
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect and classify
                detections = self.detect_and_classify(frame)
                
                # Draw results
                self.draw_detections(frame, detections)
                self.draw_statistics(frame)
            
            # Display
            cv2.imshow('🎯 Optimized Football Detection', frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                self.stats = {k: 0 for k in self.stats}
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("🎉 Processing complete!")
        print(f"📊 Final stats: {self.stats}")

def main():
    detector = OptimizedFootballDetector()
    
    # Find video file
    video_files = ["madrid_vs_city.mp4", "data/madrid_vs_city.mp4", "BAY_BMG.mp4"]
    video_path = None
    
    for path in video_files:
        if os.path.exists(path):
            video_path = path
            break
    
    if video_path:
        detector.process_video(video_path)
    else:
        print("❌ No video file found for testing")

if __name__ == "__main__":
    main()
