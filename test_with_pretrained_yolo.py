#!/usr/bin/env python3
"""
Test with Pretrained YOLOv8 + Team Classification
This will show you proper bounding boxes on players, ball, etc.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

class PretrainedFootballAnalyzer:
    """Football analysis using pretrained YOLO + team classification"""
    
    def __init__(self):
        print("🚀 Loading pretrained YOLOv8n model...")
        self.model = YOLO('yolov8n.pt')  # Use pretrained COCO model
        
        # COCO classes relevant to football
        self.football_classes = {
            0: 'person',        # People (players, refs, etc.)
            32: 'sports ball',  # Ball
            # Add more if needed
        }
        
        # Colors for different teams (we'll classify by jersey color)
        self.team_colors = {
            'team_a': (0, 0, 255),      # Red
            'team_b': (255, 0, 0),      # Blue  
            'referee': (0, 255, 255),   # Yellow
            'unknown': (0, 255, 0)      # Green
        }
        
        # Statistics
        self.frame_count = 0
        self.detection_stats = {
            'players': 0,
            'ball': 0,
            'total_detections': 0
        }
    
    def classify_team_by_color(self, image, bbox):
        """Classify team based on jersey color"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Extract jersey region (upper part of person)
            jersey_height = max(1, int((y2 - y1) * 0.4))
            jersey_region = image[y1:y1+jersey_height, x1:x2]
            
            if jersey_region.size == 0:
                return 'unknown'
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
            
            # Calculate dominant color
            pixels = hsv.reshape(-1, 3)
            dominant_color = np.mean(pixels, axis=0)
            h, s, v = dominant_color
            
            # Enhanced color classification
            if s < 30:  # Low saturation = white/gray (referee or white kit)
                if v > 150:
                    return 'referee'  # Bright white/gray
                else:
                    return 'unknown'
            elif 0 <= h <= 15 or 170 <= h <= 180:  # Red range
                return 'team_a'
            elif 100 <= h <= 130:  # Blue range  
                return 'team_b'
            elif 15 <= h <= 35:  # Yellow/orange range
                return 'referee'
            else:
                return 'unknown'
                
        except Exception as e:
            return 'unknown'
    
    def draw_detection(self, image, bbox, class_id, confidence, team=None):
        """Draw enhanced detection with team classification"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Determine color and label based on class and team
        if class_id == 0:  # Person
            color = self.team_colors.get(team, (0, 255, 0))
            if team == 'team_a':
                label = f"Team A: {confidence:.2f}"
            elif team == 'team_b':
                label = f"Team B: {confidence:.2f}"
            elif team == 'referee':
                label = f"Referee: {confidence:.2f}"
            else:
                label = f"Player: {confidence:.2f}"
        elif class_id == 32:  # Sports ball
            color = (0, 0, 255)  # Red for ball
            label = f"Ball: {confidence:.2f}"
        else:
            color = (255, 255, 255)
            label = f"Object: {confidence:.2f}"
        
        # Draw bounding box
        thickness = 4 if class_id == 32 else 2  # Thicker for ball
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 15), 
                     (x1 + label_size[0] + 10, y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Special highlighting for ball
        if class_id == 32:  # Ball
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(image, center, 25, (0, 0, 255), 4)
            cv2.putText(image, "⚽", (center[0] - 15, center[1] + 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    def draw_info_overlay(self, image, frame_detections):
        """Draw information overlay"""
        height, width = image.shape[:2]
        
        # Background for stats
        cv2.rectangle(image, (10, 10), (600, 180), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (600, 180), (255, 255, 255), 3)
        
        # Title
        cv2.putText(image, "🎯 GODSEYE AI - PRETRAINED YOLO DETECTION", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Current frame detections
        y_offset = 70
        cv2.putText(image, f"Frame {self.frame_count}: {frame_detections['total']} detections", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(image, f"Players: {frame_detections['players']} | Ball: {frame_detections['ball']}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Total statistics
        y_offset += 30
        cv2.putText(image, "TOTAL STATS:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
        cv2.putText(image, f"Players: {self.detection_stats['players']} | Ball: {self.detection_stats['ball']}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controls
        cv2.putText(image, "Controls: SPACE=pause, Q=quit, C=change confidence", 
                   (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def process_video(self, video_path, max_frames=300):
        """Process video with pretrained YOLO"""
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
        print(f"🚀 Starting analysis with pretrained YOLO...")
        print(f"💡 This should show proper bounding boxes on players and ball!")
        
        paused = False
        confidence_threshold = 0.3  # Good starting confidence
        start_time = time.time()
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or self.frame_count >= max_frames:
                    print("📹 Stopping analysis")
                    break
                
                self.frame_count += 1
                
                # Run YOLO inference
                results = self.model(frame, conf=confidence_threshold, verbose=False)
                
                # Count detections for this frame
                frame_detections = {
                    'players': 0,
                    'ball': 0,
                    'total': 0
                }
                
                # Process detections
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        bbox = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Only process relevant classes
                        if class_id in self.football_classes:
                            frame_detections['total'] += 1
                            
                            if class_id == 0:  # Person
                                frame_detections['players'] += 1
                                self.detection_stats['players'] += 1
                                
                                # Classify team by jersey color
                                team = self.classify_team_by_color(frame, bbox)
                                self.draw_detection(frame, bbox, class_id, confidence, team)
                                
                            elif class_id == 32:  # Sports ball
                                frame_detections['ball'] += 1
                                self.detection_stats['ball'] += 1
                                self.draw_detection(frame, bbox, class_id, confidence)
                
                # Draw info overlay
                self.draw_info_overlay(frame, frame_detections)
                
                # Show confidence threshold
                cv2.putText(frame, f"Confidence: {confidence_threshold:.2f}", 
                           (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Print progress every 30 frames
                if self.frame_count % 30 == 0:
                    print(f"Frame {self.frame_count}: {frame_detections['players']} players, {frame_detections['ball']} balls")
            
            # Display frame
            cv2.imshow('🎯 Godseye AI - Pretrained YOLO Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to pause/unpause
                paused = not paused
                print(f"⏸️ {'Paused' if paused else 'Resumed'}")
            elif key == ord('c'):  # Change confidence
                confidence_threshold = 0.1 if confidence_threshold > 0.2 else 0.5
                print(f"🎯 Changed confidence to: {confidence_threshold}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        elapsed_time = time.time() - start_time
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"⏱️ Processing time: {elapsed_time:.1f} seconds")
        print(f"📊 Frames processed: {self.frame_count}")
        print(f"🎯 Total detections:")
        print(f"   Players: {self.detection_stats['players']}")
        print(f"   Ball: {self.detection_stats['ball']}")
        print(f"📈 Average per frame: {(self.detection_stats['players'] + self.detection_stats['ball'])/max(1, self.frame_count):.2f}")

def main():
    """Main function"""
    print("🎯 GODSEYE AI - PRETRAINED YOLO TESTING")
    print("=" * 50)
    print("🚀 Using pretrained YOLOv8n for better detection!")
    
    analyzer = PretrainedFootballAnalyzer()
    
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
        return
    
    # Process video
    analyzer.process_video(video_path, max_frames=300)

if __name__ == "__main__":
    main()
