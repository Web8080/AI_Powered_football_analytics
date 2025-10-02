#!/usr/bin/env python3
"""
Final Optimized Team Classifier for Manchester City vs Real Madrid
Specifically tuned for light blue vs white jerseys
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

class FinalTeamClassifier:
    """Final optimized team classifier for Manchester City match"""
    
    def __init__(self):
        print("🚀 Loading YOLOv8n for Manchester City vs Real Madrid...")
        self.model = YOLO('yolov8n.pt')
        
        self.frame_count = 0
        self.team_stats = {
            'man_city': 0,      # Light blue
            'real_madrid': 0,   # White  
            'referee': 0,       # Black/other
            'unknown': 0,
            'ball': 0
        }
    
    def classify_manchester_teams(self, image, bbox):
        """Optimized classification for Manchester City (light blue) vs Real Madrid (white)"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Extract jersey region
            jersey_height = max(1, int((y2 - y1) * 0.45))
            jersey_region = image[y1:y1+jersey_height, x1:x2]
            
            if jersey_region.size == 0:
                return 'unknown'
            
            # Resize for consistent analysis
            jersey_region = cv2.resize(jersey_region, (40, 40))
            
            # Convert to HSV and LAB color spaces
            hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2LAB)
            
            # Get average colors
            avg_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
            avg_bgr = np.mean(jersey_region.reshape(-1, 3), axis=0)
            avg_lab = np.mean(lab.reshape(-1, 3), axis=0)
            
            h, s, v = avg_hsv
            b, g, r = avg_bgr
            l, a, lab_b = avg_lab
            
            # Manchester City Detection (Light Blue)
            # Look for blue hue with moderate saturation
            if 90 <= h <= 130 and s > 40 and v > 80:  # Blue range
                return 'man_city'
            
            # Alternative blue detection using BGR
            if b > r + 20 and b > g + 10 and v > 100:  # More blue than red/green
                return 'man_city'
            
            # Real Madrid Detection (White/Light colors)
            # Look for high brightness, low saturation
            if s < 50 and v > 140:  # Low saturation, high brightness = white
                return 'real_madrid'
            
            # Alternative white detection using LAB
            if l > 120 and abs(a - 128) < 20 and abs(lab_b - 128) < 20:  # High lightness, neutral colors
                return 'real_madrid'
            
            # Referee Detection (Dark colors, black, or bright yellow)
            if v < 70:  # Very dark (black referee kit)
                return 'referee'
            
            if 15 <= h <= 35 and s > 100:  # Bright yellow
                return 'referee'
            
            # Green detection (referee or pitch)
            if 40 <= h <= 80 and s > 60:
                return 'referee'
            
            # If none of the above, classify based on dominant color
            if r > b and r > g:  # Reddish
                return 'real_madrid'  # Could be red away kit
            elif b > r and b > g:  # Bluish
                return 'man_city'
            else:
                return 'unknown'
                
        except Exception as e:
            return 'unknown'
    
    def draw_team_detection(self, image, bbox, class_id, confidence, team=None):
        """Draw detection with Manchester City specific colors"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Team-specific colors and labels
        team_info = {
            'man_city': ((255, 150, 0), "MAN CITY"),      # Sky blue
            'real_madrid': ((255, 255, 255), "REAL MADRID"), # White
            'referee': ((0, 255, 255), "REFEREE"),        # Yellow
            'unknown': ((128, 128, 128), "PLAYER")        # Gray
        }
        
        if class_id == 0:  # Person
            color, label_text = team_info.get(team, ((0, 255, 0), "PLAYER"))
            label = f"{label_text}: {confidence:.2f}"
            thickness = 3
            
        elif class_id == 32:  # Ball
            color = (0, 0, 255)  # Red
            label = f"BALL: {confidence:.2f}"
            thickness = 4
        else:
            color = (255, 255, 255)
            label = f"OBJECT: {confidence:.2f}"
            thickness = 2
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        font_scale = 0.7
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        
        # Label background
        cv2.rectangle(image, (x1, y1 - label_size[1] - 15), 
                     (x1 + label_size[0] + 10, y1), color, -1)
        
        # Label text
        text_color = (0, 0, 0) if team == 'real_madrid' else (255, 255, 255)
        cv2.putText(image, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
        
        # Special ball highlighting
        if class_id == 32:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(image, center, 25, (0, 0, 255), 3)
            cv2.putText(image, "⚽", (center[0] - 15, center[1] + 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Team indicator
        if class_id == 0:
            indicator_size = 20
            cv2.rectangle(image, (x2 + 5, y1), (x2 + 5 + indicator_size, y1 + indicator_size), color, -1)
            cv2.rectangle(image, (x2 + 5, y1), (x2 + 5 + indicator_size, y1 + indicator_size), (255, 255, 255), 2)
    
    def draw_match_statistics(self, image, frame_stats):
        """Draw Manchester City match statistics"""
        height, width = image.shape[:2]
        
        # Background
        cv2.rectangle(image, (10, 10), (800, 250), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (800, 250), (255, 255, 255), 3)
        
        # Title
        cv2.putText(image, "🎯 MANCHESTER CITY vs REAL MADRID - TEAM DETECTION", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Frame info
        cv2.putText(image, f"Frame: {self.frame_count}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Current frame stats
        y_offset = 100
        cv2.putText(image, "CURRENT FRAME:", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        y_offset += 30
        # Manchester City
        cv2.rectangle(image, (20, y_offset - 20), (40, y_offset), (255, 150, 0), -1)
        cv2.putText(image, f"Man City: {frame_stats.get('man_city', 0)}", 
                   (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
        
        # Real Madrid  
        cv2.rectangle(image, (200, y_offset - 20), (220, y_offset), (255, 255, 255), -1)
        cv2.rectangle(image, (200, y_offset - 20), (220, y_offset), (0, 0, 0), 2)
        cv2.putText(image, f"Real Madrid: {frame_stats.get('real_madrid', 0)}", 
                   (230, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Referee
        cv2.rectangle(image, (420, y_offset - 20), (440, y_offset), (0, 255, 255), -1)
        cv2.putText(image, f"Referee: {frame_stats.get('referee', 0)}", 
                   (450, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Ball
        cv2.circle(image, (580, y_offset - 10), 10, (0, 0, 255), -1)
        cv2.putText(image, f"Ball: {frame_stats.get('ball', 0)}", 
                   (600, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Total statistics
        y_offset += 50
        cv2.putText(image, "MATCH TOTALS:", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        y_offset += 30
        teams = [
            ('man_city', 'Man City', (255, 150, 0)),
            ('real_madrid', 'Real Madrid', (255, 255, 255)),
            ('referee', 'Referee', (0, 255, 255)),
            ('ball', 'Ball', (0, 0, 255))
        ]
        
        x_offset = 20
        for team_key, team_name, color in teams:
            count = self.team_stats.get(team_key, 0)
            text_color = (0, 0, 0) if team_key == 'real_madrid' else color
            cv2.putText(image, f"{team_name}: {count}", 
                       (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            x_offset += 150
        
        # Controls
        cv2.putText(image, "Controls: SPACE=pause, Q=quit, R=reset", 
                   (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def process_video(self, video_path, max_frames=500):
        """Process Manchester City video"""
        print(f"🎥 Processing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video: {width}x{height} @ {fps}fps")
        print(f"🚀 Analyzing Manchester City vs Real Madrid...")
        print(f"💡 Sky Blue = Man City, White = Real Madrid, Yellow = Referee")
        
        paused = False
        confidence = 0.4
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or self.frame_count >= max_frames:
                    break
                
                self.frame_count += 1
                
                # YOLO detection
                results = self.model(frame, conf=confidence, verbose=False)
                
                # Frame stats
                frame_stats = {
                    'man_city': 0,
                    'real_madrid': 0,
                    'referee': 0,
                    'unknown': 0,
                    'ball': 0
                }
                
                # Process detections
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        bbox = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        if class_id == 0:  # Person
                            team = self.classify_manchester_teams(frame, bbox)
                            frame_stats[team] += 1
                            self.team_stats[team] += 1
                            self.draw_team_detection(frame, bbox, class_id, conf, team)
                            
                        elif class_id == 32:  # Ball
                            frame_stats['ball'] += 1
                            self.team_stats['ball'] += 1
                            self.draw_team_detection(frame, bbox, class_id, conf)
                
                # Draw statistics
                self.draw_match_statistics(frame, frame_stats)
                
                # Progress update
                if self.frame_count % 50 == 0:
                    print(f"Frame {self.frame_count}: City: {frame_stats['man_city']}, Madrid: {frame_stats['real_madrid']}, Ref: {frame_stats['referee']}, Ball: {frame_stats['ball']}")
            
            # Display
            cv2.imshow('🎯 Manchester City vs Real Madrid - Team Detection', frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                self.team_stats = {k: 0 for k in self.team_stats}
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final results
        print(f"\n🎉 MANCHESTER CITY vs REAL MADRID ANALYSIS COMPLETE!")
        print(f"📊 Frames analyzed: {self.frame_count}")
        print(f"🎯 FINAL TEAM DETECTION RESULTS:")
        print(f"   ⚪ Manchester City: {self.team_stats['man_city']}")
        print(f"   🤍 Real Madrid: {self.team_stats['real_madrid']}")
        print(f"   🟡 Referee: {self.team_stats['referee']}")
        print(f"   ⚽ Ball: {self.team_stats['ball']}")

def main():
    print("🎯 MANCHESTER CITY vs REAL MADRID - FINAL TEAM CLASSIFIER")
    print("=" * 70)
    
    classifier = FinalTeamClassifier()
    
    # Find video
    video_paths = ["madrid_vs_city.mp4", "data/madrid_vs_city.mp4"]
    video_path = None
    
    for path in video_paths:
        if os.path.exists(path):
            video_path = path
            break
    
    if not video_path:
        print("❌ Manchester City video not found!")
        return
    
    classifier.process_video(video_path)

if __name__ == "__main__":
    main()
