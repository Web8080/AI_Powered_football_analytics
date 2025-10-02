#!/usr/bin/env python3
"""
Improved Team Classification with Better Color Analysis
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
from collections import Counter

class ImprovedTeamClassifier:
    """Improved football analysis with better team classification"""
    
    def __init__(self):
        print("🚀 Loading pretrained YOLOv8n model...")
        self.model = YOLO('yolov8n.pt')
        
        # Statistics
        self.frame_count = 0
        self.team_stats = {
            'team_a': 0,
            'team_b': 0, 
            'referee': 0,
            'unknown': 0,
            'ball': 0
        }
    
    def extract_dominant_colors(self, image, bbox, num_colors=3):
        """Extract dominant colors from jersey region using k-means"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Extract jersey region (upper 40% of person)
            jersey_height = max(1, int((y2 - y1) * 0.4))
            jersey_region = image[y1:y1+jersey_height, x1:x2]
            
            if jersey_region.size == 0:
                return []
            
            # Resize for faster processing
            jersey_region = cv2.resize(jersey_region, (50, 50))
            
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2LAB)
            
            # Flatten pixels
            pixels_hsv = hsv.reshape(-1, 3)
            pixels_bgr = jersey_region.reshape(-1, 3)
            
            # Remove very dark and very bright pixels (shadows/highlights)
            mask = (pixels_hsv[:, 2] > 30) & (pixels_hsv[:, 2] < 220)  # V channel
            if np.sum(mask) < 10:  # If too few pixels, use all
                mask = np.ones(len(pixels_hsv), dtype=bool)
            
            filtered_hsv = pixels_hsv[mask]
            filtered_bgr = pixels_bgr[mask]
            
            if len(filtered_hsv) == 0:
                return []
            
            # Calculate dominant color in HSV
            dominant_hsv = np.mean(filtered_hsv, axis=0)
            dominant_bgr = np.mean(filtered_bgr, axis=0)
            
            return dominant_hsv, dominant_bgr
            
        except Exception as e:
            return []
    
    def classify_team_improved(self, image, bbox):
        """Improved team classification with better color analysis"""
        try:
            colors = self.extract_dominant_colors(image, bbox)
            if not colors:
                return 'unknown'
            
            dominant_hsv, dominant_bgr = colors
            h, s, v = dominant_hsv
            b, g, r = dominant_bgr
            
            # Debug: print color values occasionally
            if self.frame_count % 60 == 0:  # Every 60 frames
                print(f"Color analysis - HSV: ({h:.1f}, {s:.1f}, {v:.1f}), BGR: ({b:.1f}, {g:.1f}, {r:.1f})")
            
            # Enhanced classification logic
            
            # 1. Check for referee (typically black/white or bright colors)
            if s < 40 and v > 100:  # Low saturation, high brightness = white/gray
                if v > 180:
                    return 'referee'  # Bright white
                elif 100 < v < 180:
                    return 'referee'  # Gray
            
            # 2. Check for black (referee or dark kit)
            if v < 60:  # Very dark
                return 'referee'
            
            # 3. Check for yellow/green (referee)
            if 20 <= h <= 60 and s > 50:  # Yellow to green range with good saturation
                return 'referee'
            
            # 4. Team classification based on hue
            if s > 30:  # Good saturation for team colors
                
                # Red team (Manchester City away or similar)
                if (0 <= h <= 15) or (165 <= h <= 180):
                    return 'team_a'
                
                # Blue team (Manchester City home)
                elif 90 <= h <= 130:
                    return 'team_b'
                
                # Additional team colors
                elif 30 <= h <= 50:  # Orange/yellow range
                    return 'team_a'
                elif 130 <= h <= 165:  # Purple/magenta range
                    return 'team_b'
            
            # 5. Fallback: use BGR values
            # Check if predominantly one color
            if r > g + 30 and r > b + 30:  # More red
                return 'team_a'
            elif b > r + 30 and b > g + 30:  # More blue
                return 'team_b'
            elif g > r + 20 and g > b + 20:  # More green (could be referee)
                return 'referee'
            
            return 'unknown'
            
        except Exception as e:
            return 'unknown'
    
    def draw_enhanced_detection(self, image, bbox, class_id, confidence, team=None):
        """Draw detection with improved team visualization"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Team colors (more distinct)
        team_colors = {
            'team_a': (0, 0, 255),      # Red
            'team_b': (255, 0, 0),      # Blue
            'referee': (0, 255, 255),   # Yellow
            'unknown': (128, 128, 128)  # Gray
        }
        
        if class_id == 0:  # Person
            color = team_colors.get(team, (0, 255, 0))
            
            # Create more descriptive labels
            team_labels = {
                'team_a': "TEAM A",
                'team_b': "TEAM B", 
                'referee': "REFEREE",
                'unknown': "PLAYER"
            }
            
            label = f"{team_labels.get(team, 'PLAYER')}: {confidence:.2f}"
            
            # Thicker boxes for better visibility
            thickness = 3
            
        elif class_id == 32:  # Sports ball
            color = (0, 0, 255)  # Bright red for ball
            label = f"BALL: {confidence:.2f}"
            thickness = 4
        else:
            color = (255, 255, 255)
            label = f"OBJECT: {confidence:.2f}"
            thickness = 2
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label with background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 20), 
                     (x1 + label_size[0] + 15, y1), color, -1)
        cv2.putText(image, label, (x1 + 5, y1 - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Special effects for ball
        if class_id == 32:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(image, center, 30, (0, 0, 255), 4)
            cv2.putText(image, "⚽", (center[0] - 20, center[1] + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Add team indicator on the side for persons
        if class_id == 0 and team != 'unknown':
            # Small colored rectangle on the right side
            indicator_x = x2 + 5
            indicator_y = y1
            cv2.rectangle(image, (indicator_x, indicator_y), 
                         (indicator_x + 15, indicator_y + 30), color, -1)
            cv2.rectangle(image, (indicator_x, indicator_y), 
                         (indicator_x + 15, indicator_y + 30), (255, 255, 255), 2)
    
    def draw_team_statistics(self, image, frame_stats):
        """Draw enhanced team statistics"""
        height, width = image.shape[:2]
        
        # Background
        cv2.rectangle(image, (10, 10), (700, 220), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (700, 220), (255, 255, 255), 3)
        
        # Title
        cv2.putText(image, "🎯 GODSEYE AI - IMPROVED TEAM CLASSIFICATION", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Frame info
        cv2.putText(image, f"Frame: {self.frame_count}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Current frame stats
        y_offset = 100
        cv2.putText(image, "CURRENT FRAME:", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        y_offset += 25
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (128, 128, 128), (0, 0, 255)]
        labels = ["Team A", "Team B", "Referee", "Unknown", "Ball"]
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            count = frame_stats.get(label.lower().replace(" ", "_"), 0)
            cv2.putText(image, f"{label}: {count}", 
                       (20 + (i * 120), y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Total statistics
        y_offset += 40
        cv2.putText(image, "TOTAL DETECTIONS:", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        y_offset += 25
        for i, (label, color) in enumerate(zip(labels, colors)):
            count = self.team_stats.get(label.lower().replace(" ", "_"), 0)
            cv2.putText(image, f"{label}: {count}", 
                       (20 + (i * 120), y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Controls
        cv2.putText(image, "Controls: SPACE=pause, Q=quit, R=reset stats", 
                   (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def process_video(self, video_path, max_frames=400):
        """Process video with improved team classification"""
        print(f"🎥 Opening video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video: {width}x{height}, {fps} FPS")
        print(f"🚀 Starting improved team classification...")
        print(f"💡 Look for RED (Team A), BLUE (Team B), YELLOW (Referee) boxes!")
        
        paused = False
        confidence = 0.4
        start_time = time.time()
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or self.frame_count >= max_frames:
                    break
                
                self.frame_count += 1
                
                # Run detection
                results = self.model(frame, conf=confidence, verbose=False)
                
                # Frame statistics
                frame_stats = {
                    'team_a': 0,
                    'team_b': 0,
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
                            # Classify team
                            team = self.classify_team_improved(frame, bbox)
                            frame_stats[team] += 1
                            self.team_stats[team] += 1
                            
                            # Draw detection
                            self.draw_enhanced_detection(frame, bbox, class_id, conf, team)
                            
                        elif class_id == 32:  # Ball
                            frame_stats['ball'] += 1
                            self.team_stats['ball'] += 1
                            self.draw_enhanced_detection(frame, bbox, class_id, conf)
                
                # Draw statistics
                self.draw_team_statistics(frame, frame_stats)
                
                # Print progress
                if self.frame_count % 50 == 0:
                    total_players = frame_stats['team_a'] + frame_stats['team_b'] + frame_stats['referee'] + frame_stats['unknown']
                    print(f"Frame {self.frame_count}: Team A: {frame_stats['team_a']}, Team B: {frame_stats['team_b']}, Ref: {frame_stats['referee']}, Unknown: {frame_stats['unknown']}, Ball: {frame_stats['ball']}")
            
            # Display
            cv2.imshow('🎯 Godseye AI - Improved Team Classification', frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print(f"⏸️ {'Paused' if paused else 'Resumed'}")
            elif key == ord('r'):
                self.team_stats = {k: 0 for k in self.team_stats}
                print("🔄 Statistics reset")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final results
        elapsed = time.time() - start_time
        print(f"\n🎉 IMPROVED CLASSIFICATION COMPLETE!")
        print(f"⏱️ Time: {elapsed:.1f}s | Frames: {self.frame_count}")
        print(f"🎯 FINAL TEAM STATISTICS:")
        for team, count in self.team_stats.items():
            print(f"   {team.upper()}: {count}")

def main():
    print("🎯 GODSEYE AI - IMPROVED TEAM CLASSIFICATION")
    print("=" * 60)
    print("🚀 Enhanced color analysis for better team detection!")
    
    classifier = ImprovedTeamClassifier()
    
    # Find video
    video_paths = ["madrid_vs_city.mp4", "data/madrid_vs_city.mp4", "BAY_BMG.mp4"]
    video_path = None
    
    for path in video_paths:
        if os.path.exists(path):
            video_path = path
            break
    
    if not video_path:
        print("❌ No video found!")
        return
    
    classifier.process_video(video_path)

if __name__ == "__main__":
    main()
