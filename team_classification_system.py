#!/usr/bin/env python3
"""
===============================================================================
GODSEYE AI - TEAM CLASSIFICATION SYSTEM
===============================================================================

Advanced team classification system that:
- Identifies Team A vs Team B players
- Detects goalkeepers separately
- Identifies referees
- Uses color-coded bounding boxes
- Shows proper labels (not just "person")

Author: Victor
Date: 2025
Version: 1.0.0
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
from collections import defaultdict, deque
import math
from sklearn.cluster import KMeans

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TeamClassificationSystem:
    """Advanced team classification and ball tracking system"""
    
    def __init__(self):
        # Load YOLO model
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Team classification
        self.team_colors = {
            'team_a': (255, 0, 0),      # Red
            'team_a_goalkeeper': (200, 0, 0),  # Dark Red
            'team_b': (0, 0, 255),      # Blue
            'team_b_goalkeeper': (0, 0, 200),  # Dark Blue
            'referee': (0, 255, 0),     # Green
            'ball': (255, 255, 0),      # Yellow
            'other': (128, 128, 128),   # Gray
            'staff': (255, 0, 255)      # Magenta
        }
        
        # Team color learning
        self.team_a_colors = deque(maxlen=100)
        self.team_b_colors = deque(maxlen=100)
        self.referee_colors = deque(maxlen=50)
        
        # Ball tracking
        self.ball_history = deque(maxlen=30)
        self.ball_radius = 15
        
        # Color detection for ball
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 30, 255])
        
        logger.info("✅ Team Classification System initialized")
    
    def extract_dominant_colors(self, frame, bbox):
        """Extract dominant colors from a bounding box region"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return []
        
        # Extract region
        region = frame[y1:y2, x1:x2]
        
        if region.size == 0:
            return []
        
        # Reshape for clustering
        pixels = region.reshape(-1, 3)
        
        # Sample pixels for efficiency
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels = pixels[indices]
        
        # K-means clustering to find dominant colors
        if len(pixels) > 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get dominant colors
            colors = kmeans.cluster_centers_.astype(int)
            return colors.tolist()
        
        return []
    
    def classify_team_by_color(self, colors, position):
        """Classify team based on dominant colors and position"""
        if not colors:
            return 'other'
        
        # Convert to HSV for better color analysis
        hsv_colors = []
        for color in colors:
            bgr_color = np.uint8([[color]])
            hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
            hsv_colors.append(hsv_color)
        
        # Analyze colors
        dominant_color = colors[0]  # Most dominant color
        b, g, r = dominant_color
        
        # Team classification logic
        if r > g and r > b and r > 100:  # Red dominant
            if len(self.team_a_colors) < 10:  # Learning phase
                self.team_a_colors.append(dominant_color)
            return 'team_a'
        
        elif b > g and b > r and b > 100:  # Blue dominant
            if len(self.team_b_colors) < 10:  # Learning phase
                self.team_b_colors.append(dominant_color)
            return 'team_b'
        
        elif g > r and g > b and g > 100:  # Green dominant (referee)
            if len(self.referee_colors) < 10:  # Learning phase
                self.referee_colors.append(dominant_color)
            return 'referee'
        
        # Use learned colors if available
        if len(self.team_a_colors) > 5:
            for learned_color in self.team_a_colors:
                if self.color_similarity(dominant_color, learned_color) > 0.8:
                    return 'team_a'
        
        if len(self.team_b_colors) > 5:
            for learned_color in self.team_b_colors:
                if self.color_similarity(dominant_color, learned_color) > 0.8:
                    return 'team_b'
        
        if len(self.referee_colors) > 3:
            for learned_color in self.referee_colors:
                if self.color_similarity(dominant_color, learned_color) > 0.8:
                    return 'referee'
        
        return 'other'
    
    def color_similarity(self, color1, color2):
        """Calculate color similarity"""
        diff = np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))
        return 1 - (diff / 441.67)  # Normalize by max possible difference
    
    def is_goalkeeper(self, bbox, position, team):
        """Determine if a player is a goalkeeper based on position"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Goalkeepers are usually in goal areas
        frame_height = 720  # Assuming standard height
        frame_width = 1280  # Assuming standard width
        
        # Goal areas (top and bottom 20% of frame)
        in_goal_area = center_y < frame_height * 0.2 or center_y > frame_height * 0.8
        
        # Also check if near goal posts (left/right edges)
        near_goal_post = center_x < frame_width * 0.1 or center_x > frame_width * 0.9
        
        return in_goal_area or near_goal_post
    
    def detect_ball(self, frame):
        """Detect ball using color and motion"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for white/light colors
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:  # Ball size range
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if 5 < radius < 25:  # Reasonable ball radius
                    return {
                        'center': (int(x), int(y)),
                        'radius': int(radius),
                        'confidence': min(1.0, area / 200)
                    }
        
        return None
    
    def classify_players(self, frame, detections):
        """Classify all detected players"""
        classified_players = []
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Extract colors from player region
            colors = self.extract_dominant_colors(frame, bbox)
            
            # Classify team
            team = self.classify_team_by_color(colors, bbox)
            
            # Check if goalkeeper
            is_gk = self.is_goalkeeper(bbox, bbox, team)
            
            # Determine final classification
            if is_gk and team in ['team_a', 'team_b']:
                final_class = f"{team}_goalkeeper"
            else:
                final_class = team
            
            classified_players.append({
                'bbox': bbox,
                'confidence': confidence,
                'class': final_class,
                'colors': colors
            })
        
        return classified_players
    
    def draw_classified_players(self, frame, players):
        """Draw players with proper team colors and labels"""
        for player in players:
            bbox = player['bbox']
            confidence = player['confidence']
            class_name = player['class']
            
            x1, y1, x2, y2 = bbox
            color = self.team_colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1-30), (x1+label_size[0]+10, y1), color, -1)
            cv2.putText(frame, label, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_ball_tracking(self, frame, ball_info):
        """Draw ball with red circle tracking"""
        if ball_info:
            center = ball_info['center']
            radius = ball_info['radius']
            confidence = ball_info['confidence']
            
            # Draw red circle around ball
            cv2.circle(frame, center, radius + 5, (0, 0, 255), 3)  # Red circle
            cv2.circle(frame, center, radius, (0, 255, 255), 2)    # Yellow inner circle
            
            # Draw ball label
            cv2.putText(frame, f"Ball: {confidence:.2f}", 
                       (center[0] - 30, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw ball trail
            if len(self.ball_history) > 1:
                points = list(self.ball_history)
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame, points[i-1], points[i], (0, 0, 255), thickness)
            
            # Add to history
            self.ball_history.append(center)
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame and return classified results"""
        # Detect people using YOLO
        results = self.yolo_model(frame, conf=0.3, verbose=False)
        
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if cls_id == 0:  # 'person' class
                    x1, y1, x2, y2 = map(int, box)
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf
                    })
        
        # Classify players
        classified_players = self.classify_players(frame, detections)
        
        # Detect ball
        ball_info = self.detect_ball(frame)
        
        return classified_players, ball_info

def main():
    """Main function for team classification demo"""
    logger.info("🚀 Godseye AI - Team Classification System")
    logger.info("=" * 50)
    
    # Initialize system
    classifier = TeamClassificationSystem()
    
    # Open video
    cap = cv2.VideoCapture('madrid_vs_city.mp4')
    if not cap.isOpened():
        logger.error("❌ Cannot open video: madrid_vs_city.mp4")
        return
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"📊 Video Info: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
    
    # Setup display window
    cv2.namedWindow('Godseye AI - Team Classification', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Godseye AI - Team Classification', 1280, 720)
    
    logger.info("🎬 Starting team classification demo...")
    logger.info("📝 Press 'q' to quit, 'p' to pause, 's' to save screenshot")
    logger.info("🔴 Red = Team A, 🔵 Blue = Team B, 🟢 Green = Referee")
    logger.info("🔴 Red Circle = Ball tracking")
    
    frame_count = 0
    paused = False
    team_stats = defaultdict(int)
    ball_detections = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            ret = True
        
        if ret:
            # Process frame
            players, ball_info = classifier.process_frame(frame)
            
            # Update statistics
            for player in players:
                team_stats[player['class']] += 1
            
            if ball_info:
                ball_detections += 1
            
            # Draw visualizations
            frame = classifier.draw_classified_players(frame, players)
            frame = classifier.draw_ball_tracking(frame, ball_info)
            
            # Add info overlay
            info_y = 30
            cv2.putText(frame, "Godseye AI - Team Classification", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            info_y += 35
            
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 30
            
            cv2.putText(frame, f"Players: {len(players)}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 30
            
            cv2.putText(frame, f"Ball Detections: {ball_detections}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            info_y += 30
            
            # Show team statistics
            cv2.putText(frame, "Team Stats:", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            info_y += 25
            
            for team, count in team_stats.items():
                if count > 0:
                    team_text = f"  {team}: {count}"
                    cv2.putText(frame, team_text, (10, info_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    info_y += 20
            
            # Add controls
            cv2.putText(frame, "Controls: 'q'=quit, 'p'=pause, 's'=screenshot", 
                       (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            # Add pause indicator
            if paused:
                cv2.putText(frame, "PAUSED", (width-150, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Show frame
            cv2.imshow('Godseye AI - Team Classification', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("🛑 Quit requested by user")
                break
            elif key == ord('p'):
                paused = not paused
                logger.info(f"⏸️  {'Paused' if paused else 'Resumed'}")
            elif key == ord('s'):
                screenshot_path = f"team_classification_screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, frame)
                logger.info(f"📸 Screenshot saved: {screenshot_path}")
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"📈 Progress: {progress:.1f}% - Ball: {ball_detections}, Teams: {dict(team_stats)}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    logger.info("=" * 50)
    logger.info("✅ Team classification demo completed!")
    logger.info(f"📊 Final team statistics: {dict(team_stats)}")
    logger.info(f"📈 Ball detections: {ball_detections}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
