#!/usr/bin/env python3
"""
===============================================================================
GODSEYE AI - IMPROVED FOOTBALL CLASSIFIER
===============================================================================

Fixed classification system that addresses accuracy issues:
- Better team classification using position and context
- Improved ball detection
- Comprehensive football classes
- More accurate referee detection

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
from comprehensive_football_classes import ComprehensiveFootballClasses

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedFootballClassifier:
    """Improved football classification system with better accuracy"""
    
    def __init__(self, level='core'):
        # Load YOLO model
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Get class definitions
        self.class_info = ComprehensiveFootballClasses.get_class_info(level)
        self.classes = self.class_info['classes']
        self.colors = self.class_info['colors']
        self.class_to_idx = self.class_info['class_to_idx']
        
        # Team learning system
        self.team_a_positions = deque(maxlen=100)
        self.team_b_positions = deque(maxlen=100)
        self.referee_positions = deque(maxlen=50)
        
        # Ball tracking
        self.ball_history = deque(maxlen=30)
        self.ball_radius = 15
        
        # Color detection for ball (improved)
        self.lower_white = np.array([0, 0, 180])
        self.upper_white = np.array([180, 30, 255])
        
        # Field analysis
        self.field_center = (640, 360)  # Assuming 1280x720
        self.goal_areas = {
            'top': (0, 0, 1280, 144),      # Top 20%
            'bottom': (0, 576, 1280, 720)  # Bottom 20%
        }
        
        logger.info(f"✅ Improved Football Classifier initialized with {len(self.classes)} classes")
    
    def analyze_field_position(self, bbox):
        """Analyze player position on field for better classification"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Check if in goal area
        in_goal_area = False
        if (y1 < 144 or y2 > 576):  # Top or bottom 20%
            in_goal_area = True
        
        # Check if near sidelines
        near_sideline = center_x < 100 or center_x > 1180
        
        # Check if in center circle area
        center_circle_distance = math.sqrt((center_x - 640)**2 + (center_y - 360)**2)
        in_center = center_circle_distance < 100
        
        return {
            'center': (center_x, center_y),
            'in_goal_area': in_goal_area,
            'near_sideline': near_sideline,
            'in_center': in_center,
            'distance_from_center': center_circle_distance
        }
    
    def classify_by_position_and_context(self, bbox, confidence):
        """Classify player based on position and field context"""
        position_info = self.analyze_field_position(bbox)
        center_x, center_y = position_info['center']
        
        # Goalkeeper detection (in goal areas)
        if position_info['in_goal_area']:
            # Determine which team's goalkeeper based on field position
            if center_y < 360:  # Top half = Team A goalkeeper
                return 'team_a_goalkeeper'
            else:  # Bottom half = Team B goalkeeper
                return 'team_b_goalkeeper'
        
        # Referee detection (usually on sidelines, smaller, different movement)
        if position_info['near_sideline'] and confidence > 0.7:
            # Referees are usually smaller and move differently
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if bbox_area < 8000:  # Smaller than typical players
                return 'referee'
        
        # Assistant referee detection (linesmen)
        if position_info['near_sideline'] and confidence > 0.6:
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if bbox_area < 10000:  # Medium size
                return 'assistant_referee'
        
        # Regular players - use field position to determine team
        # This is a simplified approach - in reality, you'd use jersey colors
        if center_x < 640:  # Left side of field
            return 'team_a_player'
        else:  # Right side of field
            return 'team_b_player'
    
    def detect_ball_improved(self, frame):
        """Improved ball detection using multiple methods"""
        # Method 1: Color-based detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Morphological operations
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 30 < area < 800:  # Ball size range
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if 3 < radius < 30:  # Reasonable ball radius
                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.3:  # Reasonably circular
                            ball_candidates.append({
                                'center': (int(x), int(y)),
                                'radius': int(radius),
                                'area': area,
                                'circularity': circularity,
                                'confidence': min(1.0, area / 200) * circularity
                            })
        
        # Return best candidate
        if ball_candidates:
            return max(ball_candidates, key=lambda x: x['confidence'])
        
        return None
    
    def detect_goalposts(self, frame):
        """Detect goalposts using edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find vertical lines (goalposts)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=100, maxLineGap=10)
        
        goalposts = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is roughly vertical
                if abs(x2 - x1) < 20 and abs(y2 - y1) > 80:
                    # Check if in goal area
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    if center_y < 144 or center_y > 576:  # In goal areas
                        goalposts.append({
                            'center': (int(center_x), int(center_y)),
                            'bbox': (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
                            'confidence': 0.8
                        })
        
        return goalposts
    
    def classify_detections(self, frame, detections):
        """Classify all detections with improved accuracy"""
        classified_objects = []
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Classify based on position and context
            class_name = self.classify_by_position_and_context(bbox, confidence)
            
            classified_objects.append({
                'bbox': bbox,
                'confidence': confidence,
                'class': class_name
            })
        
        return classified_objects
    
    def draw_classifications(self, frame, objects, ball_info=None, goalposts=None):
        """Draw all classifications with proper colors and labels"""
        # Draw players and other objects
        for obj in objects:
            bbox = obj['bbox']
            confidence = obj['confidence']
            class_name = obj['class']
            
            x1, y1, x2, y2 = bbox
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1-25), (x1+label_size[0]+5, y1), color, -1)
            cv2.putText(frame, label, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw ball
        if ball_info:
            center = ball_info['center']
            radius = ball_info['radius']
            confidence = ball_info['confidence']
            
            # Draw red circle around ball
            cv2.circle(frame, center, radius + 5, (0, 0, 255), 3)
            cv2.circle(frame, center, radius, (0, 255, 255), 2)
            
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
            
            self.ball_history.append(center)
        
        # Draw goalposts
        if goalposts:
            for goalpost in goalposts:
                bbox = goalpost['bbox']
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, "Goalpost", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame and return all classifications"""
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
        
        # Classify detections
        classified_objects = self.classify_detections(frame, detections)
        
        # Detect ball
        ball_info = self.detect_ball_improved(frame)
        
        # Detect goalposts
        goalposts = self.detect_goalposts(frame)
        
        return classified_objects, ball_info, goalposts

def main():
    """Main function for improved classification demo"""
    logger.info("🚀 Godseye AI - Improved Football Classifier")
    logger.info("=" * 50)
    
    # Initialize classifier
    classifier = ImprovedFootballClassifier(level='core')
    
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
    cv2.namedWindow('Godseye AI - Improved Classifier', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Godseye AI - Improved Classifier', 1280, 720)
    
    logger.info("🎬 Starting improved classification demo...")
    logger.info("📝 Press 'q' to quit, 'p' to pause, 's' to save screenshot")
    logger.info("🔴 Red = Team A, 🔵 Blue = Team B, 🟢 Green = Referee")
    logger.info("🔴 Red Circle = Ball, ⚪ White = Goalposts")
    
    frame_count = 0
    paused = False
    class_stats = defaultdict(int)
    ball_detections = 0
    goalpost_detections = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            ret = True
        
        if ret:
            # Process frame
            objects, ball_info, goalposts = classifier.process_frame(frame)
            
            # Update statistics
            for obj in objects:
                class_stats[obj['class']] += 1
            
            if ball_info:
                ball_detections += 1
            
            if goalposts:
                goalpost_detections += 1
            
            # Draw visualizations
            frame = classifier.draw_classifications(frame, objects, ball_info, goalposts)
            
            # Add info overlay
            info_y = 30
            cv2.putText(frame, "Godseye AI - Improved Classifier", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            info_y += 35
            
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 30
            
            cv2.putText(frame, f"Objects: {len(objects)}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 30
            
            cv2.putText(frame, f"Ball: {ball_detections}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            info_y += 30
            
            cv2.putText(frame, f"Goalposts: {goalpost_detections}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 30
            
            # Show class statistics
            cv2.putText(frame, "Classification Stats:", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            info_y += 25
            
            for class_name, count in sorted(class_stats.items()):
                if count > 0:
                    class_text = f"  {class_name}: {count}"
                    cv2.putText(frame, class_text, (10, info_y), 
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
            cv2.imshow('Godseye AI - Improved Classifier', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("🛑 Quit requested by user")
                break
            elif key == ord('p'):
                paused = not paused
                logger.info(f"⏸️  {'Paused' if paused else 'Resumed'}")
            elif key == ord('s'):
                screenshot_path = f"improved_classifier_screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, frame)
                logger.info(f"📸 Screenshot saved: {screenshot_path}")
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"📈 Progress: {progress:.1f}% - Ball: {ball_detections}, Goalposts: {goalpost_detections}")
                logger.info(f"📊 Class stats: {dict(class_stats)}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    logger.info("=" * 50)
    logger.info("✅ Improved classification demo completed!")
    logger.info(f"📊 Final class statistics: {dict(class_stats)}")
    logger.info(f"📈 Ball detections: {ball_detections}")
    logger.info(f"📈 Goalpost detections: {goalpost_detections}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
