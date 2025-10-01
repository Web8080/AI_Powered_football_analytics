#!/usr/bin/env python3
"""
===============================================================================
GODSEYE AI - IMPROVED BALL TRACKER
===============================================================================

Advanced ball tracking system that combines:
- YOLO detection for players
- Color-based ball detection
- Motion tracking
- Real-time red circle around ball

Author: Victor
Date: 2025
Version: 1.0.0
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
from collections import deque
import math

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedBallTracker:
    """Advanced ball tracking system"""
    
    def __init__(self):
        # Load YOLO model for player detection
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Ball tracking parameters
        self.ball_tracker = None
        self.ball_history = deque(maxlen=30)  # Store last 30 ball positions
        self.ball_radius = 15  # Expected ball radius
        
        # Color range for ball detection (white/light colors)
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 30, 255])
        
        # Motion detection
        self.prev_frame = None
        self.motion_threshold = 30
        
        logger.info("✅ Improved Ball Tracker initialized")
    
    def detect_ball_by_color(self, frame):
        """Detect ball using color-based detection"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for white/light colors
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:  # Ball size range
                # Get bounding circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if 5 < radius < 25:  # Reasonable ball radius
                    ball_candidates.append({
                        'center': (int(x), int(y)),
                        'radius': int(radius),
                        'area': area,
                        'confidence': min(1.0, area / 200)  # Higher area = higher confidence
                    })
        
        return ball_candidates
    
    def detect_ball_by_motion(self, frame):
        """Detect ball using motion detection"""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        
        # Threshold the difference
        _, thresh = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 300:  # Small moving objects
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if 3 < radius < 20:
                    motion_candidates.append({
                        'center': (int(x), int(y)),
                        'radius': int(radius),
                        'area': area,
                        'confidence': min(1.0, area / 100)
                    })
        
        self.prev_frame = gray
        return motion_candidates
    
    def combine_ball_detections(self, color_candidates, motion_candidates):
        """Combine color and motion detections to find the best ball candidate"""
        all_candidates = color_candidates + motion_candidates
        
        if not all_candidates:
            return None
        
        # Score candidates based on multiple factors
        best_candidate = None
        best_score = 0
        
        for candidate in all_candidates:
            # Base score from confidence
            score = candidate['confidence']
            
            # Bonus for circular shape (if we had shape analysis)
            # Bonus for being in expected ball area (center of field)
            center = candidate['center']
            field_center_x = 640  # Assuming 1280 width
            field_center_y = 360  # Assuming 720 height
            
            distance_from_center = math.sqrt((center[0] - field_center_x)**2 + (center[1] - field_center_y)**2)
            center_bonus = max(0, 1 - distance_from_center / 400)  # Bonus for being near center
            score += center_bonus * 0.3
            
            # Bonus for consistent size
            if 8 < candidate['radius'] < 18:
                score += 0.2
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate if best_score > 0.3 else None
    
    def track_ball(self, frame):
        """Main ball tracking function"""
        # Detect ball using multiple methods
        color_candidates = self.detect_ball_by_color(frame)
        motion_candidates = self.detect_ball_by_motion(frame)
        
        # Combine detections
        ball_candidate = self.combine_ball_detections(color_candidates, motion_candidates)
        
        if ball_candidate:
            # Add to history
            self.ball_history.append(ball_candidate['center'])
            return ball_candidate
        
        return None
    
    def draw_ball_tracking(self, frame, ball_info):
        """Draw ball tracking visualization"""
        if ball_info:
            center = ball_info['center']
            radius = ball_info['radius']
            confidence = ball_info['confidence']
            
            # Draw red circle around ball
            cv2.circle(frame, center, radius + 5, (0, 0, 255), 3)  # Red circle
            cv2.circle(frame, center, radius, (0, 255, 255), 2)    # Yellow inner circle
            
            # Draw confidence
            cv2.putText(frame, f"Ball: {confidence:.2f}", 
                       (center[0] - 30, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw ball trail
            if len(self.ball_history) > 1:
                points = list(self.ball_history)
                for i in range(1, len(points)):
                    alpha = i / len(points)  # Fade effect
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame, points[i-1], points[i], (0, 0, 255), thickness)
        
        return frame
    
    def detect_players(self, frame):
        """Detect players using YOLO"""
        results = self.yolo_model(frame, conf=0.3, verbose=False)
        
        players = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if cls_id == 0:  # 'person' class
                    x1, y1, x2, y2 = map(int, box)
                    players.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf
                    })
        
        return players
    
    def draw_players(self, frame, players):
        """Draw player bounding boxes"""
        for player in players:
            x1, y1, x2, y2 = player['bbox']
            conf = player['confidence']
            
            # Draw green rectangle for players
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Player: {conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

def main():
    """Main function for real-time ball tracking demo"""
    logger.info("🚀 Godseye AI - Improved Ball Tracker")
    logger.info("=" * 50)
    
    # Initialize tracker
    tracker = ImprovedBallTracker()
    
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
    cv2.namedWindow('Godseye AI - Ball Tracking Demo', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Godseye AI - Ball Tracking Demo', 1280, 720)
    
    logger.info("🎬 Starting ball tracking demo...")
    logger.info("📝 Press 'q' to quit, 'p' to pause, 's' to save screenshot")
    logger.info("🔴 Red circle = Ball tracking")
    logger.info("🟢 Green boxes = Players")
    
    frame_count = 0
    paused = False
    ball_detections = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            ret = True
        
        if ret:
            # Track ball
            ball_info = tracker.track_ball(frame)
            if ball_info:
                ball_detections += 1
            
            # Detect players
            players = tracker.detect_players(frame)
            
            # Draw visualizations
            frame = tracker.draw_ball_tracking(frame, ball_info)
            frame = tracker.draw_players(frame, players)
            
            # Add info overlay
            info_y = 30
            cv2.putText(frame, "Godseye AI - Ball Tracking Demo", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            info_y += 35
            
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 30
            
            cv2.putText(frame, f"Ball Detections: {ball_detections}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            info_y += 30
            
            cv2.putText(frame, f"Players: {len(players)}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += 30
            
            if ball_info:
                cv2.putText(frame, f"Ball Confidence: {ball_info['confidence']:.2f}", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Add controls
            cv2.putText(frame, "Controls: 'q'=quit, 'p'=pause, 's'=screenshot", 
                       (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            # Add pause indicator
            if paused:
                cv2.putText(frame, "PAUSED", (width-150, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Show frame
            cv2.imshow('Godseye AI - Ball Tracking Demo', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("🛑 Quit requested by user")
                break
            elif key == ord('p'):
                paused = not paused
                logger.info(f"⏸️  {'Paused' if paused else 'Resumed'}")
            elif key == ord('s'):
                screenshot_path = f"ball_tracking_screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, frame)
                logger.info(f"📸 Screenshot saved: {screenshot_path}")
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"📈 Progress: {progress:.1f}% - Ball detections: {ball_detections}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    logger.info("=" * 50)
    logger.info("✅ Ball tracking demo completed!")
    logger.info(f"📊 Total ball detections: {ball_detections}")
    logger.info(f"📈 Detection rate: {(ball_detections/frame_count)*100:.1f}%")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
