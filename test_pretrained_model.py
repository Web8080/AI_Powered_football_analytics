#!/usr/bin/env python3
"""
===============================================================================
GODSEYE AI - PRE-TRAINED MODEL TESTER
===============================================================================

Tests the pre-trained YOLOv8 model on madrid_vs_city.mp4 to show
bounding boxes and demonstrate the system working.

Author: Victor
Date: 2025
Version: 1.0.0
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pretrained_model():
    """Test pre-trained YOLO model with real-time display"""
    
    # Load pre-trained model
    model = YOLO('yolov8n.pt')
    logger.info("✅ Loaded pre-trained YOLOv8 model")
    logger.info(f"📋 Model classes: {len(model.names)} classes")
    
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
    cv2.namedWindow('Godseye AI - Pre-trained Model Demo', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Godseye AI - Pre-trained Model Demo', 1280, 720)
    
    logger.info("🎬 Starting real-time demo...")
    logger.info("📝 Press 'q' to quit, 'p' to pause, 's' to save screenshot")
    
    frame_count = 0
    paused = False
    
    # Colors for different classes
    colors = {
        'person': (0, 255, 0),      # Green for people
        'sports ball': (255, 0, 0), # Red for ball
    }
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            ret = True
        
        if ret:
            # Run inference
            results = model(frame, conf=0.3, verbose=False)
            
            # Create annotated frame
            annotated_frame = frame.copy()
            detections = []
            
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    class_name = model.names[cls_id]
                    
                    # Only show relevant classes
                    if class_name in ['person', 'sports ball']:
                        x1, y1, x2, y2 = map(int, box)
                        color = colors.get(class_name, (255, 255, 255))
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1-30), (x1+label_size[0]+10, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        detections.append({'class': class_name, 'confidence': conf})
            
            # Add info overlay
            info_y = 30
            cv2.putText(annotated_frame, "Godseye AI - Pre-trained YOLOv8 Demo", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            info_y += 35
            
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 30
            
            cv2.putText(annotated_frame, f"Detections: {len(detections)}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 30
            
            # Show current detections
            if detections:
                cv2.putText(annotated_frame, "Current Detections:", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                info_y += 25
                
                for det in detections[:5]:  # Show max 5
                    det_text = f"  {det['class']}: {det['confidence']:.2f}"
                    cv2.putText(annotated_frame, det_text, (10, info_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    info_y += 20
            
            # Add controls
            cv2.putText(annotated_frame, "Controls: 'q'=quit, 'p'=pause, 's'=screenshot", 
                       (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            # Add pause indicator
            if paused:
                cv2.putText(annotated_frame, "PAUSED", (width-150, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Show frame
            cv2.imshow('Godseye AI - Pre-trained Model Demo', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("🛑 Quit requested by user")
                break
            elif key == ord('p'):
                paused = not paused
                logger.info(f"⏸️  {'Paused' if paused else 'Resumed'}")
            elif key == ord('s'):
                screenshot_path = f"screenshot_frame_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                logger.info(f"📸 Screenshot saved: {screenshot_path}")
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"📈 Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    logger.info("✅ Demo completed!")

if __name__ == "__main__":
    logger.info("🚀 Godseye AI - Pre-trained Model Demo")
    logger.info("=" * 50)
    test_pretrained_model()
