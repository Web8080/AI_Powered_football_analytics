#!/usr/bin/env python3
"""
================================================================================
GODSEYE AI - TEST IMPROVED MODEL
================================================================================

Author: Victor Ibhafidon
Date: January 28, 2025
Version: 1.0.0

DESCRIPTION:
Test the improved model with enhanced referee detection on real football videos.

USAGE:
    python test_improved_model.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
import json
from pathlib import Path

def test_improved_model(video_path: str, model_path: str = "models/yolov8_improved_referee.pt"):
    """Test the improved model on real football video"""
    print("🏈 Godseye AI - Testing Improved Model")
    print("=" * 45)
    
    # Load the improved model
    model = YOLO(model_path)
    print(f"✅ Loaded model: {model_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📊 Video: {total_frames} frames, {fps} FPS")
    
    # Test on sample frames
    test_frames = [0, total_frames//4, total_frames//2, 3*total_frames//4, total_frames-1]
    
    results_summary = {
        'total_detections': 0,
        'class_counts': {},
        'referee_detections': 0,
        'team_a_detections': 0,
        'team_b_detections': 0,
        'ball_detections': 0,
        'frames_tested': len(test_frames)
    }
    
    class_names = ['team_a_player', 'team_a_goalkeeper', 'team_b_player', 'team_b_goalkeeper', 
                   'referee', 'ball', 'outlier', 'staff']
    
    for i, frame_idx in enumerate(test_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        print(f"🔍 Testing frame {i+1}/{len(test_frames)} (frame {frame_idx})")
        
        # Run inference
        results = model(frame, conf=0.3)  # Lower confidence for more detections
        
        # Process results
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    class_name = class_names[class_id]
                    results_summary['total_detections'] += 1
                    
                    if class_name in results_summary['class_counts']:
                        results_summary['class_counts'][class_name] += 1
                    else:
                        results_summary['class_counts'][class_name] = 1
                    
                    # Track specific classes
                    if class_name == 'referee':
                        results_summary['referee_detections'] += 1
                    elif class_name in ['team_a_player', 'team_a_goalkeeper']:
                        results_summary['team_a_detections'] += 1
                    elif class_name in ['team_b_player', 'team_b_goalkeeper']:
                        results_summary['team_b_detections'] += 1
                    elif class_name == 'ball':
                        results_summary['ball_detections'] += 1
                    
                    print(f"  ✅ {class_name}: {conf:.2f} confidence")
        
        # Save annotated frame
        annotated_frame = results[0].plot()
        cv2.imwrite(f"test_frame_{i+1}_improved.jpg", annotated_frame)
        print(f"  💾 Saved: test_frame_{i+1}_improved.jpg")
    
    cap.release()
    
    # Print summary
    print(f"\n📊 Test Results Summary:")
    print("=" * 30)
    print(f"Total detections: {results_summary['total_detections']}")
    print(f"Frames tested: {results_summary['frames_tested']}")
    print(f"Referee detections: {results_summary['referee_detections']}")
    print(f"Team A detections: {results_summary['team_a_detections']}")
    print(f"Team B detections: {results_summary['team_b_detections']}")
    print(f"Ball detections: {results_summary['ball_detections']}")
    
    print(f"\n🎯 Class breakdown:")
    for class_name, count in results_summary['class_counts'].items():
        print(f"  {class_name}: {count}")
    
    # Save results
    with open("improved_model_test_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ Test complete! Results saved to improved_model_test_results.json")
    
    return results_summary

def main():
    """Main function"""
    video_path = "BAY_BMG.mp4"
    model_path = "models/yolov8_improved_referee.pt"
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Test the improved model
    results = test_improved_model(video_path, model_path)
    
    # Evaluate results
    if results['total_detections'] > 0:
        print(f"\n🎉 SUCCESS! Model detected {results['total_detections']} objects!")
        if results['referee_detections'] > 0:
            print(f"✅ Referee detection working! Found {results['referee_detections']} referees")
        else:
            print("⚠️ No referee detections - may need further tuning")
    else:
        print("❌ No detections found - model may need more training")

if __name__ == "__main__":
    main()
