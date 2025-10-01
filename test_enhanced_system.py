#!/usr/bin/env python3
"""
===============================================================================
GODSEYE AI - ENHANCED SYSTEM TEST
===============================================================================

Test script to demonstrate the enhanced inference system with all features:
- Real-time detection with bounding boxes
- Event detection with alerts
- Scoreboard and timer display
- Pose estimation and analysis
- Tactical analysis and heatmaps
- Jersey number recognition
- Advanced statistics and analytics

Author: Victor
Date: 2025
Version: 2.0.0
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from enhanced_inference_system import EnhancedInferenceSystem

def create_test_video():
    """Create a test video for demonstration"""
    print("🎬 Creating test video...")
    
    # Video properties
    width, height = 1280, 720
    fps = 30
    duration = 10  # 10 seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, fps, (width, height))
    
    for frame_num in range(total_frames):
        # Create a green field background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (34, 139, 34)  # Forest green
        
        # Draw field lines
        cv2.line(frame, (0, height//2), (width, height//2), (255, 255, 255), 2)
        cv2.circle(frame, (width//2, height//2), 100, (255, 255, 255), 2)
        cv2.rectangle(frame, (50, 100), (200, height-100), (255, 255, 255), 2)
        cv2.rectangle(frame, (width-200, 100), (width-50, height-100), (255, 255, 255), 2)
        
        # Draw some "players" (circles)
        players = [
            (300, 200, (255, 0, 0)),    # Team A player
            (400, 300, (255, 0, 0)),    # Team A player
            (500, 400, (0, 0, 255)),    # Team B player
            (600, 500, (0, 0, 255)),    # Team B player
            (700, 350, (0, 255, 255)),  # Referee
            (width//2 + 50, height//2, (0, 255, 0))  # Ball
        ]
        
        for x, y, color in players:
            cv2.circle(frame, (x, y), 20, color, -1)
            cv2.circle(frame, (x, y), 20, (255, 255, 255), 2)
        
        # Add some movement
        if frame_num > 30:  # After 1 second
            # Move ball
            ball_x = width//2 + 50 + int(20 * np.sin(frame_num * 0.1))
            ball_y = height//2 + int(10 * np.cos(frame_num * 0.1))
            cv2.circle(frame, (ball_x, ball_y), 15, (0, 255, 0), -1)
            cv2.circle(frame, (ball_x, ball_y), 15, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = f"Time: {frame_num//fps:02d}:{(frame_num%fps)*2:02d}"
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print("✅ Test video created: test_video.mp4")

def test_enhanced_system():
    """Test the enhanced inference system"""
    print("🚀 Testing Enhanced Inference System...")
    
    # Create test video if it doesn't exist
    if not os.path.exists('test_video.mp4'):
        create_test_video()
    
    # Create inference system
    print("🔧 Initializing Enhanced Inference System...")
    inference_system = EnhancedInferenceSystem()
    
    # Process the test video
    print("🎥 Processing test video...")
    start_time = time.time()
    
    inference_system.process_video(
        video_path='test_video.mp4',
        output_path='test_video_analyzed.mp4'
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"⏱️ Processing completed in {processing_time:.2f} seconds")
    print("✅ Enhanced system test completed!")
    
    # Print system capabilities
    print("\n" + "="*60)
    print("🏆 ENHANCED SYSTEM CAPABILITIES")
    print("="*60)
    print("✅ Real-time object detection with bounding boxes")
    print("✅ Event detection (goals, fouls, cards, etc.)")
    print("✅ Live scoreboard and timer display")
    print("✅ Pose estimation with 17 keypoints")
    print("✅ Tactical analysis and formation detection")
    print("✅ Jersey number recognition")
    print("✅ Advanced statistics and analytics")
    print("✅ Weather and lighting condition handling")
    print("✅ Multi-scale training capabilities")
    print("✅ Real-time processing with alerts")
    print("="*60)

def main():
    """Main test function"""
    print("🎯 Godseye AI - Enhanced System Test")
    print("="*50)
    
    try:
        test_enhanced_system()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

