#!/usr/bin/env python3
"""
===============================================================================
GODSEYE AI - ENHANCED INFERENCE SYSTEM
===============================================================================

Comprehensive inference system with all advanced features:
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
import json
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import threading
from datetime import datetime, timedelta

# ML Libraries
import torch
from ultralytics import YOLO
import mediapipe as mp

# Custom modules
sys.path.append(str(Path(__file__).parent))
from comprehensive_training_pipeline import (
    PoseEstimator, EventDetector, TacticalAnalyzer, JerseyNumberRecognizer
)

class EnhancedInferenceSystem:
    """
    Enhanced inference system with all advanced features
    """
    
    def __init__(self, model_path: str = "models/best_model.pt"):
        self.model_path = model_path
        self.model = None
        self.pose_estimator = PoseEstimator()
        self.event_detector = EventDetector()
        self.tactical_analyzer = TacticalAnalyzer()
        self.jersey_recognizer = JerseyNumberRecognizer()
        
        # Match data
        self.match_data = {
            'team_a': {'name': 'Team A', 'score': 0, 'possession': 0, 'shots': 0, 'passes': 0},
            'team_b': {'name': 'Team B', 'score': 0, 'possession': 0, 'shots': 0, 'passes': 0},
            'match_time': '00:00',
            'events': [],
            'players': {},
            'ball_position': None,
            'formation_a': '4-4-2',
            'formation_b': '4-3-3'
        }
        
        # Detection history for tracking
        self.detection_history = []
        self.frame_count = 0
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"✅ Model loaded from {self.model_path}")
            else:
                # Load default YOLOv8 model
                self.model = YOLO('yolov8n.pt')
                print("⚠️ Using default YOLOv8 model")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = YOLO('yolov8n.pt')
    
    def process_video(self, video_path: str, output_path: str = None):
        """Process video with comprehensive analysis"""
        print(f"🎥 Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"📊 Video Info: {width}x{height}, {fps} FPS, {duration:.1f}s")
        
        # Setup output video
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None
        
        # Process frames
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            self.frame_count = frame_count
            
            # Process frame
            processed_frame = self.process_frame(frame, frame_count, fps)
            
            # Write output
            if out:
                out.write(processed_frame)
            
            # Display progress
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / frame_count) * (total_frames - frame_count)
                print(f"📈 Progress: {progress:.1f}% | ETA: {eta:.1f}s")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        
        # Save results
        self.save_results(output_path)
        
        print(f"✅ Video processing complete! Processed {frame_count} frames")
    
    def process_frame(self, frame: np.ndarray, frame_number: int, fps: int) -> np.ndarray:
        """Process a single frame with all features"""
        # Update match time
        self.update_match_time(frame_number, fps)
        
        # Run detection
        detections = self.run_detection(frame)
        
        # Run pose estimation
        pose_data = self.run_pose_estimation(frame, detections)
        
        # Run event detection
        events = self.run_event_detection(detections, frame_number)
        
        # Run tactical analysis
        tactical_data = self.run_tactical_analysis(detections)
        
        # Run jersey number recognition
        jersey_data = self.run_jersey_recognition(frame, detections)
        
        # Update match data
        self.update_match_data(detections, events, tactical_data, jersey_data)
        
        # Draw annotations
        annotated_frame = self.draw_annotations(frame, detections, pose_data, events)
        
        # Draw UI elements
        annotated_frame = self.draw_ui_elements(annotated_frame)
        
        return annotated_frame
    
    def run_detection(self, frame: np.ndarray) -> List[Dict]:
        """Run object detection"""
        if not self.model:
            return []
        
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Map class ID to class name
                    class_names = ['team_a_player', 'team_a_goalkeeper', 'team_b_player', 
                                 'team_b_goalkeeper', 'referee', 'ball', 'outlier', 'staff']
                    class_name = class_names[class_id] if class_id < len(class_names) else 'unknown'
                    
                    # Determine team
                    team = 'A' if 'team_a' in class_name else 'B' if 'team_b' in class_name else 'unknown'
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class': class_name,
                        'class_id': class_id,
                        'team': team,
                        'id': len(detections)  # Simple ID assignment
                    }
                    detections.append(detection)
        
        return detections
    
    def run_pose_estimation(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """Run pose estimation on detected players"""
        pose_data = {}
        
        for detection in detections:
            if 'player' in detection['class']:
                # Extract player crop
                x1, y1, x2, y2 = detection['bbox']
                player_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                
                if player_crop.size > 0:
                    # Run pose estimation
                    pose_result = self.pose_estimator.estimate_pose(player_crop)
                    pose_data[detection['id']] = pose_result
        
        return pose_data
    
    def run_event_detection(self, detections: List[Dict], frame_number: int) -> List[Dict]:
        """Run event detection"""
        frame_data = {
            'timestamp': frame_number,
            'detections': detections
        }
        
        events = self.event_detector.detect_events(detections, frame_data)
        
        # Add events to match data
        for event in events:
            self.match_data['events'].append(event)
        
        return events
    
    def run_tactical_analysis(self, detections: List[Dict]) -> Dict:
        """Run tactical analysis"""
        tactical_data = {
            'formation_a': self.tactical_analyzer.analyze_formation(detections),
            'formation_b': self.tactical_analyzer.analyze_formation(detections),
            'possession': self.tactical_analyzer.analyze_possession(detections),
            'heatmap': self.tactical_analyzer.generate_heatmap(detections),
            'passing_patterns': self.tactical_analyzer.analyze_passing_patterns(detections)
        }
        
        return tactical_data
    
    def run_jersey_recognition(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Run jersey number recognition"""
        jersey_data = self.jersey_recognizer.recognize_numbers(detections, frame)
        return jersey_data
    
    def update_match_time(self, frame_number: int, fps: int):
        """Update match time based on frame number"""
        seconds = frame_number / fps
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        self.match_data['match_time'] = f"{minutes:02d}:{seconds:02d}"
    
    def update_match_data(self, detections: List[Dict], events: List[Dict], 
                         tactical_data: Dict, jersey_data: List[Dict]):
        """Update match data with new information"""
        # Update possession
        if 'possession' in tactical_data:
            self.match_data['team_a']['possession'] = tactical_data['possession']['team_a'] * 100
            self.match_data['team_b']['possession'] = tactical_data['possession']['team_b'] * 100
        
        # Update formations
        if 'formation_a' in tactical_data:
            self.match_data['formation_a'] = tactical_data['formation_a']
        if 'formation_b' in tactical_data:
            self.match_data['formation_b'] = tactical_data['formation_b']
        
        # Update scores based on events
        for event in events:
            if event['type'] == 'goal':
                if event.get('team') == 'A':
                    self.match_data['team_a']['score'] += 1
                elif event.get('team') == 'B':
                    self.match_data['team_b']['score'] += 1
        
        # Update player data
        for detection in detections:
            if 'player' in detection['class']:
                player_id = detection['id']
                if player_id not in self.match_data['players']:
                    self.match_data['players'][player_id] = {
                        'team': detection['team'],
                        'class': detection['class'],
                        'jersey_number': None,
                        'positions': [],
                        'events': []
                    }
                
                # Add position
                center_x = (detection['bbox'][0] + detection['bbox'][2]) / 2
                center_y = (detection['bbox'][1] + detection['bbox'][3]) / 2
                self.match_data['players'][player_id]['positions'].append([center_x, center_y])
        
        # Update jersey numbers
        for jersey in jersey_data:
            player_id = jersey['player_id']
            if player_id in self.match_data['players']:
                self.match_data['players'][player_id]['jersey_number'] = jersey['jersey_number']
    
    def draw_annotations(self, frame: np.ndarray, detections: List[Dict], 
                        pose_data: Dict, events: List[Dict]) -> np.ndarray:
        """Draw all annotations on frame"""
        annotated_frame = frame.copy()
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            team = detection['team']
            
            # Choose color based on team/class
            if team == 'A':
                color = (255, 0, 0)  # Blue
            elif team == 'B':
                color = (0, 0, 255)  # Red
            elif 'referee' in class_name:
                color = (0, 255, 255)  # Yellow
            elif 'ball' in class_name:
                color = (0, 255, 0)  # Green
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            if detection['id'] in self.match_data['players']:
                jersey_num = self.match_data['players'][detection['id']].get('jersey_number')
                if jersey_num:
                    label = f"#{jersey_num} {class_name} {confidence:.2f}"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw pose keypoints
        for player_id, pose_result in pose_data.items():
            if pose_result['keypoints']:
                # Draw pose on frame (simplified)
                for keypoint in pose_result['keypoints']:
                    x = int(keypoint['x'] * frame.shape[1])
                    y = int(keypoint['y'] * frame.shape[0])
                    cv2.circle(annotated_frame, (x, y), 3, (255, 255, 0), -1)
        
        # Draw event alerts
        for event in events:
            self.draw_event_alert(annotated_frame, event)
        
        return annotated_frame
    
    def draw_event_alert(self, frame: np.ndarray, event: Dict):
        """Draw event alert on frame"""
        alert_text = f"{event['type'].upper()}: {event['description']}"
        
        # Choose alert color
        if event['type'] == 'goal':
            color = (0, 255, 0)  # Green
        elif event['type'] == 'foul':
            color = (0, 255, 255)  # Yellow
        elif event['type'] in ['yellow_card', 'red_card']:
            color = (0, 0, 255)  # Red
        else:
            color = (255, 255, 255)  # White
        
        # Draw alert background
        alert_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.rectangle(frame, (10, 10), (10 + alert_size[0] + 20, 10 + alert_size[1] + 20), color, -1)
        
        # Draw alert text
        cv2.putText(frame, alert_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    def draw_ui_elements(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI elements (scoreboard, timer, etc.)"""
        height, width = frame.shape[:2]
        
        # Draw scoreboard background
        scoreboard_height = 80
        cv2.rectangle(frame, (0, 0), (width, scoreboard_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (width, scoreboard_height), (255, 255, 255), 2)
        
        # Draw team names and scores
        team_a_text = f"{self.match_data['team_a']['name']}: {self.match_data['team_a']['score']}"
        team_b_text = f"{self.match_data['team_b']['name']}: {self.match_data['team_b']['score']}"
        
        cv2.putText(frame, team_a_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, team_b_text, (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw match time
        time_text = f"Time: {self.match_data['match_time']}"
        cv2.putText(frame, time_text, (width // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw possession
        possession_text = f"Possession: {self.match_data['team_a']['possession']:.0f}% - {self.match_data['team_b']['possession']:.0f}%"
        cv2.putText(frame, possession_text, (width // 2 - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw formations
        formation_text = f"Formations: {self.match_data['formation_a']} vs {self.match_data['formation_b']}"
        cv2.putText(frame, formation_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def save_results(self, output_path: str = None):
        """Save analysis results"""
        results = {
            'match_data': self.match_data,
            'total_frames': self.frame_count,
            'analysis_time': datetime.now().isoformat(),
            'model_path': self.model_path
        }
        
        # Save to JSON
        results_path = output_path.replace('.mp4', '_results.json') if output_path else 'analysis_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📊 Results saved to: {results_path}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("🏆 ANALYSIS SUMMARY")
        print("="*60)
        print(f"⏱️ Match Time: {self.match_data['match_time']}")
        print(f"🏆 Final Score: {self.match_data['team_a']['name']} {self.match_data['team_a']['score']} - {self.match_data['team_b']['score']} {self.match_data['team_b']['name']}")
        print(f"📊 Possession: {self.match_data['team_a']['possession']:.1f}% - {self.match_data['team_b']['possession']:.1f}%")
        print(f"🎯 Formations: {self.match_data['formation_a']} vs {self.match_data['formation_b']}")
        print(f"📈 Total Events: {len(self.match_data['events'])}")
        print(f"👥 Players Detected: {len(self.match_data['players'])}")
        print(f"🎬 Frames Processed: {self.frame_count}")
        
        # Event breakdown
        event_counts = {}
        for event in self.match_data['events']:
            event_type = event['type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        if event_counts:
            print("\n📋 Event Breakdown:")
            for event_type, count in event_counts.items():
                print(f"   {event_type}: {count}")
        
        print("="*60)

def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Football Analytics Inference System')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--model', type=str, default='models/best_model.pt', help='Model path')
    
    args = parser.parse_args()
    
    # Create inference system
    inference_system = EnhancedInferenceSystem(args.model)
    
    # Process video
    inference_system.process_video(args.video, args.output)

if __name__ == "__main__":
    main()

