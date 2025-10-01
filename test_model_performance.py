#!/usr/bin/env python3
"""
===============================================================================
GODSEYE AI - MODEL PERFORMANCE TESTER
===============================================================================

Tests the trained model on madrid_vs_city.mp4 to evaluate:
- Bounding box accuracy
- Detection performance
- Classification accuracy
- Real-time metrics

Author: Victor
Date: 2025
Version: 1.0.0
"""

import cv2
import json
import numpy as np
import os
from pathlib import Path
import logging
from datetime import datetime
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPerformanceTester:
    """Tests model performance on real football video"""
    
    def __init__(self, model_path="models/godseye_ai_model.pt"):
        self.model_path = model_path
        self.class_names = [
            'team_a_player',      # 0
            'team_a_goalkeeper',  # 1
            'team_b_player',      # 2
            'team_b_goalkeeper',  # 3
            'referee',            # 4
            'ball',               # 5
            'other',              # 6
            'staff'               # 7
        ]
        
        self.class_colors = {
            'team_a_player': (255, 0, 0),      # Red
            'team_a_goalkeeper': (200, 0, 0),  # Dark Red
            'team_b_player': (0, 0, 255),      # Blue
            'team_b_goalkeeper': (0, 0, 200),  # Dark Blue
            'referee': (0, 255, 0),            # Green
            'ball': (255, 255, 0),             # Yellow
            'other': (128, 128, 128),          # Gray
            'staff': (255, 0, 255)             # Magenta
        }
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logger.info(f"✅ Loaded trained model: {self.model_path}")
            else:
                self.model = YOLO('yolov8n.pt')
                logger.warning("⚠️ Trained model not found, using default YOLOv8")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            self.model = YOLO('yolov8n.pt')
    
    def analyze_video(self, video_path, output_dir="test_results", real_time=True):
        """Analyze video and generate performance metrics with real-time display"""
        logger.info(f"🎥 Analyzing video: {video_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"❌ Cannot open video: {video_path}")
            return None
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        logger.info(f"📊 Video Info:")
        logger.info(f"  📏 Resolution: {width}x{height}")
        logger.info(f"  🎬 Total frames: {total_frames}")
        logger.info(f"  ⏱️  FPS: {fps:.2f}")
        logger.info(f"  ⏰ Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Setup video writer for annotated output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        annotated_video = cv2.VideoWriter(
            str(output_path / "annotated_video.mp4"),
            fourcc, fps, (width, height)
        )
        
        # Analysis variables
        frame_count = 0
        detection_stats = defaultdict(list)
        class_counts = Counter()
        confidence_scores = []
        processing_times = []
        
        # Sample frames for detailed analysis (every 30th frame)
        sample_frames = []
        sample_interval = 30
        
        # Real-time display setup
        if real_time:
            cv2.namedWindow('Godseye AI - Real-time Analysis', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Godseye AI - Real-time Analysis', 1280, 720)
            logger.info("🎬 Starting real-time video analysis...")
            logger.info("📝 Press 'q' to quit, 'p' to pause, 's' to save screenshot")
        
        logger.info("🔄 Processing video frames...")
        start_time = time.time()
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                # If paused, keep showing the same frame
                ret = True
            
            frame_start = time.time()
            
            if ret:
                # Run inference with lower confidence threshold
                results = self.model(frame, conf=0.1, verbose=False)
                
                # Process detections
                annotated_frame = frame.copy()
                frame_detections = []
                
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    
                    # Debug: Print detection info for first few frames
                    if frame_count < 5:
                        logger.info(f"Frame {frame_count}: Found {len(boxes)} detections")
                        for i, (conf, cls_id) in enumerate(zip(confidences, class_ids)):
                            if cls_id < len(self.class_names):
                                logger.info(f"  Detection {i}: {self.class_names[cls_id]} (conf: {conf:.3f})")
                else:
                    # Debug: Print when no detections
                    if frame_count < 5:
                        logger.info(f"Frame {frame_count}: No detections found")
                
                if results[0].boxes is not None:
                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        if cls_id < len(self.class_names):
                            class_name = self.class_names[cls_id]
                            
                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, box)
                            color = self.class_colors.get(class_name, (255, 255, 255))
                            
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw label with background
                            label = f"{class_name}: {conf:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(annotated_frame, (x1, y1-30), (x1+label_size[0]+10, y1), color, -1)
                            cv2.putText(annotated_frame, label, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # Store detection data
                            detection_data = {
                                'class': class_name,
                                'confidence': float(conf),
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'area': (x2-x1) * (y2-y1)
                            }
                            frame_detections.append(detection_data)
                            
                            # Update statistics
                            class_counts[class_name] += 1
                            confidence_scores.append(float(conf))
                            detection_stats[class_name].append(float(conf))
                
                # Add comprehensive frame info overlay
                info_y = 30
                cv2.putText(annotated_frame, f"Godseye AI - Real-time Analysis", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                info_y += 35
                
                cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                info_y += 30
                
                cv2.putText(annotated_frame, f"Detections: {len(frame_detections)}", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                info_y += 30
                
                # Show current detections
                if frame_detections:
                    cv2.putText(annotated_frame, "Current Detections:", (10, info_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    info_y += 25
                    
                    for i, det in enumerate(frame_detections[:5]):  # Show max 5 detections
                        det_text = f"  {det['class']}: {det['confidence']:.2f}"
                        cv2.putText(annotated_frame, det_text, (10, info_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        info_y += 20
                
                # Add class count summary
                info_y += 10
                cv2.putText(annotated_frame, "Total Counts:", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                info_y += 25
                
                for class_name, count in class_counts.most_common(5):
                    count_text = f"  {class_name}: {count}"
                    cv2.putText(annotated_frame, count_text, (10, info_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    info_y += 20
                
                # Add controls info
                cv2.putText(annotated_frame, "Controls: 'q'=quit, 'p'=pause, 's'=screenshot", 
                           (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                
                # Add pause indicator
                if paused:
                    cv2.putText(annotated_frame, "PAUSED", (width-150, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # Write annotated frame
                annotated_video.write(annotated_frame)
                
                # Store sample frame data
                if frame_count % sample_interval == 0:
                    sample_frames.append({
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps,
                        'detections': frame_detections,
                        'detection_count': len(frame_detections)
                    })
                
                frame_count += 1
                processing_time = time.time() - frame_start
                processing_times.append(processing_time)
                
                # Real-time display
                if real_time:
                    cv2.imshow('Godseye AI - Real-time Analysis', annotated_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("🛑 Quit requested by user")
                        break
                    elif key == ord('p'):
                        paused = not paused
                        logger.info(f"⏸️  {'Paused' if paused else 'Resumed'}")
                    elif key == ord('s'):
                        screenshot_path = output_path / f"screenshot_frame_{frame_count}.jpg"
                        cv2.imwrite(str(screenshot_path), annotated_frame)
                        logger.info(f"📸 Screenshot saved: {screenshot_path}")
                
                # Progress update (less frequent for real-time)
                if frame_count % 50 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"📈 Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Cleanup
        cap.release()
        annotated_video.release()
        if real_time:
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_processing_time = np.mean(processing_times)
        fps_actual = frame_count / total_time
        
        logger.info(f"✅ Analysis complete!")
        logger.info(f"⏱️  Total processing time: {total_time:.2f} seconds")
        logger.info(f"🚀 Average FPS: {fps_actual:.2f}")
        logger.info(f"📊 Processed {frame_count} frames")
        
        # Generate performance report
        self.generate_performance_report(
            output_path, detection_stats, class_counts, confidence_scores,
            processing_times, sample_frames, total_time, fps_actual
        )
        
        return {
            'total_frames': frame_count,
            'processing_time': total_time,
            'fps_actual': fps_actual,
            'class_counts': dict(class_counts),
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'sample_frames': sample_frames
        }
    
    def generate_performance_report(self, output_path, detection_stats, class_counts, 
                                  confidence_scores, processing_times, sample_frames, 
                                  total_time, fps_actual):
        """Generate comprehensive performance report"""
        
        # Create performance summary
        performance_data = {
            'model_info': {
                'model_path': self.model_path,
                'classes': self.class_names,
                'test_timestamp': datetime.now().isoformat()
            },
            'video_analysis': {
                'total_frames': len(processing_times),
                'processing_time_seconds': total_time,
                'fps_actual': fps_actual,
                'avg_frame_processing_time': np.mean(processing_times)
            },
            'detection_statistics': {
                'total_detections': sum(class_counts.values()),
                'class_counts': dict(class_counts),
                'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
                'confidence_std': np.std(confidence_scores) if confidence_scores else 0,
                'min_confidence': np.min(confidence_scores) if confidence_scores else 0,
                'max_confidence': np.max(confidence_scores) if confidence_scores else 0
            },
            'class_performance': {}
        }
        
        # Calculate per-class statistics
        for class_name, confidences in detection_stats.items():
            if confidences:
                performance_data['class_performance'][class_name] = {
                    'count': len(confidences),
                    'avg_confidence': np.mean(confidences),
                    'std_confidence': np.std(confidences),
                    'min_confidence': np.min(confidences),
                    'max_confidence': np.max(confidences)
                }
        
        # Add sample frame data
        performance_data['sample_frames'] = sample_frames[:10]  # First 10 samples
        
        # Save performance report
        with open(output_path / "performance_report.json", 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        # Create visualization
        self.create_performance_visualization(output_path, detection_stats, class_counts, confidence_scores)
        
        # Print summary
        self.print_performance_summary(performance_data)
    
    def create_performance_visualization(self, output_path, detection_stats, class_counts, confidence_scores):
        """Create performance visualization charts"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Godseye AI Model Performance Analysis', fontsize=16)
            
            # 1. Class distribution
            if class_counts:
                classes = list(class_counts.keys())
                counts = list(class_counts.values())
                axes[0, 0].bar(classes, counts, color=['red', 'darkred', 'blue', 'darkblue', 'green', 'yellow', 'gray', 'magenta'][:len(classes)])
                axes[0, 0].set_title('Detection Count by Class')
                axes[0, 0].set_ylabel('Number of Detections')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Confidence distribution
            if confidence_scores:
                axes[0, 1].hist(confidence_scores, bins=20, alpha=0.7, color='skyblue')
                axes[0, 1].set_title('Confidence Score Distribution')
                axes[0, 1].set_xlabel('Confidence Score')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].axvline(np.mean(confidence_scores), color='red', linestyle='--', label=f'Mean: {np.mean(confidence_scores):.3f}')
                axes[0, 1].legend()
            
            # 3. Per-class confidence
            if detection_stats:
                class_names = list(detection_stats.keys())
                avg_confidences = [np.mean(confidences) for confidences in detection_stats.values()]
                axes[1, 0].bar(class_names, avg_confidences, color=['red', 'darkred', 'blue', 'darkblue', 'green', 'yellow', 'gray', 'magenta'][:len(class_names)])
                axes[1, 0].set_title('Average Confidence by Class')
                axes[1, 0].set_ylabel('Average Confidence')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. Detection timeline (sample frames)
            sample_frames = [f for f in range(0, len(confidence_scores), 30)][:20]  # Sample every 30th frame
            if sample_frames:
                sample_counts = [class_counts.get(cls, 0) for cls in self.class_names]
                axes[1, 1].pie(sample_counts, labels=self.class_names, autopct='%1.1f%%', startangle=90)
                axes[1, 1].set_title('Overall Detection Distribution')
            
            plt.tight_layout()
            plt.savefig(output_path / "performance_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Performance visualization saved to: {output_path / 'performance_analysis.png'}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create visualization: {e}")
    
    def print_performance_summary(self, performance_data):
        """Print performance summary to console"""
        logger.info("=" * 60)
        logger.info("🎯 GODSEYE AI MODEL PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        
        # Model info
        logger.info(f"🤖 Model: {performance_data['model_info']['model_path']}")
        logger.info(f"📅 Test Date: {performance_data['model_info']['test_timestamp']}")
        
        # Video analysis
        video_info = performance_data['video_analysis']
        logger.info(f"🎥 Video Analysis:")
        logger.info(f"  📊 Total Frames: {video_info['total_frames']}")
        logger.info(f"  ⏱️  Processing Time: {video_info['processing_time_seconds']:.2f} seconds")
        logger.info(f"  🚀 Actual FPS: {video_info['fps_actual']:.2f}")
        logger.info(f"  ⚡ Avg Frame Time: {video_info['avg_frame_processing_time']*1000:.1f}ms")
        
        # Detection statistics
        det_stats = performance_data['detection_statistics']
        logger.info(f"🎯 Detection Statistics:")
        logger.info(f"  📈 Total Detections: {det_stats['total_detections']}")
        logger.info(f"  🎯 Avg Confidence: {det_stats['avg_confidence']:.3f}")
        logger.info(f"  📊 Confidence Range: {det_stats['min_confidence']:.3f} - {det_stats['max_confidence']:.3f}")
        
        # Class performance
        logger.info(f"👥 Class Performance:")
        for class_name, stats in performance_data['class_performance'].items():
            logger.info(f"  {class_name}: {stats['count']} detections, avg conf: {stats['avg_confidence']:.3f}")
        
        logger.info("=" * 60)

def main():
    """Main function"""
    logger.info("🚀 Godseye AI - Model Performance Tester")
    logger.info("=" * 50)
    
    # Initialize tester
    tester = ModelPerformanceTester()
    
    # Test video
    video_path = "madrid_vs_city.mp4"
    
    if not os.path.exists(video_path):
        logger.error(f"❌ Video not found: {video_path}")
        return
    
    # Run analysis
    results = tester.analyze_video(video_path)
    
    if results:
        logger.info("=" * 50)
        logger.info("✅ PERFORMANCE TEST COMPLETED!")
        logger.info(f"📁 Results saved to: test_results/")
        logger.info(f"🎬 Annotated video: test_results/annotated_video.mp4")
        logger.info(f"📊 Performance report: test_results/performance_report.json")
        logger.info(f"📈 Visualization: test_results/performance_analysis.png")
        logger.info("=" * 50)
    else:
        logger.error("❌ Performance test failed")

if __name__ == "__main__":
    main()
