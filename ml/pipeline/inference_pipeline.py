"""
================================================================================
GODSEYE AI - INFERENCE PIPELINE MODULE
================================================================================

Author: Victor Ibhafidon
Date: January 28, 2025
Version: 1.0.0

DESCRIPTION:
This module provides the comprehensive inference pipeline for the Godseye AI
sports analytics platform. It integrates all ML models including detection,
tracking, pose estimation, and event detection into a unified real-time
processing pipeline. This is the core orchestration module for all AI analysis.

PIPELINE INTEGRATION:
- Orchestrates: ml/models/detection.py for object detection
- Coordinates: ml/models/tracking.py for multi-object tracking
- Integrates: ml/models/pose_estimation.py for pose analysis
- Manages: ml/models/event_detection.py for event recognition
- Processes: ml/analytics/statistics.py for analytics calculation
- Serves: Real-time results to Frontend components
- Connects: With inference/serve.py for API endpoints

FEATURES:
- Unified real-time inference pipeline
- Multi-model coordination and synchronization
- Efficient frame processing with caching
- Real-time analytics generation
- Configurable processing parameters
- Error handling and recovery
- Performance monitoring and optimization
- Batch and streaming processing modes

DEPENDENCIES:
- torch for model inference
- opencv-python for video processing
- numpy for numerical operations
- All ML model modules for integrated processing

USAGE:
    from ml.pipeline.inference_pipeline import InferencePipeline
    
    # Initialize pipeline
    pipeline = InferencePipeline(config_path='configs/pipeline.yaml')
    
    # Process video frame
    results = pipeline.process_frame(frame)

COMPETITOR ANALYSIS:
Based on analysis of industry-leading inference pipelines from VeoCam, Stats
Perform, and other professional sports analytics platforms. Implements
enterprise-grade pipeline orchestration with professional performance and reliability.

================================================================================
"""

import cv2
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import time
from collections import defaultdict, deque
import json
import threading
from queue import Queue, Empty

# Import our custom modules
from ..models.detection import YOLODetector, create_detection_model
from ..models.tracking import MultiObjectTracker, Track
from ..models.pose_estimation import MediaPipePoseEstimator, PoseTracker, PoseEstimation
from ..models.event_detection import EventDetector, Event, EventTracker
from ..utils.class_mapping import FootballClassMapper
from ..analytics.statistics import StatisticsCalculator, MatchStatistics

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""
    # Detection settings
    detection_model: str = "yolov8n"
    detection_confidence: float = 0.5
    detection_iou_threshold: float = 0.45
    
    # Tracking settings
    tracking_max_disappeared: int = 30
    tracking_max_distance: float = 0.1
    
    # Pose estimation settings
    pose_estimator_type: str = "mediapipe"
    pose_confidence_threshold: float = 0.5
    
    # Event detection settings
    event_confidence_threshold: float = 0.5
    event_tracking_duration: float = 5.0
    
    # Processing settings
    frame_skip: int = 1  # Process every Nth frame
    batch_size: int = 1
    device: str = "auto"
    
    # Output settings
    save_visualizations: bool = True
    save_annotations: bool = True
    output_format: str = "json"  # json, csv, xml


@dataclass
class InferenceResults:
    """Results from inference pipeline."""
    frame_id: int
    timestamp: float
    processing_time: float
    
    # Detection results
    detections: List[Dict] = field(default_factory=list)
    
    # Tracking results
    tracks: List[Dict] = field(default_factory=list)
    
    # Pose estimation results
    poses: List[PoseEstimation] = field(default_factory=list)
    
    # Event detection results
    events: List[Event] = field(default_factory=list)
    
    # Statistics
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Visualization
    annotated_frame: Optional[np.ndarray] = None


class InferencePipeline:
    """Main inference pipeline for sports analytics."""
    
    def __init__(self, config: InferenceConfig):
        """
        Initialize inference pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'auto' else config.device)
        
        # Initialize components
        self._initialize_components()
        
        # Statistics calculator
        self.stats_calculator = StatisticsCalculator()
        
        # Class mapper
        self.class_mapper = FootballClassMapper()
        
        # Processing state
        self.frame_count = 0
        self.start_time = None
        self.results_history = deque(maxlen=1000)
        
        # Performance metrics
        self.performance_metrics = {
            'total_frames': 0,
            'total_processing_time': 0.0,
            'avg_fps': 0.0,
            'component_times': defaultdict(list)
        }
        
        logger.info(f"Inference pipeline initialized on {self.device}")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Detection model
            logger.info("Initializing detection model...")
            self.detector = create_detection_model(
                model_name=self.config.detection_model,
                num_classes=len(self.class_mapper.CLASSES),
                device=self.device
            )
            
            # Tracking
            logger.info("Initializing tracking system...")
            self.tracker = MultiObjectTracker(
                max_disappeared=self.config.tracking_max_disappeared,
                max_distance=self.config.tracking_max_distance
            )
            
            # Pose estimation
            logger.info("Initializing pose estimation...")
            self.pose_estimator = MediaPipePoseEstimator(
                model_complexity=2,
                min_detection_confidence=self.config.pose_confidence_threshold,
                min_tracking_confidence=self.config.pose_confidence_threshold
            )
            self.pose_tracker = PoseTracker(
                max_disappeared=self.config.tracking_max_disappeared,
                max_distance=self.config.tracking_max_distance,
                pose_estimator=self.pose_estimator
            )
            
            # Event detection
            logger.info("Initializing event detection...")
            self.event_detector = EventDetector(
                confidence_threshold=self.config.event_confidence_threshold
            )
            self.event_tracker = EventTracker(
                max_event_duration=self.config.event_tracking_duration
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray, frame_id: int = None, timestamp: float = None) -> InferenceResults:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input frame
            frame_id: Frame number (auto-incremented if None)
            timestamp: Frame timestamp (auto-calculated if None)
            
        Returns:
            Inference results
        """
        start_time = time.time()
        
        # Set frame ID and timestamp
        if frame_id is None:
            frame_id = self.frame_count
        if timestamp is None:
            timestamp = time.time()
        
        self.frame_count += 1
        
        # Skip frames if configured
        if self.frame_count % self.config.frame_skip != 0:
            return InferenceResults(
                frame_id=frame_id,
                timestamp=timestamp,
                processing_time=time.time() - start_time
            )
        
        try:
            # 1. Object Detection
            detection_start = time.time()
            detections = self._run_detection(frame)
            detection_time = time.time() - detection_start
            self.performance_metrics['component_times']['detection'].append(detection_time)
            
            # 2. Object Tracking
            tracking_start = time.time()
            tracks = self._run_tracking(frame, detections)
            tracking_time = time.time() - tracking_start
            self.performance_metrics['component_times']['tracking'].append(tracking_time)
            
            # 3. Pose Estimation
            pose_start = time.time()
            poses = self._run_pose_estimation(frame, tracks)
            pose_time = time.time() - pose_start
            self.performance_metrics['component_times']['pose'].append(pose_time)
            
            # 4. Event Detection
            event_start = time.time()
            events = self._run_event_detection(frame, detections, tracks, poses, frame_id, timestamp)
            event_time = time.time() - event_start
            self.performance_metrics['component_times']['events'].append(event_time)
            
            # 5. Statistics Calculation
            stats_start = time.time()
            statistics = self._calculate_statistics(detections, tracks, poses, events)
            stats_time = time.time() - stats_start
            self.performance_metrics['component_times']['statistics'].append(stats_time)
            
            # 6. Visualization
            vis_start = time.time()
            annotated_frame = self._create_visualization(frame, detections, tracks, poses, events)
            vis_time = time.time() - vis_start
            self.performance_metrics['component_times']['visualization'].append(vis_time)
            
            # Create results
            processing_time = time.time() - start_time
            results = InferenceResults(
                frame_id=frame_id,
                timestamp=timestamp,
                processing_time=processing_time,
                detections=detections,
                tracks=tracks,
                poses=poses,
                events=events,
                statistics=statistics,
                annotated_frame=annotated_frame
            )
            
            # Update performance metrics
            self._update_performance_metrics(processing_time)
            
            # Store results
            self.results_history.append(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
            return InferenceResults(
                frame_id=frame_id,
                timestamp=timestamp,
                processing_time=time.time() - start_time
            )
    
    def _run_detection(self, frame: np.ndarray) -> List[Dict]:
        """Run object detection on frame."""
        try:
            # Run detection
            detections = self.detector.detect(frame)
            
            # Filter by confidence
            filtered_detections = [
                det for det in detections 
                if det.get('confidence', 0) >= self.config.detection_confidence
            ]
            
            # Add class information
            for detection in filtered_detections:
                class_id = detection.get('class_id', -1)
                detection['class_name'] = self.class_mapper.get_class_name(class_id)
                detection['team_id'] = self.class_mapper.get_team_from_index(class_id)
                detection['role'] = self.class_mapper.get_role_from_index(class_id)
            
            return filtered_detections
            
        except Exception as e:
            logger.error(f"Error in detection: {e}")
            return []
    
    def _run_tracking(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Run object tracking on frame."""
        try:
            # Update tracker with detections
            tracks = self.tracker.update(detections, frame)
            
            # Add additional tracking information
            for track in tracks:
                track['team_id'] = self.class_mapper.get_team_from_index(track.get('class_id', -1))
                track['role'] = self.class_mapper.get_role_from_index(track.get('class_id', -1))
            
            return tracks
            
        except Exception as e:
            logger.error(f"Error in tracking: {e}")
            return []
    
    def _run_pose_estimation(self, frame: np.ndarray, tracks: List[Dict]) -> List[PoseEstimation]:
        """Run pose estimation on frame."""
        try:
            # Update pose tracker
            poses = self.pose_tracker.update(frame, tracks)
            
            # Set frame information
            for pose in poses:
                pose.frame_id = self.frame_count
                pose.timestamp = time.time()
            
            return poses
            
        except Exception as e:
            logger.error(f"Error in pose estimation: {e}")
            return []
    
    def _run_event_detection(self, frame: np.ndarray, detections: List[Dict], 
                           tracks: List[Dict], poses: List[PoseEstimation], 
                           frame_id: int, timestamp: float) -> List[Event]:
        """Run event detection on frame."""
        try:
            # Detect events
            new_events = self.event_detector.detect_events(
                frame=frame,
                detections=detections,
                tracking_results=tracks,
                pose_results=poses,
                frame_id=frame_id,
                timestamp=timestamp
            )
            
            # Update event tracker
            completed_events = self.event_tracker.update_events(new_events)
            
            # Return both new and completed events
            all_events = new_events + completed_events
            
            return all_events
            
        except Exception as e:
            logger.error(f"Error in event detection: {e}")
            return []
    
    def _calculate_statistics(self, detections: List[Dict], tracks: List[Dict], 
                            poses: List[PoseEstimation], events: List[Event]) -> Dict[str, Any]:
        """Calculate statistics from current frame results."""
        try:
            # Basic statistics
            stats = {
                'frame_id': self.frame_count,
                'timestamp': time.time(),
                'detections_count': len(detections),
                'tracks_count': len(tracks),
                'poses_count': len(poses),
                'events_count': len(events)
            }
            
            # Team statistics
            team_stats = self.class_mapper.get_team_statistics(detections)
            stats['team_statistics'] = team_stats
            
            # Event statistics
            event_stats = defaultdict(int)
            for event in events:
                event_stats[event.event_name] += 1
            stats['event_statistics'] = dict(event_stats)
            
            # Performance statistics
            stats['performance'] = self.get_performance_metrics()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}
    
    def _create_visualization(self, frame: np.ndarray, detections: List[Dict], 
                            tracks: List[Dict], poses: List[PoseEstimation], 
                            events: List[Event]) -> np.ndarray:
        """Create visualization of all results."""
        try:
            vis_frame = frame.copy()
            
            # Draw detections
            for detection in detections:
                bbox = detection.get('bbox')
                if bbox:
                    x, y, w, h = bbox
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)
                    
                    # Get color based on team
                    team_id = detection.get('team_id')
                    color = self.class_mapper.get_team_color(team_id)
                    
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{detection.get('class_name', 'unknown')} ({detection.get('confidence', 0):.2f})"
                    cv2.putText(vis_frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw tracks
            for track in tracks:
                bbox = track.get('bbox')
                if bbox:
                    x, y, w, h = bbox
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)
                    
                    # Get color based on team
                    team_id = track.get('team_id')
                    color = self.class_mapper.get_team_color(team_id)
                    
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw track ID
                    track_id = track.get('track_id', 'unknown')
                    label = f"ID:{track_id}"
                    cv2.putText(vis_frame, label, (x1, y1 - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw poses
            from ..models.pose_estimation import visualize_poses
            vis_frame = visualize_poses(vis_frame, poses)
            
            # Draw events
            from ..models.event_detection import visualize_events
            vis_frame = visualize_events(vis_frame, events)
            
            return vis_frame
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return frame
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics."""
        self.performance_metrics['total_frames'] += 1
        self.performance_metrics['total_processing_time'] += processing_time
        
        if self.performance_metrics['total_processing_time'] > 0:
            self.performance_metrics['avg_fps'] = (
                self.performance_metrics['total_frames'] / 
                self.performance_metrics['total_processing_time']
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = self.performance_metrics.copy()
        
        # Calculate average component times
        for component, times in metrics['component_times'].items():
            if times:
                metrics[f'avg_{component}_time'] = np.mean(times)
                metrics[f'max_{component}_time'] = np.max(times)
        
        return metrics
    
    def process_video(self, video_path: str, output_path: str = None) -> List[InferenceResults]:
        """
        Process entire video through pipeline.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            
        Returns:
            List of inference results for each frame
        """
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video if requested
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        results = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result = self.process_frame(frame, frame_count, frame_count / fps)
                results.append(result)
                
                # Write annotated frame to output video
                if out and result.annotated_frame is not None:
                    out.write(result.annotated_frame)
                
                frame_count += 1
                
                # Log progress
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
            
            logger.info(f"Video processing completed. Processed {frame_count} frames.")
            
        finally:
            cap.release()
            if out:
                out.release()
        
        return results
    
    def process_stream(self, stream_url: str, duration: float = None) -> List[InferenceResults]:
        """
        Process live video stream.
        
        Args:
            stream_url: URL or path to video stream
            duration: Maximum duration to process (seconds)
            
        Returns:
            List of inference results
        """
        logger.info(f"Processing stream: {stream_url}")
        
        # Open stream
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise ValueError(f"Could not open stream: {stream_url}")
        
        # Process frames
        results = []
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from stream")
                    time.sleep(0.1)  # Wait before retrying
                    continue
                
                # Process frame
                result = self.process_frame(frame, frame_count, time.time() - start_time)
                results.append(result)
                
                frame_count += 1
                
                # Log progress
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Processed {frame_count} frames in {elapsed:.1f}s")
            
        finally:
            cap.release()
        
        logger.info(f"Stream processing completed. Processed {frame_count} frames.")
        return results
    
    def export_results(self, results: List[InferenceResults], output_path: str, format: str = "json"):
        """
        Export inference results to file.
        
        Args:
            results: List of inference results
            output_path: Output file path
            format: Export format (json, csv, xml)
        """
        if format.lower() == "json":
            self._export_json(results, output_path)
        elif format.lower() == "csv":
            self._export_csv(results, output_path)
        elif format.lower() == "xml":
            self._export_xml(results, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, results: List[InferenceResults], output_path: str):
        """Export results to JSON format."""
        export_data = {
            'pipeline_config': {
                'detection_model': self.config.detection_model,
                'detection_confidence': self.config.detection_confidence,
                'tracking_max_disappeared': self.config.tracking_max_disappeared,
                'pose_estimator_type': self.config.pose_estimator_type,
                'event_confidence_threshold': self.config.event_confidence_threshold
            },
            'performance_metrics': self.get_performance_metrics(),
            'results': []
        }
        
        for result in results:
            result_data = {
                'frame_id': result.frame_id,
                'timestamp': result.timestamp,
                'processing_time': result.processing_time,
                'detections': result.detections,
                'tracks': result.tracks,
                'poses': [
                    {
                        'person_id': pose.person_id,
                        'confidence': pose.confidence,
                        'bbox': pose.bbox,
                        'keypoints': [
                            {
                                'name': kp.name,
                                'x': kp.x,
                                'y': kp.y,
                                'confidence': kp.confidence,
                                'visibility': kp.visibility
                            } for kp in pose.keypoints
                        ],
                        'body_orientation': pose.body_orientation,
                        'running_phase': pose.running_phase,
                        'balance_score': pose.balance_score,
                        'fatigue_indicators': pose.fatigue_indicators
                    } for pose in result.poses
                ],
                'events': [
                    {
                        'event_id': event.event_id,
                        'event_name': event.event_name,
                        'confidence': event.confidence,
                        'timestamp': event.timestamp,
                        'frame_id': event.frame_id,
                        'bbox': event.bbox,
                        'player_id': event.player_id,
                        'team_id': event.team_id,
                        'description': event.description,
                        'event_data': event.event_data
                    } for event in result.events
                ],
                'statistics': result.statistics
            }
            export_data['results'].append(result_data)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Results exported to {output_path}")
    
    def _export_csv(self, results: List[InferenceResults], output_path: str):
        """Export results to CSV format."""
        import pandas as pd
        
        # Flatten results for CSV
        csv_data = []
        for result in results:
            base_row = {
                'frame_id': result.frame_id,
                'timestamp': result.timestamp,
                'processing_time': result.processing_time,
                'detections_count': len(result.detections),
                'tracks_count': len(result.tracks),
                'poses_count': len(result.poses),
                'events_count': len(result.events)
            }
            
            # Add detection data
            for i, detection in enumerate(result.detections):
                row = base_row.copy()
                row.update({
                    'detection_id': i,
                    'detection_class': detection.get('class_name'),
                    'detection_confidence': detection.get('confidence'),
                    'detection_bbox': str(detection.get('bbox')),
                    'detection_team_id': detection.get('team_id'),
                    'detection_role': detection.get('role')
                })
                csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Results exported to {output_path}")
    
    def _export_xml(self, results: List[InferenceResults], output_path: str):
        """Export results to XML format."""
        import xml.etree.ElementTree as ET
        
        root = ET.Element("inference_results")
        
        # Add config
        config_elem = ET.SubElement(root, "config")
        for key, value in self.config.__dict__.items():
            ET.SubElement(config_elem, key).text = str(value)
        
        # Add performance metrics
        metrics_elem = ET.SubElement(root, "performance_metrics")
        for key, value in self.get_performance_metrics().items():
            ET.SubElement(metrics_elem, key).text = str(value)
        
        # Add results
        results_elem = ET.SubElement(root, "results")
        for result in results:
            result_elem = ET.SubElement(results_elem, "frame")
            result_elem.set("id", str(result.frame_id))
            result_elem.set("timestamp", str(result.timestamp))
            result_elem.set("processing_time", str(result.processing_time))
            
            # Add detections
            detections_elem = ET.SubElement(result_elem, "detections")
            for detection in result.detections:
                det_elem = ET.SubElement(detections_elem, "detection")
                det_elem.set("class", detection.get('class_name', ''))
                det_elem.set("confidence", str(detection.get('confidence', 0)))
                det_elem.set("bbox", str(detection.get('bbox', [])))
            
            # Add events
            events_elem = ET.SubElement(result_elem, "events")
            for event in result.events:
                event_elem = ET.SubElement(events_elem, "event")
                event_elem.set("name", event.event_name)
                event_elem.set("confidence", str(event.confidence))
                event_elem.set("player_id", str(event.player_id or ''))
                event_elem.set("team_id", str(event.team_id or ''))
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        logger.info(f"Results exported to {output_path}")


def create_inference_pipeline(config: InferenceConfig = None) -> InferencePipeline:
    """
    Factory function to create inference pipeline.
    
    Args:
        config: Pipeline configuration (uses default if None)
        
    Returns:
        Inference pipeline instance
    """
    if config is None:
        config = InferenceConfig()
    
    return InferencePipeline(config)


# Example usage
if __name__ == "__main__":
    # Create pipeline
    config = InferenceConfig(
        detection_model="yolov8n",
        detection_confidence=0.5,
        pose_estimator_type="mediapipe",
        event_confidence_threshold=0.6
    )
    
    pipeline = create_inference_pipeline(config)
    
    # Process video
    results = pipeline.process_video("input_video.mp4", "output_video.mp4")
    
    # Export results
    pipeline.export_results(results, "results.json", "json")
    
    # Print performance metrics
    metrics = pipeline.get_performance_metrics()
    print(f"Average FPS: {metrics['avg_fps']:.2f}")
    print(f"Total frames processed: {metrics['total_frames']}")
