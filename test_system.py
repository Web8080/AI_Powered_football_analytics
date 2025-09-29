#!/usr/bin/env python3
"""
Test script for Godseye AI sports analytics system.
This script allows you to test the complete pipeline with your football video.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules
from ml.pipeline.inference_pipeline import InferencePipeline, InferenceConfig
from ml.utils.class_mapping import FootballClassMapper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_class_mapping():
    """Test the class mapping system."""
    logger.info("Testing class mapping system...")
    
    mapper = FootballClassMapper()
    
    # Test class information
    logger.info(f"Number of classes: {len(mapper.CLASSES)}")
    logger.info(f"Classes: {mapper.CLASSES}")
    
    # Test team mapping
    for i, class_name in enumerate(mapper.CLASSES):
        team_id = mapper.get_team_from_class(class_name)
        role = mapper.get_role_from_class(class_name)
        logger.info(f"Class {i}: {class_name} -> Team: {team_id}, Role: {role}")
    
    logger.info("Class mapping test completed ✓")


def test_detection_model():
    """Test the detection model."""
    logger.info("Testing detection model...")
    
    try:
        from ml.models.detection import create_detection_model
        
        # Create a simple detection model
        model = create_detection_model(
            model_name="yolov8n",
            num_classes=8,  # Our 8 football classes
            device="cpu"  # Use CPU for testing
        )
        
        logger.info(f"Detection model created: {type(model).__name__}")
        logger.info("Detection model test completed ✓")
        
    except Exception as e:
        logger.error(f"Detection model test failed: {e}")
        return False
    
    return True


def test_pose_estimation():
    """Test pose estimation."""
    logger.info("Testing pose estimation...")
    
    try:
        from ml.models.pose_estimation import MediaPipePoseEstimator
        
        # Create pose estimator
        pose_estimator = MediaPipePoseEstimator(
            model_complexity=1,  # Use lower complexity for testing
            min_detection_confidence=0.5
        )
        
        logger.info(f"Pose estimator created: {type(pose_estimator).__name__}")
        logger.info("Pose estimation test completed ✓")
        
    except Exception as e:
        logger.error(f"Pose estimation test failed: {e}")
        return False
    
    return True


def test_event_detection():
    """Test event detection."""
    logger.info("Testing event detection...")
    
    try:
        from ml.models.event_detection import EventDetector
        
        # Create event detector
        event_detector = EventDetector(
            confidence_threshold=0.5
        )
        
        logger.info(f"Event detector created: {type(event_detector).__name__}")
        logger.info("Event detection test completed ✓")
        
    except Exception as e:
        logger.error(f"Event detection test failed: {e}")
        return False
    
    return True


def test_inference_pipeline():
    """Test the complete inference pipeline."""
    logger.info("Testing inference pipeline...")
    
    try:
        # Create pipeline configuration
        config = InferenceConfig(
            detection_model="yolov8n",
            detection_confidence=0.5,
            pose_estimator_type="mediapipe",
            event_confidence_threshold=0.5,
            device="cpu"  # Use CPU for testing
        )
        
        # Create pipeline
        pipeline = InferencePipeline(config)
        
        logger.info(f"Inference pipeline created successfully")
        logger.info("Inference pipeline test completed ✓")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Inference pipeline test failed: {e}")
        return None


def test_with_video(pipeline, video_path: str):
    """Test the pipeline with a video file."""
    logger.info(f"Testing pipeline with video: {video_path}")
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    
    try:
        # Process video
        results = pipeline.process_video(video_path)
        
        logger.info(f"Video processing completed. Processed {len(results)} frames.")
        
        # Print summary statistics
        total_detections = sum(len(r.detections) for r in results)
        total_tracks = sum(len(r.tracks) for r in results)
        total_poses = sum(len(r.poses) for r in results)
        total_events = sum(len(r.events) for r in results)
        
        logger.info(f"Summary:")
        logger.info(f"  Total detections: {total_detections}")
        logger.info(f"  Total tracks: {total_tracks}")
        logger.info(f"  Total poses: {total_poses}")
        logger.info(f"  Total events: {total_events}")
        
        # Export results
        output_path = "test_results.json"
        pipeline.export_results(results, output_path, "json")
        logger.info(f"Results exported to {output_path}")
        
        # Print performance metrics
        metrics = pipeline.get_performance_metrics()
        logger.info(f"Performance metrics:")
        logger.info(f"  Average FPS: {metrics.get('avg_fps', 0):.2f}")
        logger.info(f"  Total frames: {metrics.get('total_frames', 0)}")
        
        logger.info("Video processing test completed ✓")
        return True
        
    except Exception as e:
        logger.error(f"Video processing test failed: {e}")
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Godseye AI sports analytics system")
    parser.add_argument("--video", type=str, help="Path to test video file")
    parser.add_argument("--skip-tests", action="store_true", help="Skip individual component tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Godseye AI system tests...")
    logger.info("=" * 50)
    
    # Test individual components
    if not args.skip_tests:
        test_class_mapping()
        print()
        
        if not test_detection_model():
            logger.error("Detection model test failed. Exiting.")
            return 1
        print()
        
        if not test_pose_estimation():
            logger.error("Pose estimation test failed. Exiting.")
            return 1
        print()
        
        if not test_event_detection():
            logger.error("Event detection test failed. Exiting.")
            return 1
        print()
    
    # Test complete pipeline
    pipeline = test_inference_pipeline()
    if not pipeline:
        logger.error("Inference pipeline test failed. Exiting.")
        return 1
    print()
    
    # Test with video if provided
    if args.video:
        if not test_with_video(pipeline, args.video):
            logger.error("Video processing test failed.")
            return 1
    else:
        logger.info("No video provided. Use --video <path> to test with a video file.")
        logger.info("Example: python test_system.py --video your_football_video.mp4")
    
    logger.info("=" * 50)
    logger.info("All tests completed successfully! ✓")
    logger.info("Your Godseye AI system is ready for testing.")
    
    return 0


if __name__ == "__main__":
    exit(main())
