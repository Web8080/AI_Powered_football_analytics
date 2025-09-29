#!/usr/bin/env python3
"""
Setup script for Godseye AI testing environment.
This script helps you set up the testing environment and install dependencies.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible."""
    logger.info("Checking Python version...")
    
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required. Current version: {}.{}.{}".format(
            sys.version_info.major, sys.version_info.minor, sys.version_info.micro
        ))
        return False
    
    logger.info(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ✓")
    return True


def check_dependencies():
    """Check if required dependencies are available."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'opencv-python',
        'numpy',
        'pandas',
        'ultralytics',
        'mediapipe'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"  {package} ✓")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"  {package} ✗")
    
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        return False
    
    logger.info("All required dependencies are available ✓")
    return True


def install_dependencies():
    """Install missing dependencies."""
    logger.info("Installing dependencies...")
    
    try:
        # Install ML requirements
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'ml/requirements.txt'
        ])
        logger.info("Dependencies installed successfully ✓")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def create_test_directories():
    """Create necessary directories for testing."""
    logger.info("Creating test directories...")
    
    directories = [
        'data/raw',
        'data/processed', 
        'data/cache',
        'models/artifacts',
        'outputs',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"  Created: {directory}")
    
    logger.info("Test directories created ✓")


def download_sample_data():
    """Download sample data for testing."""
    logger.info("Downloading sample data...")
    
    try:
        # This would download sample data in a real scenario
        # For now, we'll just create placeholder files
        sample_files = [
            'data/raw/sample_video.mp4',
            'data/raw/sample_annotations.json'
        ]
        
        for file_path in sample_files:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write("# Placeholder file for testing\n")
            logger.info(f"  Created: {file_path}")
        
        logger.info("Sample data prepared ✓")
        
    except Exception as e:
        logger.error(f"Failed to prepare sample data: {e}")
        return False
    
    return True


def run_basic_test():
    """Run a basic system test."""
    logger.info("Running basic system test...")
    
    try:
        # Import and test basic functionality
        from ml.utils.class_mapping import FootballClassMapper
        
        mapper = FootballClassMapper()
        logger.info(f"  Class mapper: {len(mapper.CLASSES)} classes ✓")
        
        # Test detection model creation
        from ml.models.detection import create_detection_model
        model = create_detection_model("yolov8n", 8, "cpu")
        logger.info(f"  Detection model: {type(model).__name__} ✓")
        
        # Test pose estimation
        from ml.models.pose_estimation import MediaPipePoseEstimator
        pose_estimator = MediaPipePoseEstimator()
        logger.info(f"  Pose estimator: {type(pose_estimator).__name__} ✓")
        
        # Test event detection
        from ml.models.event_detection import EventDetector
        event_detector = EventDetector()
        logger.info(f"  Event detector: {type(event_detector).__name__} ✓")
        
        logger.info("Basic system test completed ✓")
        return True
        
    except Exception as e:
        logger.error(f"Basic system test failed: {e}")
        return False


def main():
    """Main setup function."""
    logger.info("Setting up Godseye AI testing environment...")
    logger.info("=" * 50)
    
    # Check Python version
    if not check_python_version():
        logger.error("Python version check failed. Please upgrade Python.")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        logger.warning("Some dependencies are missing. Installing...")
        if not install_dependencies():
            logger.error("Failed to install dependencies. Please install manually.")
            return 1
    
    # Create directories
    create_test_directories()
    
    # Prepare sample data
    if not download_sample_data():
        logger.warning("Failed to prepare sample data. Continuing anyway...")
    
    # Run basic test
    if not run_basic_test():
        logger.error("Basic system test failed. Please check the installation.")
        return 1
    
    logger.info("=" * 50)
    logger.info("Setup completed successfully! ✓")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Test the system: python test_system.py")
    logger.info("2. Test with your video: python test_system.py --video your_video.mp4")
    logger.info("3. Check TESTING.md for detailed instructions")
    logger.info("")
    logger.info("Your Godseye AI system is ready for testing!")
    
    return 0


if __name__ == "__main__":
    exit(main())
