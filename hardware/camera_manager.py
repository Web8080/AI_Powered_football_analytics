#!/usr/bin/env python3
"""
================================================================================
GODSEYE AI - CAMERA MANAGER MODULE
================================================================================

Author: Victor Ibhafidon
Date: January 28, 2025
Version: 1.0.0

DESCRIPTION:
This module provides unified camera management for the Godseye AI sports analytics
platform. It handles connections to various sports cameras including VeoCam,
Raspberry Pi cameras, NVIDIA Jetson cameras, and standard USB cameras. This is
the central hub for all camera operations in the system.

PIPELINE INTEGRATION:
- Connects to: Frontend HardwareManager.tsx component
- Provides: Camera discovery and connection services
- Integrates: With cam_hardware_integration.py for VeoCam support
- Supports: RealTimeDashboard.tsx for live streaming
- Records: Videos for VideoUpload.tsx analysis pipeline
- Manages: Hardware configuration and status monitoring

FEATURES:
- Multi-camera support (USB, VeoCam, Raspberry Pi, NVIDIA Jetson)
- Automatic camera discovery and scanning
- Platform detection (Linux, macOS, Windows, Raspberry Pi, Jetson)
- Threaded frame processing for optimal performance
- Real-time recording with configurable quality settings
- Hardware status monitoring and diagnostics
- Cross-platform compatibility

DEPENDENCIES:
- OpenCV for video capture and processing
- NumPy for array operations
- Threading for concurrent operations
- Platform detection for hardware-specific features
- Queue for frame buffering

USAGE:
    from camera_manager import CameraManager, CameraConfig
    
    # Initialize camera manager
    manager = CameraManager()
    
    # Scan for available cameras
    cameras = manager.scan_cameras()
    
    # Connect to first available camera
    if cameras:
        manager.connect_camera(cameras[0])
        manager.start_recording("match.mp4")

COMPETITOR ANALYSIS:
Based on analysis of industry leaders like VeoCam, Stats Perform, and other
sports analytics platforms. Implements enterprise-grade camera management
with professional features for live sports broadcasting and analysis.

================================================================================
"""

import cv2
import numpy as np
import time
import threading
import queue
import subprocess
import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import platform
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CameraConfig:
    """Camera configuration settings"""
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    quality: int = 90
    auto_focus: bool = True
    auto_exposure: bool = True
    night_mode: bool = False
    stabilization: bool = True

@dataclass
class HardwareInfo:
    """Hardware information"""
    device_type: str  # 'veocam', 'raspberry_pi', 'jetson', 'usb_camera'
    model: str
    capabilities: List[str]
    connection_type: str  # 'usb', 'network', 'gpio'
    status: str  # 'connected', 'disconnected', 'error'

class CameraManager:
    """
    Manages connections to various sports cameras
    Supports VeoCam, Raspberry Pi, NVIDIA Jetson, and standard USB cameras
    """
    
    def __init__(self):
        self.camera = None
        self.config = CameraConfig()
        self.hardware_info = None
        self.is_connected = False
        self.is_recording = False
        self.frame_queue = queue.Queue(maxsize=30)
        self.latest_frame = None
        self.recording_thread = None
        self.frame_thread = None
        self.recording_path = None
        self.video_writer = None
        
        # Platform detection
        self.is_linux = platform.system() == "Linux"
        self.is_raspberry_pi = self._detect_raspberry_pi()
        self.is_jetson = self._detect_jetson()
        
        logger.info(f"Camera Manager initialized - Platform: {platform.system()}, Pi: {self.is_raspberry_pi}, Jetson: {self.is_jetson}")
    
    def _detect_raspberry_pi(self) -> bool:
        """Detect if running on Raspberry Pi"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                return 'BCM' in cpuinfo or 'Raspberry Pi' in cpuinfo
        except:
            return False
    
    def _detect_jetson(self) -> bool:
        """Detect if running on NVIDIA Jetson"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                return 'Jetson' in model
        except:
            return False
    
    def scan_cameras(self) -> List[Dict[str, Any]]:
        """Scan for available cameras"""
        available_cameras = []
        
        # Check USB cameras
        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    available_cameras.append({
                        'index': i,
                        'type': 'usb_camera',
                        'resolution': (width, height),
                        'name': f'USB Camera {i}',
                        'connection': 'usb'
                    })
                cap.release()
        
        # Check for VeoCam (network camera)
        veocam_ip = self._scan_veocam()
        if veocam_ip:
            available_cameras.append({
                'index': -1,  # Special index for network cameras
                'type': 'veocam',
                'resolution': (1920, 1080),
                'name': 'VeoCam 360',
                'connection': 'network',
                'ip': veocam_ip
            })
        
        # Check for Raspberry Pi Camera
        if self.is_raspberry_pi:
            if self._check_pi_camera():
                available_cameras.append({
                    'index': -2,  # Special index for Pi camera
                    'type': 'raspberry_pi',
                    'resolution': (1920, 1080),
                    'name': 'Raspberry Pi Camera',
                    'connection': 'gpio'
                })
        
        logger.info(f"Found {len(available_cameras)} available cameras")
        return available_cameras
    
    def _scan_veocam(self) -> Optional[str]:
        """Scan for VeoCam on network"""
        import socket
        
        # Common VeoCam IP ranges
        ip_ranges = [
            "192.168.1.",  # Common home network
            "192.168.0.",  # Common home network
            "10.0.0.",     # Common office network
            "172.16.0."    # Common office network
        ]
        
        for ip_range in ip_ranges:
            for i in range(1, 255):
                ip = f"{ip_range}{i}"
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.1)
                    result = sock.connect_ex((ip, 8080))  # VeoCam typically uses port 8080
                    sock.close()
                    if result == 0:
                        logger.info(f"Found potential VeoCam at {ip}")
                        return ip
                except:
                    continue
        return None
    
    def _check_pi_camera(self) -> bool:
        """Check if Raspberry Pi camera is available"""
        try:
            if self.is_linux:
                # Check if camera is detected
                result = subprocess.run(['vcgencmd', 'get_camera'], 
                                      capture_output=True, text=True)
                return 'detected=1' in result.stdout
        except:
            pass
        return False
    
    def connect_camera(self, camera_info: Dict[str, Any]) -> bool:
        """Connect to a specific camera"""
        try:
            if camera_info['type'] == 'veocam':
                return self._connect_veocam(camera_info)
            elif camera_info['type'] == 'raspberry_pi':
                return self._connect_pi_camera()
            elif camera_info['type'] == 'usb_camera':
                return self._connect_usb_camera(camera_info['index'])
            else:
                logger.error(f"Unsupported camera type: {camera_info['type']}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to camera: {e}")
            return False
    
    def _connect_veocam(self, camera_info: Dict[str, Any]) -> bool:
        """Connect to VeoCam"""
        try:
            ip = camera_info.get('ip', '192.168.1.100')
            url = f"http://{ip}:8080/video"
            
            self.camera = cv2.VideoCapture(url)
            if self.camera.isOpened():
                self.hardware_info = HardwareInfo(
                    device_type='veocam',
                    model='VeoCam 360',
                    capabilities=['360_view', 'auto_tracking', 'night_vision'],
                    connection_type='network',
                    status='connected'
                )
                self.is_connected = True
                self._start_frame_thread()
                logger.info(f"Connected to VeoCam at {ip}")
                return True
        except Exception as e:
            logger.error(f"Failed to connect to VeoCam: {e}")
        return False
    
    def _connect_pi_camera(self) -> bool:
        """Connect to Raspberry Pi camera"""
        try:
            if self.is_raspberry_pi:
                # Use libcamera for newer Pi OS
                self.camera = cv2.VideoCapture(0)
                if self.camera.isOpened():
                    self.hardware_info = HardwareInfo(
                        device_type='raspberry_pi',
                        model='Raspberry Pi Camera Module',
                        capabilities=['high_resolution', 'low_light'],
                        connection_type='gpio',
                        status='connected'
                    )
                    self.is_connected = True
                    self._start_frame_thread()
                    logger.info("Connected to Raspberry Pi Camera")
                    return True
        except Exception as e:
            logger.error(f"Failed to connect to Pi Camera: {e}")
        return False
    
    def _connect_usb_camera(self, index: int) -> bool:
        """Connect to USB camera"""
        try:
            self.camera = cv2.VideoCapture(index)
            if self.camera.isOpened():
                # Get camera properties
                width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                self.hardware_info = HardwareInfo(
                    device_type='usb_camera',
                    model=f'USB Camera {index}',
                    capabilities=['standard_video'],
                    connection_type='usb',
                    status='connected'
                )
                self.is_connected = True
                self._start_frame_thread()
                logger.info(f"Connected to USB Camera {index} ({width}x{height})")
                return True
        except Exception as e:
            logger.error(f"Failed to connect to USB Camera {index}: {e}")
        return False
    
    def _start_frame_thread(self):
        """Start frame reading thread"""
        if self.frame_thread is None or not self.frame_thread.is_alive():
            self.frame_thread = threading.Thread(target=self._frame_reader, daemon=True)
            self.frame_thread.start()
    
    def _frame_reader(self):
        """Read frames from camera in separate thread"""
        while self.is_connected and self.camera and self.camera.isOpened():
            try:
                ret, frame = self.camera.read()
                if ret:
                    self.latest_frame = frame.copy()
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                else:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Frame reading error: {e}")
                time.sleep(0.1)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from camera"""
        return self.latest_frame
    
    def start_recording(self, output_path: str) -> bool:
        """Start recording video"""
        if not self.is_connected:
            logger.error("Cannot start recording: camera not connected")
            return False
        
        try:
            self.recording_path = output_path
            self.is_recording = True
            
            # Get frame dimensions
            if self.latest_frame is not None:
                height, width = self.latest_frame.shape[:2]
            else:
                width, height = self.config.resolution
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, self.config.fps, (width, height)
            )
            
            # Start recording thread
            self.recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
            self.recording_thread.start()
            
            logger.info(f"Started recording to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False
    
    def _recording_worker(self):
        """Recording worker thread"""
        while self.is_recording and self.video_writer:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    self.video_writer.write(frame)
                else:
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Recording error: {e}")
                break
    
    def stop_recording(self) -> Optional[str]:
        """Stop recording and return path to recorded video"""
        if not self.is_recording:
            return None
        
        self.is_recording = False
        
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        recording_path = self.recording_path
        self.recording_path = None
        
        logger.info(f"Stopped recording: {recording_path}")
        return recording_path
    
    def set_config(self, config: CameraConfig):
        """Update camera configuration"""
        self.config = config
        if self.camera and self.camera.isOpened():
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, config.fps)
    
    def get_status(self) -> Dict[str, Any]:
        """Get camera status"""
        return {
            'connected': self.is_connected,
            'recording': self.is_recording,
            'hardware_info': self.hardware_info.__dict__ if self.hardware_info else None,
            'config': {
                'resolution': self.config.resolution,
                'fps': self.config.fps,
                'quality': self.config.quality
            },
            'frame_queue_size': self.frame_queue.qsize(),
            'latest_frame_available': self.latest_frame is not None
        }
    
    def disconnect(self):
        """Disconnect from camera"""
        if self.is_recording:
            self.stop_recording()
        
        self.is_connected = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        self.latest_frame = None
        self.hardware_info = None
        
        logger.info("Camera disconnected")

# Example usage and testing
if __name__ == "__main__":
    manager = CameraManager()
    
    # Scan for cameras
    cameras = manager.scan_cameras()
    print(f"Found {len(cameras)} cameras:")
    for cam in cameras:
        print(f"  - {cam['name']} ({cam['type']}) at {cam.get('ip', f'index {cam['index']}')}")
    
    # Connect to first available camera
    if cameras:
        print(f"\nConnecting to {cameras[0]['name']}...")
        if manager.connect_camera(cameras[0]):
            print("Connected successfully!")
            
            # Test recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"test_recording_{timestamp}.mp4"
            
            print(f"Starting recording to {output_path}...")
            if manager.start_recording(output_path):
                print("Recording started. Press Ctrl+C to stop...")
                try:
                    time.sleep(10)  # Record for 10 seconds
                except KeyboardInterrupt:
                    pass
                
                recording_path = manager.stop_recording()
                print(f"Recording saved to: {recording_path}")
            
            manager.disconnect()
        else:
            print("Failed to connect to camera")
    else:
        print("No cameras found")
