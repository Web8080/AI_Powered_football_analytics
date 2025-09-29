#!/usr/bin/env python3
"""
================================================================================
GODSEYE AI - HARDWARE SETUP SCRIPT
================================================================================

Author: Victor Ibhafidon
Date: January 28, 2025
Version: 1.0.0

DESCRIPTION:
This script provides automated hardware setup for the Godseye AI sports analytics
platform. It handles installation of dependencies, camera permissions, system
configuration, and service setup for various sports cameras and hardware platforms.
This is the one-command setup solution for the entire hardware pipeline.

PIPELINE INTEGRATION:
- Sets up: All hardware dependencies for camera_manager.py
- Configures: System permissions for cam_hardware_integration.py
- Creates: Hardware configuration files for the entire pipeline
- Enables: System services for automated camera management
- Prepares: Environment for Frontend HardwareManager.tsx component
- Supports: All camera types (VeoCam, USB, Raspberry Pi, NVIDIA Jetson)

FEATURES:
- Cross-platform support (Linux, macOS, Windows)
- Automatic dependency installation
- Camera permission configuration
- Hardware configuration file generation
- System service setup (systemd for Linux)
- Camera discovery and enumeration
- Platform-specific optimizations

DEPENDENCIES:
- Python 3.8+ with pip
- Platform-specific package managers (apt, brew, etc.)
- System administration privileges for service setup
- Network access for package downloads

USAGE:
    # Run complete hardware setup
    python setup_hardware.py
    
    # Or import and use programmatically
    from setup_hardware import HardwareSetup
    setup = HardwareSetup()
    setup.run_setup()

COMPETITOR ANALYSIS:
Based on analysis of industry-standard hardware setup procedures from VeoCam,
Stats Perform, and other professional sports analytics platforms. Implements
enterprise-grade setup procedures for production deployment.

================================================================================
"""

import os
import sys
import subprocess
import platform
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HardwareSetup:
    """
    Automated hardware setup for Godseye AI
    Supports VeoCam, Raspberry Pi, NVIDIA Jetson, and USB cameras
    """
    
    def __init__(self):
        self.platform = platform.system()
        self.is_linux = self.platform == "Linux"
        self.is_raspberry_pi = self._detect_raspberry_pi()
        self.is_jetson = self._detect_jetson()
        self.setup_dir = Path(__file__).parent
        self.config_file = self.setup_dir / "hardware_config.json"
        
        logger.info(f"Hardware Setup initialized - Platform: {self.platform}")
        logger.info(f"Raspberry Pi: {self.is_raspberry_pi}, Jetson: {self.is_jetson}")
    
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
    
    def install_dependencies(self):
        """Install required dependencies"""
        logger.info("Installing dependencies...")
        
        # Python packages
        python_packages = [
            'opencv-python',
            'numpy',
            'requests',
            'flask',
            'flask-cors',
            'ultralytics',
            'torch',
            'torchvision',
            'pillow',
            'psutil'
        ]
        
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install'] + python_packages, 
                         check=True, capture_output=True)
            logger.info("Python packages installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Python packages: {e}")
            return False
        
        # Platform-specific dependencies
        if self.is_linux:
            self._install_linux_dependencies()
        elif self.platform == "Darwin":  # macOS
            self._install_macos_dependencies()
        elif self.platform == "Windows":
            self._install_windows_dependencies()
        
        return True
    
    def _install_linux_dependencies(self):
        """Install Linux-specific dependencies"""
        logger.info("Installing Linux dependencies...")
        
        # Update package list
        subprocess.run(['sudo', 'apt', 'update'], check=True)
        
        # Install system packages
        packages = [
            'python3-opencv',
            'python3-pip',
            'python3-dev',
            'libopencv-dev',
            'ffmpeg',
            'v4l-utils',
            'usbutils'
        ]
        
        if self.is_raspberry_pi:
            packages.extend([
                'python3-picamera2',
                'libcamera-tools',
                'gpiozero'
            ])
        
        if self.is_jetson:
            packages.extend([
                'nvidia-jetpack',
                'libnvinfer-dev'
            ])
        
        try:
            subprocess.run(['sudo', 'apt', 'install', '-y'] + packages, check=True)
            logger.info("Linux dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Linux dependencies: {e}")
    
    def _install_macos_dependencies(self):
        """Install macOS-specific dependencies"""
        logger.info("Installing macOS dependencies...")
        
        # Check if Homebrew is installed
        try:
            subprocess.run(['brew', '--version'], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.info("Installing Homebrew...")
            subprocess.run([
                '/bin/bash', '-c', 
                '$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)'
            ], check=True)
        
        # Install packages via Homebrew
        packages = ['opencv', 'ffmpeg', 'python@3.9']
        
        try:
            subprocess.run(['brew', 'install'] + packages, check=True)
            logger.info("macOS dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install macOS dependencies: {e}")
    
    def _install_windows_dependencies(self):
        """Install Windows-specific dependencies"""
        logger.info("Installing Windows dependencies...")
        
        # Windows dependencies are mostly handled by pip
        # Additional setup might be needed for OpenCV
        logger.info("Windows dependencies handled by pip")
    
    def setup_camera_permissions(self):
        """Setup camera permissions"""
        logger.info("Setting up camera permissions...")
        
        if self.is_linux:
            # Add user to video group
            try:
                subprocess.run(['sudo', 'usermod', '-a', '-G', 'video', os.getenv('USER')], 
                             check=True)
                logger.info("Added user to video group")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to add user to video group: {e}")
        
        elif self.platform == "Darwin":  # macOS
            # macOS camera permissions are handled by the system
            logger.info("macOS camera permissions handled by system")
        
        elif self.platform == "Windows":
            # Windows camera permissions are handled by the system
            logger.info("Windows camera permissions handled by system")
    
    def create_hardware_config(self):
        """Create hardware configuration file"""
        logger.info("Creating hardware configuration...")
        
        config = {
            'platform': self.platform,
            'is_raspberry_pi': self.is_raspberry_pi,
            'is_jetson': self.is_jetson,
            'cameras': {
                'usb_cameras': [],
                'veocam': {
                    'enabled': False,
                    'ip': None,
                    'port': 8080
                },
                'raspberry_pi_camera': {
                    'enabled': self.is_raspberry_pi,
                    'resolution': [1920, 1080],
                    'fps': 30
                }
            },
            'recording': {
                'output_directory': str(self.setup_dir / 'recordings'),
                'default_format': 'mp4',
                'quality': 90
            },
            'streaming': {
                'enabled': True,
                'port': 8080,
                'resolution': [1920, 1080],
                'fps': 30
            }
        }
        
        # Create recordings directory
        recordings_dir = Path(config['recording']['output_directory'])
        recordings_dir.mkdir(exist_ok=True)
        
        # Save configuration
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Hardware configuration saved to {self.config_file}")
        return config
    
    def scan_cameras(self):
        """Scan for available cameras"""
        logger.info("Scanning for cameras...")
        
        cameras = []
        
        # Check USB cameras
        for i in range(10):
            try:
                import cv2
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        height, width = frame.shape[:2]
                        cameras.append({
                            'index': i,
                            'type': 'usb_camera',
                            'resolution': [width, height],
                            'name': f'USB Camera {i}'
                        })
                    cap.release()
            except Exception as e:
                logger.debug(f"Error checking camera {i}: {e}")
        
        # Check VeoCam
        veocam_ip = self._scan_veocam()
        if veocam_ip:
            cameras.append({
                'index': -1,
                'type': 'veocam',
                'resolution': [1920, 1080],
                'name': 'VeoCam 360',
                'ip': veocam_ip
            })
        
        # Check Raspberry Pi Camera
        if self.is_raspberry_pi and self._check_pi_camera():
            cameras.append({
                'index': -2,
                'type': 'raspberry_pi',
                'resolution': [1920, 1080],
                'name': 'Raspberry Pi Camera'
            })
        
        logger.info(f"Found {len(cameras)} cameras")
        return cameras
    
    def _scan_veocam(self):
        """Scan for VeoCam on network"""
        import socket
        
        ip_ranges = ["192.168.1.", "192.168.0.", "10.0.0.", "172.16.0."]
        
        for ip_range in ip_ranges:
            for i in range(1, 255):
                ip = f"{ip_range}{i}"
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.1)
                    result = sock.connect_ex((ip, 8080))
                    sock.close()
                    if result == 0:
                        return ip
                except:
                    continue
        return None
    
    def _check_pi_camera(self):
        """Check if Raspberry Pi camera is available"""
        try:
            if self.is_linux:
                result = subprocess.run(['vcgencmd', 'get_camera'], 
                                      capture_output=True, text=True)
                return 'detected=1' in result.stdout
        except:
            pass
        return False
    
    def setup_services(self):
        """Setup system services"""
        logger.info("Setting up system services...")
        
        if self.is_linux:
            self._setup_linux_services()
        else:
            logger.info("Service setup not required for this platform")
    
    def _setup_linux_services(self):
        """Setup Linux system services"""
        # Create systemd service for Godseye AI
        service_content = f"""[Unit]
Description=Godseye AI Camera Service
After=network.target

[Service]
Type=simple
User={os.getenv('USER')}
WorkingDirectory={self.setup_dir.parent}
ExecStart={sys.executable} {self.setup_dir}/camera_manager.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_file = Path('/etc/systemd/system/godseye-camera.service')
        
        try:
            with open(service_file, 'w') as f:
                f.write(service_content)
            
            subprocess.run(['sudo', 'systemctl', 'daemon-reload'], check=True)
            subprocess.run(['sudo', 'systemctl', 'enable', 'godseye-camera'], check=True)
            
            logger.info("Systemd service created and enabled")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup systemd service: {e}")
    
    def run_setup(self):
        """Run complete hardware setup"""
        logger.info("Starting Godseye AI hardware setup...")
        
        try:
            # Install dependencies
            if not self.install_dependencies():
                logger.error("Failed to install dependencies")
                return False
            
            # Setup camera permissions
            self.setup_camera_permissions()
            
            # Create hardware configuration
            config = self.create_hardware_config()
            
            # Scan for cameras
            cameras = self.scan_cameras()
            config['cameras']['usb_cameras'] = cameras
            
            # Update configuration with found cameras
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Setup services
            self.setup_services()
            
            logger.info("Hardware setup completed successfully!")
            logger.info(f"Found {len(cameras)} cameras:")
            for camera in cameras:
                logger.info(f"  - {camera['name']} ({camera['type']})")
            
            return True
            
        except Exception as e:
            logger.error(f"Hardware setup failed: {e}")
            return False

def main():
    """Main setup function"""
    print("🚀 Godseye AI Hardware Setup")
    print("=" * 50)
    
    setup = HardwareSetup()
    
    if setup.run_setup():
        print("\n✅ Hardware setup completed successfully!")
        print("\nNext steps:")
        print("1. Connect your camera (VeoCam, USB camera, or Pi camera)")
        print("2. Run: python hardware/camera_manager.py")
        print("3. Open the frontend and go to Hardware tab")
        print("4. Scan and connect to your camera")
    else:
        print("\n❌ Hardware setup failed!")
        print("Please check the logs and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
