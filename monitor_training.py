#!/usr/bin/env python3
"""
Real-time training monitor for Godseye AI
Shows live progress of the training pipeline
"""

import os
import time
import subprocess
import glob
from pathlib import Path

def get_process_info():
    """Get information about the training process"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'robust_local_training.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 11:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    time_str = parts[9]
                    return {
                        'pid': pid,
                        'cpu': cpu,
                        'memory': mem,
                        'runtime': time_str,
                        'status': 'RUNNING'
                    }
    except:
        pass
    return {'status': 'NOT_RUNNING'}

def get_disk_usage():
    """Get disk usage of SoccerNet data"""
    try:
        result = subprocess.run(['du', '-sh', 'data/SoccerNet/'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split()[0]
    except:
        pass
    return "Unknown"

def count_files():
    """Count downloaded files"""
    try:
        # Count videos
        videos = len(glob.glob('data/SoccerNet/**/*.mkv', recursive=True))
        
        # Count labels
        labels = len(glob.glob('data/SoccerNet/**/Labels-v2.json', recursive=True))
        
        # Count extracted images
        images = len(glob.glob('data/**/*.jpg', recursive=True)) + len(glob.glob('data/**/*.png', recursive=True))
        
        return {
            'videos': videos,
            'labels': labels,
            'images': images
        }
    except:
        return {'videos': 0, 'labels': 0, 'images': 0}

def get_recent_activity():
    """Get recent file activity"""
    try:
        # Find recently modified files
        result = subprocess.run(['find', 'data/SoccerNet/', '-type', 'f', '-mmin', '-5'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            files = result.stdout.strip().split('\n')
            return [f for f in files if f]
    except:
        pass
    return []

def monitor_training():
    """Main monitoring loop"""
    print("🎯 GODSEYE AI - REAL-TIME TRAINING MONITOR")
    print("=" * 60)
    
    while True:
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🎯 GODSEYE AI - REAL-TIME TRAINING MONITOR")
        print("=" * 60)
        print(f"⏰ Current Time: {time.strftime('%H:%M:%S')}")
        print()
        
        # Process status
        process_info = get_process_info()
        if process_info['status'] == 'RUNNING':
            print("🔄 TRAINING STATUS: ACTIVE")
            print(f"   PID: {process_info['pid']}")
            print(f"   CPU: {process_info['cpu']}%")
            print(f"   Memory: {process_info['memory']}%")
            print(f"   Runtime: {process_info['runtime']}")
        else:
            print("❌ TRAINING STATUS: NOT RUNNING")
        
        print()
        
        # Data progress
        disk_usage = get_disk_usage()
        file_counts = count_files()
        
        print("📊 DATA PROGRESS:")
        print(f"   📁 SoccerNet Size: {disk_usage}")
        print(f"   🎬 Videos Downloaded: {file_counts['videos']}")
        print(f"   📋 Labels Downloaded: {file_counts['labels']}")
        print(f"   🖼️  Images Extracted: {file_counts['images']}")
        
        print()
        
        # Current phase
        if file_counts['videos'] > 0 and file_counts['images'] == 0:
            print("🔄 CURRENT PHASE: Video Processing")
            print("   Converting videos to training images...")
        elif file_counts['images'] > 0:
            print("🔄 CURRENT PHASE: Model Training")
            print("   Training YOLOv8 model...")
        else:
            print("🔄 CURRENT PHASE: Data Download")
            print("   Downloading SoccerNet dataset...")
        
        print()
        
        # Recent activity
        recent_files = get_recent_activity()
        if recent_files:
            print("📝 RECENT ACTIVITY:")
            for file in recent_files[-3:]:  # Show last 3 files
                filename = os.path.basename(file)
                print(f"   📄 {filename}")
        
        print()
        print("Press Ctrl+C to stop monitoring")
        print("=" * 60)
        
        time.sleep(5)  # Update every 5 seconds

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
