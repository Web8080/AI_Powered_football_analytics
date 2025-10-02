#!/usr/bin/env python3
"""
Stop current training and restart with fixed download limits
"""

import os
import signal
import subprocess
import time

def stop_current_training():
    """Stop the current training process"""
    print("🛑 Stopping current training process...")
    
    try:
        # Find the training process
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        for line in lines:
            if 'robust_local_training.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 2:
                    pid = int(parts[1])
                    print(f"📋 Found training process PID: {pid}")
                    
                    # Send SIGTERM to gracefully stop
                    os.kill(pid, signal.SIGTERM)
                    print(f"✅ Sent SIGTERM to process {pid}")
                    
                    # Wait a bit for graceful shutdown
                    time.sleep(5)
                    
                    # Check if still running
                    try:
                        os.kill(pid, 0)  # Check if process exists
                        print(f"⚠️ Process still running, sending SIGKILL...")
                        os.kill(pid, signal.SIGKILL)
                        print(f"✅ Sent SIGKILL to process {pid}")
                    except ProcessLookupError:
                        print(f"✅ Process {pid} stopped gracefully")
                    
                    return True
        
        print("ℹ️ No training process found")
        return False
        
    except Exception as e:
        print(f"❌ Error stopping training: {e}")
        return False

def cleanup_excess_data():
    """Clean up excess downloaded data"""
    print("🧹 Cleaning up excess data...")
    
    try:
        soccernet_dir = "data/SoccerNet"
        if not os.path.exists(soccernet_dir):
            print("ℹ️ No SoccerNet data to clean up")
            return
        
        # Find all video files
        video_files = []
        for root, dirs, files in os.walk(soccernet_dir):
            for file in files:
                if file.endswith('.mkv'):
                    video_files.append(os.path.join(root, file))
        
        print(f"📹 Found {len(video_files)} video files")
        
        # Keep only first 10 videos (5 games × 2 cameras)
        if len(video_files) > 10:
            excess_videos = video_files[10:]
            print(f"🗑️ Removing {len(excess_videos)} excess video files")
            
            for video in excess_videos:
                try:
                    os.remove(video)
                    print(f"✅ Removed: {os.path.basename(video)}")
                except Exception as e:
                    print(f"⚠️ Failed to remove {video}: {e}")
        
        # Check final size
        total_size = 0
        for root, dirs, files in os.walk(soccernet_dir):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
        
        size_gb = total_size / (1024**3)
        print(f"💾 Final dataset size: {size_gb:.1f}GB")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

def restart_training():
    """Restart training with fixed script"""
    print("🚀 Restarting training with fixed download limits...")
    
    try:
        # Set environment variable
        env = os.environ.copy()
        env['SOCCERNET_PASSWORD'] = 's0cc3rn3t'
        
        # Start new training process
        process = subprocess.Popen(
            ['python', 'robust_local_training.py'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print(f"✅ Started new training process PID: {process.pid}")
        print("📊 Training will now be limited to 5 games maximum")
        print("💾 Disk usage will be monitored and limited")
        
        return process
        
    except Exception as e:
        print(f"❌ Error restarting training: {e}")
        return None

def main():
    """Main function"""
    print("🎯 GODSEYE AI - TRAINING RESTART UTILITY")
    print("=" * 50)
    
    # Stop current training
    if stop_current_training():
        time.sleep(2)
    
    # Clean up excess data
    cleanup_excess_data()
    
    # Restart training
    process = restart_training()
    
    if process:
        print("\n✅ Training restarted successfully!")
        print("📊 The new training will:")
        print("   - Download only 5 games maximum")
        print("   - Monitor disk space")
        print("   - Clean up excess data automatically")
        print("   - Extract frames and start training")
    else:
        print("\n❌ Failed to restart training")

if __name__ == "__main__":
    main()
