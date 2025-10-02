#!/usr/bin/env python3
"""
Monitor Advanced Training Progress
Real-time monitoring of transfer learning + curriculum learning
"""

import time
import os
import subprocess
from pathlib import Path
import re

class AdvancedTrainingMonitor:
    """Monitor advanced training progress"""
    
    def __init__(self):
        self.log_file = Path("advanced_training.log")
        self.models_dir = Path("models")
        
    def get_training_status(self):
        """Get current training status"""
        try:
            # Check if process is running
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            is_running = 'advanced_transfer_curriculum_training.py' in result.stdout
            
            if not is_running:
                return "STOPPED"
            
            # Read log file
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()
                
                # Get last few lines
                last_lines = lines[-10:] if len(lines) >= 10 else lines
                
                # Determine current stage
                current_stage = "UNKNOWN"
                for line in reversed(last_lines):
                    if "STAGE 1: TRANSFER LEARNING" in line:
                        current_stage = "STAGE 1: Transfer Learning"
                        break
                    elif "STAGE 2: CURRICULUM LEARNING - EASY" in line:
                        current_stage = "STAGE 2: Curriculum Learning (Easy)"
                        break
                    elif "STAGE 3: CURRICULUM LEARNING - HARD" in line:
                        current_stage = "STAGE 3: Curriculum Learning (Hard)"
                        break
                    elif "EVALUATING MODEL PERFORMANCE" in line:
                        current_stage = "EVALUATION"
                        break
                    elif "ADVANCED TRAINING COMPLETED" in line:
                        current_stage = "COMPLETED"
                        break
                
                return current_stage
            else:
                return "STARTING"
                
        except Exception as e:
            return f"ERROR: {e}"
    
    def get_model_files(self):
        """Get list of saved model files"""
        if not self.models_dir.exists():
            return []
        
        model_files = []
        for file in self.models_dir.glob("*.pt"):
            model_files.append({
                "name": file.name,
                "size": file.stat().st_size,
                "modified": file.stat().st_mtime
            })
        
        return sorted(model_files, key=lambda x: x["modified"], reverse=True)
    
    def get_training_metrics(self):
        """Extract training metrics from log"""
        if not self.log_file.exists():
            return {}
        
        try:
            with open(self.log_file, 'r') as f:
                content = f.read()
            
            metrics = {}
            
            # Extract epoch information
            epoch_matches = re.findall(r'Epoch\s+(\d+)/(\d+)', content)
            if epoch_matches:
                current_epoch, total_epochs = epoch_matches[-1]
                metrics["current_epoch"] = int(current_epoch)
                metrics["total_epochs"] = int(total_epochs)
                metrics["progress_percent"] = (int(current_epoch) / int(total_epochs)) * 100
            
            # Extract loss values
            loss_matches = re.findall(r'box_loss\s+([\d.]+)\s+cls_loss\s+([\d.]+)\s+dfl_loss\s+([\d.]+)', content)
            if loss_matches:
                box_loss, cls_loss, dfl_loss = loss_matches[-1]
                metrics["box_loss"] = float(box_loss)
                metrics["cls_loss"] = float(cls_loss)
                metrics["dfl_loss"] = float(dfl_loss)
            
            # Extract instances
            instance_matches = re.findall(r'Instances\s+(\d+)', content)
            if instance_matches:
                metrics["instances"] = int(instance_matches[-1])
            
            return metrics
            
        except Exception as e:
            return {"error": str(e)}
    
    def display_status(self):
        """Display current training status"""
        print("\n" + "="*70)
        print("🎯 GODSEYE AI - ADVANCED TRAINING MONITOR")
        print("="*70)
        
        # Get status
        status = self.get_training_status()
        print(f"📊 Status: {status}")
        
        # Get metrics
        metrics = self.get_training_metrics()
        if metrics and "error" not in metrics:
            print(f"📈 Progress: {metrics.get('current_epoch', 0)}/{metrics.get('total_epochs', 0)} epochs ({metrics.get('progress_percent', 0):.1f}%)")
            print(f"📉 Losses: Box={metrics.get('box_loss', 0):.3f}, Cls={metrics.get('cls_loss', 0):.3f}, DFL={metrics.get('dfl_loss', 0):.3f}")
            print(f"🎯 Instances: {metrics.get('instances', 0)}")
        
        # Get model files
        models = self.get_model_files()
        if models:
            print(f"💾 Models saved: {len(models)}")
            for model in models[:3]:  # Show latest 3
                size_mb = model['size'] / (1024*1024)
                print(f"   - {model['name']} ({size_mb:.1f}MB)")
        
        print("="*70)
        
        # Provide recommendations based on status
        if "STAGE 1" in status:
            print("🔄 Currently in Transfer Learning phase...")
            print("   ✅ Using COCO pretrained weights")
            print("   ✅ Freezing backbone layers")
            print("   ⏱️ Expected: 30 epochs")
        elif "STAGE 2" in status:
            print("🎓 Currently in Curriculum Learning (Easy) phase...")
            print("   ✅ Training on clear, well-lit images")
            print("   ✅ Unfrozen all layers")
            print("   ⏱️ Expected: 25 epochs")
        elif "STAGE 3" in status:
            print("🎯 Currently in Curriculum Learning (Hard) phase...")
            print("   ✅ Training on challenging conditions")
            print("   ✅ Aggressive augmentation")
            print("   ⏱️ Expected: 45 epochs")
        elif status == "COMPLETED":
            print("🎉 Training completed!")
            print("   ✅ All stages finished")
            print("   📊 Check final model performance")
        elif status == "STOPPED":
            print("⚠️ Training stopped")
            print("   🔍 Check logs for errors")
        
        return status
    
    def monitor_continuously(self, interval=30):
        """Monitor training continuously"""
        print("🚀 Starting continuous monitoring...")
        print(f"⏱️ Update interval: {interval} seconds")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                status = self.display_status()
                
                if status == "COMPLETED":
                    print("\n🎉 Training completed successfully!")
                    break
                elif status == "STOPPED":
                    print("\n⚠️ Training stopped. Check logs for details.")
                    break
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")

if __name__ == "__main__":
    monitor = AdvancedTrainingMonitor()
    
    # Single status check
    status = monitor.display_status()
    
    # Ask if user wants continuous monitoring
    if status not in ["COMPLETED", "STOPPED"]:
        response = input("\n🔄 Start continuous monitoring? (y/n): ")
        if response.lower() == 'y':
            monitor.monitor_continuously()
