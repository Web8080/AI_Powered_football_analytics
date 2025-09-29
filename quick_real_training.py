#!/usr/bin/env python3
"""
================================================================================
GODSEYE AI - QUICK REAL TRAINING
================================================================================

Author: Victor Ibhafidon
Date: January 28, 2025
Version: 1.0.0

DESCRIPTION:
Quick training using real football data extracted from Madrid vs City video.
This should improve detection accuracy on real footage.

USAGE:
    python quick_real_training.py
"""

import os
import time
from pathlib import Path
from ultralytics import YOLO

def main():
    """Quick training with real data"""
    print("🏈 Godseye AI - Quick Real Training")
    print("=" * 40)
    
    # Check if real data exists
    data_path = "data/real_football/dataset.yaml"
    if not os.path.exists(data_path):
        print(f"❌ Real training data not found: {data_path}")
        print("Please run create_real_training_data.py first")
        return
    
    print("🚀 Starting quick training with real data...")
    
    # Initialize model
    model = YOLO('yolov8n.pt')
    
    # Quick training (5 minutes max)
    start_time = time.time()
    max_time = 5 * 60  # 5 minutes
    
    try:
        results = model.train(
            data=data_path,
            epochs=3,  # Very quick training
            imgsz=640,
            batch=8,
            device='cpu',
            project='training_results',
            name='real_football_quick',
            save=True,
            save_period=1,
            plots=False,  # Skip plots for speed
            verbose=True,
            patience=2,  # Early stopping
            workers=2
        )
        
        # Export model
        model_path = "models/yolov8_real_football.pt"
        model.save(model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Export ONNX
        try:
            onnx_path = "models/yolov8_real_football.onnx"
            model.export(format='onnx', imgsz=640)
            exported_onnx = Path("training_results/real_football_quick/weights/best.onnx")
            if exported_onnx.exists():
                exported_onnx.rename(onnx_path)
                print(f"✅ ONNX model saved: {onnx_path}")
        except Exception as e:
            print(f"⚠️ ONNX export failed: {e}")
        
        elapsed_time = time.time() - start_time
        print(f"\n🎉 Quick training completed in {elapsed_time/60:.1f} minutes!")
        print("🚀 Ready to test on real footage!")
        
        return model_path
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

if __name__ == "__main__":
    main()
