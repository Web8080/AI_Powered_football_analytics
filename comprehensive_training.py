#!/usr/bin/env python3
"""
================================================================================
GODSEYE AI - COMPREHENSIVE TRAINING
================================================================================

Author: Victor Ibhafidon
Date: January 28, 2025
Version: 2.0.0

DESCRIPTION:
Comprehensive training using real 90-minute football match data.
This should significantly improve detection accuracy on real footage.

USAGE:
    python comprehensive_training.py
"""

import os
import time
from pathlib import Path
from ultralytics import YOLO

def main():
    """Comprehensive training with real match data"""
    print("🏈 Godseye AI - Comprehensive Training")
    print("=" * 45)
    
    # Check if comprehensive data exists
    data_path = "data/comprehensive_football/dataset.yaml"
    if not os.path.exists(data_path):
        print(f"❌ Comprehensive training data not found: {data_path}")
        print("Please run create_comprehensive_training.py first")
        return
    
    print("🚀 Starting comprehensive training with 90-minute match data...")
    
    # Initialize model
    model = YOLO('yolov8n.pt')
    
    # Comprehensive training (10 minutes max)
    start_time = time.time()
    max_time = 10 * 60  # 10 minutes
    
    try:
        results = model.train(
            data=data_path,
            epochs=10,  # More epochs for better learning
            imgsz=640,
            batch=8,
            device='cpu',
            project='training_results',
            name='comprehensive_football',
            save=True,
            save_period=2,
            plots=True,  # Enable plots for better monitoring
            verbose=True,
            patience=5,  # More patience for better convergence
            workers=2,
            # Enhanced training parameters
            lr0=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            # Data augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0
        )
        
        # Export model
        model_path = "models/yolov8_comprehensive.pt"
        model.save(model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Export ONNX
        try:
            onnx_path = "models/yolov8_comprehensive.onnx"
            model.export(format='onnx', imgsz=640)
            exported_onnx = Path("training_results/comprehensive_football/weights/best.onnx")
            if exported_onnx.exists():
                exported_onnx.rename(onnx_path)
                print(f"✅ ONNX model saved: {onnx_path}")
        except Exception as e:
            print(f"⚠️ ONNX export failed: {e}")
        
        elapsed_time = time.time() - start_time
        print(f"\n🎉 Comprehensive training completed in {elapsed_time/60:.1f} minutes!")
        print("🚀 Ready to test on real footage!")
        print("\n📊 Training Summary:")
        print("  ✅ 90-minute match data")
        print("  ✅ First half + Second half")
        print("  ✅ High activity moments")
        print("  ✅ Advertisement filtering")
        print("  ✅ Enhanced detection methods")
        
        return model_path
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

if __name__ == "__main__":
    main()
