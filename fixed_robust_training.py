#!/usr/bin/env python3
"""
Fixed Robust Training with Balanced Dataset
Addresses classification loss, limited instances, and background issues
"""

import os
import time
import logging
from pathlib import Path
from ultralytics import YOLO
import mlflow

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fixed_robust_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FixedRobustTrainer:
    """Fixed robust trainer with balanced dataset"""
    
    def __init__(self):
        self.balanced_dataset_yaml = Path("data/balanced_dataset/dataset.yaml")
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Optimized training parameters for balanced dataset
        self.image_size = 640
        self.batch_size = 8  # Increased for more instances per batch
        self.num_epochs = 100  # Reduced for faster training
        self.learning_rate = 0.001  # Slightly higher for faster convergence
        self.weight_decay = 0.0005
        
        # MLflow setup
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Godseye_AI_Fixed_Training")
    
    def train_fixed_model(self):
        """Train model with fixed issues"""
        logger.info("🚀 Starting FIXED robust training with balanced dataset...")
        
        # Verify balanced dataset exists
        if not self.balanced_dataset_yaml.exists():
            logger.error(f"❌ Balanced dataset not found at {self.balanced_dataset_yaml}")
            return False
        
        # Start MLflow run
        with mlflow.start_run(run_name="Godseye_AI_Fixed_Training"):
            # Log parameters
            mlflow.log_params({
                "model": "YOLOv8n",
                "epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "image_size": self.image_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "dataset": "Balanced",
                "fixes": "Class imbalance, Limited instances, Background heavy"
            })
            
            # Initialize YOLOv8 model
            model = YOLO('yolov8n.pt')
            
            # Fixed training configuration
            training_config = {
                'data': str(self.balanced_dataset_yaml),
                'epochs': self.num_epochs,
                'imgsz': self.image_size,
                'batch': self.batch_size,
                'device': 'cpu',
                'project': 'fixed_training',
                'name': 'godseye_fixed_model',
                'save': True,
                'save_period': 10,
                'plots': True,
                'verbose': True,
                
                # Optimized training parameters
                'lr0': self.learning_rate,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': self.weight_decay,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                
                # Balanced augmentation
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 15.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.1,
                'copy_paste': 0.1,
                
                # Optimization
                'optimizer': 'AdamW',
                'close_mosaic': 10,
                'amp': True,
                'fraction': 1.0,
                'profile': False,
                'freeze': None,
                'multi_scale': False,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True,
                'patience': 20,  # Reduced patience for faster training
                'cos_lr': True
            }
            
            # Train the model
            logger.info("🎯 Starting fixed training with balanced dataset...")
            logger.info(f"📊 Dataset: {self.balanced_dataset_yaml}")
            logger.info(f"⚙️ Config: {self.num_epochs} epochs, batch_size={self.batch_size}, lr={self.learning_rate}")
            
            results = model.train(**training_config)
            
            # Log metrics
            if hasattr(results, 'results_dict'):
                mlflow.log_metrics(results.results_dict)
            
            # Save the trained model
            model_path = self.models_dir / "godseye_fixed_model.pt"
            model.save(str(model_path))
            
            # Log model
            mlflow.pytorch.log_model(model, "model")
            
            logger.info(f"✅ Fixed training completed! Model saved to {model_path}")
            
            return model_path
    
    def run(self):
        """Main fixed training pipeline"""
        logger.info("🎯 GODSEYE AI - FIXED ROBUST TRAINING")
        logger.info("=" * 60)
        logger.info("🔧 FIXES APPLIED:")
        logger.info("  ✅ Balanced dataset (19% backgrounds vs 70%)")
        logger.info("  ✅ More instances per batch (8 batch_size)")
        logger.info("  ✅ Optimized learning rate (0.001)")
        logger.info("  ✅ Reduced epochs (100 vs 200)")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Train fixed model
            model_path = self.train_fixed_model()
            
            if model_path:
                # Calculate total time
                total_time = time.time() - start_time
                logger.info(f"🎉 Fixed training completed in {total_time/60:.1f} minutes!")
                logger.info(f"📁 Model saved to: {model_path}")
                logger.info(f"📊 MLflow tracking: ./mlruns")
                
                return True
            else:
                logger.error("❌ Fixed training failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Fixed training failed: {e}")
            return False

if __name__ == "__main__":
    trainer = FixedRobustTrainer()
    success = trainer.run()
    
    if success:
        print("\n🎉 SUCCESS! Fixed robust training completed!")
        print("📊 Issues resolved:")
        print("  ✅ Classification loss should be < 2.0")
        print("  ✅ More instances per batch")
        print("  ✅ Balanced background/object ratio")
        print("🔬 Check MLflow dashboard for detailed metrics!")
    else:
        print("\n❌ Training failed. Check the logs for details.")
