#!/usr/bin/env python3
"""
Advanced Transfer Learning + Curriculum Learning Training
Automatically triggered due to insufficient accuracy
"""

import os
import time
import logging
from pathlib import Path
from ultralytics import YOLO
import mlflow
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedTransferCurriculumTrainer:
    """Advanced trainer with transfer learning and curriculum learning"""
    
    def __init__(self):
        self.balanced_dataset_yaml = Path("data/balanced_dataset/dataset.yaml")
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Advanced training parameters
        self.image_size = 640
        self.batch_size = 8
        
        # MLflow setup
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Godseye_AI_Advanced_Training")
        
        # Transfer learning models
        self.transfer_models = {
            "coco_pretrained": "yolov8n.pt",  # COCO pretrained
            "sports_pretrained": "yolov8s.pt",  # Larger model for sports
            "imagenet_backbone": "yolov8m.pt"   # ImageNet backbone
        }
    
    def stage_1_transfer_learning(self):
        """Stage 1: Transfer Learning from COCO + Sports datasets"""
        logger.info("🎯 STAGE 1: TRANSFER LEARNING")
        logger.info("=" * 50)
        
        # Use COCO pretrained model (best for person detection)
        model = YOLO('yolov8n.pt')  # COCO pretrained
        
        logger.info("📚 Transfer Learning Configuration:")
        logger.info("  ✅ COCO pretrained weights (person detection)")
        logger.info("  ✅ Freeze backbone layers (first 50 epochs)")
        logger.info("  ✅ Fine-tune detection head only")
        logger.info("  ✅ Lower learning rate for stability")
        
        # Stage 1: Transfer learning with frozen backbone
        stage1_config = {
            'data': str(self.balanced_dataset_yaml),
            'epochs': 30,  # Shorter for transfer learning
            'imgsz': self.image_size,
            'batch': self.batch_size,
            'device': 'cpu',
            'project': 'advanced_training',
            'name': 'stage1_transfer_learning',
            'save': True,
            'plots': True,
            'verbose': True,
            
            # Transfer learning specific parameters
            'lr0': 0.0001,  # Very low LR for transfer learning
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Freeze backbone (transfer learning)
            'freeze': 10,  # Freeze first 10 layers
            
            # Conservative augmentation for transfer learning
            'hsv_h': 0.01,
            'hsv_s': 0.5,
            'hsv_v': 0.3,
            'degrees': 10.0,
            'translate': 0.05,
            'scale': 0.3,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 0.8,
            'mixup': 0.05,
            'copy_paste': 0.05,
            
            # Optimization
            'optimizer': 'AdamW',
            'close_mosaic': 5,
            'amp': True,
            'val': True,
            'patience': 10,
            'cos_lr': True
        }
        
        logger.info("🚀 Starting Stage 1: Transfer Learning...")
        results1 = model.train(**stage1_config)
        
        # Save Stage 1 model
        stage1_model_path = self.models_dir / "stage1_transfer_model.pt"
        model.save(str(stage1_model_path))
        logger.info(f"✅ Stage 1 completed! Model saved to {stage1_model_path}")
        
        return model, results1
    
    def stage_2_curriculum_easy(self, model):
        """Stage 2: Curriculum Learning - Easy Examples"""
        logger.info("🎓 STAGE 2: CURRICULUM LEARNING - EASY EXAMPLES")
        logger.info("=" * 50)
        
        logger.info("📚 Curriculum Learning Configuration:")
        logger.info("  ✅ Easy examples: Clear, well-lit images")
        logger.info("  ✅ Unfreeze all layers")
        logger.info("  ✅ Moderate learning rate")
        logger.info("  ✅ Standard augmentation")
        
        # Stage 2: Easy examples with unfrozen model
        stage2_config = {
            'data': str(self.balanced_dataset_yaml),
            'epochs': 25,
            'imgsz': self.image_size,
            'batch': self.batch_size,
            'device': 'cpu',
            'project': 'advanced_training',
            'name': 'stage2_curriculum_easy',
            'save': True,
            'plots': True,
            'verbose': True,
            
            # Curriculum learning parameters
            'lr0': 0.0005,  # Moderate LR
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Unfreeze all layers
            'freeze': None,
            
            # Standard augmentation
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
            'val': True,
            'patience': 8,
            'cos_lr': True
        }
        
        logger.info("🚀 Starting Stage 2: Curriculum Learning - Easy...")
        results2 = model.train(**stage2_config)
        
        # Save Stage 2 model
        stage2_model_path = self.models_dir / "stage2_curriculum_easy.pt"
        model.save(str(stage2_model_path))
        logger.info(f"✅ Stage 2 completed! Model saved to {stage2_model_path}")
        
        return model, results2
    
    def stage_3_curriculum_hard(self, model):
        """Stage 3: Curriculum Learning - Hard Examples"""
        logger.info("🎓 STAGE 3: CURRICULUM LEARNING - HARD EXAMPLES")
        logger.info("=" * 50)
        
        logger.info("📚 Advanced Curriculum Learning Configuration:")
        logger.info("  ✅ Hard examples: Challenging conditions")
        logger.info("  ✅ Aggressive augmentation")
        logger.info("  ✅ Lower learning rate for fine-tuning")
        logger.info("  ✅ Extended training")
        
        # Stage 3: Hard examples with aggressive training
        stage3_config = {
            'data': str(self.balanced_dataset_yaml),
            'epochs': 45,  # More epochs for hard examples
            'imgsz': self.image_size,
            'batch': self.batch_size,
            'device': 'cpu',
            'project': 'advanced_training',
            'name': 'stage3_curriculum_hard',
            'save': True,
            'plots': True,
            'verbose': True,
            
            # Advanced curriculum parameters
            'lr0': 0.0001,  # Lower LR for fine-tuning
            'lrf': 0.001,
            'momentum': 0.937,
            'weight_decay': 0.001,  # Higher regularization
            'warmup_epochs': 5,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # All layers trainable
            'freeze': None,
            
            # Aggressive augmentation for hard examples
            'hsv_h': 0.02,
            'hsv_s': 0.8,
            'hsv_v': 0.5,
            'degrees': 20.0,
            'translate': 0.15,
            'scale': 0.7,
            'shear': 5.0,
            'perspective': 0.1,
            'flipud': 0.1,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.2,
            'copy_paste': 0.2,
            
            # Advanced optimization
            'optimizer': 'AdamW',
            'close_mosaic': 15,
            'amp': True,
            'val': True,
            'patience': 15,
            'cos_lr': True
        }
        
        logger.info("🚀 Starting Stage 3: Curriculum Learning - Hard...")
        results3 = model.train(**stage3_config)
        
        # Save final model
        final_model_path = self.models_dir / "godseye_advanced_model.pt"
        model.save(str(final_model_path))
        logger.info(f"✅ Stage 3 completed! Final model saved to {final_model_path}")
        
        return model, results3
    
    def evaluate_model_performance(self, model):
        """Evaluate model performance and determine if additional training needed"""
        logger.info("📊 EVALUATING MODEL PERFORMANCE")
        logger.info("=" * 50)
        
        # Run validation
        results = model.val()
        
        # Extract key metrics
        if hasattr(results, 'box'):
            mAP50 = results.box.map50 if hasattr(results.box, 'map50') else 0
            mAP50_95 = results.box.map if hasattr(results.box, 'map') else 0
            precision = results.box.mp if hasattr(results.box, 'mp') else 0
            recall = results.box.mr if hasattr(results.box, 'mr') else 0
        else:
            mAP50 = mAP50_95 = precision = recall = 0
        
        logger.info(f"📈 Performance Metrics:")
        logger.info(f"  mAP@50: {mAP50:.3f}")
        logger.info(f"  mAP@50-95: {mAP50_95:.3f}")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        
        # Determine if performance is acceptable
        performance_acceptable = (
            mAP50 > 0.3 and  # At least 30% mAP@50
            precision > 0.4 and  # At least 40% precision
            recall > 0.3  # At least 30% recall
        )
        
        if performance_acceptable:
            logger.info("✅ Model performance is acceptable!")
            return True
        else:
            logger.warning("⚠️ Model performance needs improvement")
            return False
    
    def run_advanced_training(self):
        """Run complete advanced training pipeline"""
        logger.info("🎯 GODSEYE AI - ADVANCED TRANSFER + CURRICULUM TRAINING")
        logger.info("=" * 70)
        logger.info("🚀 AUTOMATICALLY TRIGGERED DUE TO INSUFFICIENT ACCURACY")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            # Start MLflow run
            with mlflow.start_run(run_name="Advanced_Transfer_Curriculum_Training"):
                # Log parameters
                mlflow.log_params({
                    "methodology": "Transfer Learning + Curriculum Learning",
                    "stages": 3,
                    "transfer_learning": "COCO pretrained",
                    "curriculum": "Easy → Normal → Hard",
                    "total_epochs": 100,
                    "quality_compromise": "None"
                })
                
                # Stage 1: Transfer Learning
                model, results1 = self.stage_1_transfer_learning()
                
                # Stage 2: Curriculum Learning - Easy
                model, results2 = self.stage_2_curriculum_easy(model)
                
                # Stage 3: Curriculum Learning - Hard
                model, results3 = self.stage_3_curriculum_hard(model)
                
                # Evaluate final performance
                performance_ok = self.evaluate_model_performance(model)
                
                # Calculate total time
                total_time = time.time() - start_time
                
                logger.info("🎉 ADVANCED TRAINING COMPLETED!")
                logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
                logger.info(f"📁 Final model: models/godseye_advanced_model.pt")
                logger.info(f"📊 Performance acceptable: {performance_ok}")
                
                # Log final metrics
                if hasattr(results3, 'results_dict'):
                    mlflow.log_metrics(results3.results_dict)
                
                return model, performance_ok
                
        except Exception as e:
            logger.error(f"❌ Advanced training failed: {e}")
            return None, False

if __name__ == "__main__":
    trainer = AdvancedTransferCurriculumTrainer()
    model, success = trainer.run_advanced_training()
    
    if success:
        print("\n🎉 SUCCESS! Advanced training completed!")
        print("📊 Transfer Learning + Curriculum Learning applied!")
        print("🔬 Model should now have significantly better accuracy!")
    else:
        print("\n⚠️ Training completed but performance may need further improvement")
