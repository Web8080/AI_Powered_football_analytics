#!/usr/bin/env python3
"""
Specialized training script for Godseye AI detection models.
Uses the new 8-class professional football classification system.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml
from omegaconf import OmegaConf

# MLflow for experiment tracking
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# Computer Vision libraries
import cv2
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Custom modules
from data.dataset import FootballDataset
from models.detection import YOLODetector, create_detection_model
from utils.metrics import calculate_metrics, plot_metrics
from utils.visualization import visualize_detections
from utils.augmentation import get_comprehensive_augmentation_pipeline
from utils.class_mapping import FootballClassMapper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detection_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DetectionTrainer:
    """Specialized trainer for detection models with professional football classes."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Setup MLflow
        self.setup_mlflow()
        
        # Initialize class mapper
        self.class_mapper = FootballClassMapper()
        
        # Training state
        self.current_epoch = 0
        self.best_metrics = {}
        self.training_history = {
            'loss': [],
            'mAP': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    def load_config(self, config_path: str) -> Dict:
        """Load training configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_mlflow(self):
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.config.get('mlflow_uri', 'http://localhost:5000'))
        mlflow.set_experiment(self.config.get('experiment_name', 'godseye_detection'))
        
        # Start run
        self.run = mlflow.start_run(run_name=self.config.get('run_name', 'detection_training'))
        logger.info(f"MLflow run started: {self.run.info.run_id}")
        
        # Log class information
        mlflow.log_params({
            'num_classes': len(self.class_mapper.CLASSES),
            'classes': ','.join(self.class_mapper.CLASSES),
            'model_type': 'yolo_detection'
        })
    
    def prepare_datasets(self):
        """Prepare datasets for training."""
        logger.info("Preparing datasets...")
        
        # Get augmentation pipeline
        augmentation_pipeline = get_comprehensive_augmentation_pipeline(
            task='detection',
            weather_conditions=['rain', 'snow', 'fog', 'shadows'],
            lighting_conditions=['bright', 'dark', 'spotlight', 'dawn', 'evening'],
            camera_conditions=['motion_blur', 'camera_shake'],
            split='train'
        )
        
        # Training dataset
        self.train_dataset = FootballDataset(
            data_dir=self.config['data']['detection_path'],
            split='train',
            transform=augmentation_pipeline,
            max_samples=self.config['training'].get('max_samples')
        )
        
        # Validation dataset
        val_augmentation = get_comprehensive_augmentation_pipeline(
            task='detection',
            split='val'
        )
        
        self.val_dataset = FootballDataset(
            data_dir=self.config['data']['detection_path'],
            split='val',
            transform=val_augmentation,
            max_samples=self.config['training'].get('max_val_samples')
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
    
    def collate_fn(self, batch):
        """Custom collate function for detection data."""
        images, targets = zip(*batch)
        return list(images), list(targets)
    
    def initialize_model(self):
        """Initialize YOLO detection model."""
        logger.info("Initializing YOLO model...")
        
        # Create YOLO model with custom classes
        self.model = YOLO(f"yolov8{self.config['models']['detection']['yolo']['model_size']}.pt")
        
        # Update model classes
        self.model.names = {i: name for i, name in enumerate(self.class_mapper.CLASSES)}
        
        logger.info(f"Model initialized with {len(self.class_mapper.CLASSES)} classes")
        logger.info(f"Classes: {self.class_mapper.CLASSES}")
    
    def train_model(self):
        """Train the detection model."""
        logger.info("Starting model training...")
        
        # Create data configuration for YOLO
        data_config = self.create_data_config()
        
        # Train the model
        results = self.model.train(
            data=data_config,
            epochs=self.config['training']['epochs'],
            batch=self.config['training']['batch_size'],
            lr0=self.config['training']['learning_rate'],
            device=self.device,
            imgsz=self.config['models']['detection']['input_size'][0],
            save=True,
            save_period=10,
            val=True,
            plots=True,
            verbose=True,
            project='runs/detect',
            name='godseye_training'
        )
        
        # Log training results
        self.log_training_results(results)
        
        logger.info("Model training completed!")
        return results
    
    def create_data_config(self) -> str:
        """Create YOLO data configuration file."""
        config_path = "data_config.yaml"
        
        config = {
            'path': str(Path(self.config['data']['detection_path']).absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.class_mapper.CLASSES),
            'names': self.class_mapper.CLASSES
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_path
    
    def log_training_results(self, results):
        """Log training results to MLflow."""
        # Log metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            mlflow.log_metrics(metrics)
        
        # Log model
        model_path = results.save_dir / 'weights' / 'best.pt'
        if model_path.exists():
            mlflow.log_artifact(str(model_path))
        
        # Log training plots
        plots_dir = results.save_dir / 'results.png'
        if plots_dir.exists():
            mlflow.log_artifact(str(plots_dir))
        
        # Log confusion matrix
        confusion_matrix_path = results.save_dir / 'confusion_matrix.png'
        if confusion_matrix_path.exists():
            mlflow.log_artifact(str(confusion_matrix_path))
    
    def evaluate_model(self):
        """Evaluate the trained model."""
        logger.info("Evaluating model...")
        
        # Run validation
        results = self.model.val(
            data=self.create_data_config(),
            device=self.device,
            imgsz=self.config['models']['detection']['input_size'][0]
        )
        
        # Log evaluation metrics
        if hasattr(results, 'box'):
            metrics = {
                'val_mAP50': results.box.map50,
                'val_mAP50-95': results.box.map,
                'val_precision': results.box.mp,
                'val_recall': results.box.mr
            }
            mlflow.log_metrics(metrics)
        
        logger.info("Model evaluation completed!")
        return results
    
    def test_inference(self, test_image_path: str):
        """Test inference on a sample image."""
        logger.info(f"Testing inference on {test_image_path}")
        
        if not os.path.exists(test_image_path):
            logger.warning(f"Test image not found: {test_image_path}")
            return
        
        # Run inference
        results = self.model(test_image_path)
        
        # Process results
        for result in results:
            # Get detections
            boxes = result.boxes
            if boxes is not None:
                detections = []
                for i in range(len(boxes)):
                    detection = {
                        'class_id': int(boxes.cls[i]),
                        'class_name': self.class_mapper.get_class_name(int(boxes.cls[i])),
                        'confidence': float(boxes.conf[i]),
                        'bbox': boxes.xyxy[i].cpu().numpy().tolist()
                    }
                    detections.append(detection)
                
                # Log detection statistics
                stats = self.class_mapper.get_team_statistics(detections)
                mlflow.log_metrics({
                    'test_total_detections': len(detections),
                    'test_team_a_players': stats['team_a']['players'],
                    'test_team_a_goalkeepers': stats['team_a']['goalkeepers'],
                    'test_team_b_players': stats['team_b']['players'],
                    'test_team_b_goalkeepers': stats['team_b']['goalkeepers'],
                    'test_referees': stats['referees'],
                    'test_balls': stats['balls']
                })
                
                # Save visualization
                vis_image = visualize_detections(
                    cv2.imread(test_image_path),
                    detections,
                    self.class_mapper
                )
                
                vis_path = "test_inference_result.jpg"
                cv2.imwrite(vis_path, vis_image)
                mlflow.log_artifact(vis_path)
                
                logger.info(f"Found {len(detections)} detections")
                logger.info(f"Team statistics: {stats}")
    
    def export_model(self):
        """Export model for deployment."""
        logger.info("Exporting model...")
        
        # Export to ONNX
        onnx_path = self.model.export(
            format='onnx',
            imgsz=self.config['models']['detection']['input_size'][0],
            optimize=True
        )
        
        mlflow.log_artifact(onnx_path)
        logger.info(f"Model exported to: {onnx_path}")
        
        # Export to TensorRT if available
        try:
            trt_path = self.model.export(
                format='engine',
                imgsz=self.config['models']['detection']['input_size'][0]
            )
            mlflow.log_artifact(trt_path)
            logger.info(f"TensorRT model exported to: {trt_path}")
        except Exception as e:
            logger.warning(f"TensorRT export failed: {e}")
    
    def generate_training_report(self):
        """Generate comprehensive training report."""
        logger.info("Generating training report...")
        
        report = {
            'config': self.config,
            'class_mapping': {
                'classes': self.class_mapper.CLASSES,
                'num_classes': len(self.class_mapper.CLASSES),
                'class_to_idx': self.class_mapper.CLASS_TO_IDX
            },
            'training_history': self.training_history,
            'best_metrics': self.best_metrics,
            'model_info': {
                'model_type': 'YOLOv8',
                'input_size': self.config['models']['detection']['input_size'],
                'num_classes': len(self.class_mapper.CLASSES),
                'device': str(self.device)
            }
        }
        
        # Save report
        report_path = "detection_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        mlflow.log_artifact(report_path)
        
        # Generate class distribution plot
        self.plot_class_distribution()
        
        logger.info("Training report generated!")
    
    def plot_class_distribution(self):
        """Plot class distribution in training data."""
        class_counts = {}
        
        for sample in self.train_dataset:
            for target in sample[1]['labels']:
                class_name = self.class_mapper.get_class_name(target.item())
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Create plot
        plt.figure(figsize=(12, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        plt.bar(classes, counts)
        plt.title('Class Distribution in Training Data')
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_path = "class_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(plot_path)
        plt.close()
    
    def run_training(self):
        """Run complete training pipeline."""
        logger.info("Starting Godseye AI detection training pipeline...")
        
        try:
            # Prepare datasets
            self.prepare_datasets()
            
            # Initialize model
            self.initialize_model()
            
            # Train model
            training_results = self.train_model()
            
            # Evaluate model
            evaluation_results = self.evaluate_model()
            
            # Test inference
            test_image_path = self.config.get('test_image_path')
            if test_image_path:
                self.test_inference(test_image_path)
            
            # Export model
            self.export_model()
            
            # Generate report
            self.generate_training_report()
            
            logger.info("Detection training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            mlflow.end_run()


def main():
    parser = argparse.ArgumentParser(description="Train Godseye AI detection models")
    parser.add_argument("--config", type=str, required=True, help="Path to training configuration")
    parser.add_argument("--test-image", type=str, help="Path to test image for inference")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = DetectionTrainer(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        trainer.config['training']['epochs'] = args.epochs
    if args.batch_size:
        trainer.config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        trainer.config['training']['learning_rate'] = args.learning_rate
    if args.test_image:
        trainer.config['test_image_path'] = args.test_image
    
    # Run training
    trainer.run_training()


if __name__ == "__main__":
    main()
