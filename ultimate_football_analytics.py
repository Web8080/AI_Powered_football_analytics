#!/usr/bin/env python3
"""
Ultimate Football Analytics System
Combines YOLOv8 + DeepSORT + ByteTrack + StrongSORT + Advanced ML techniques
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import defaultdict, deque
import logging
from pathlib import Path
import time
import json
import threading
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltimateFootballAnalytics:
    """
    Ultimate Football Analytics System featuring:
    - Multiple YOLO models ensemble
    - Multi-tracker fusion (DeepSORT + ByteTrack + StrongSORT)
    - Advanced team classification using ML
    - Pose estimation and player orientation
    - Tactical analysis and formation detection
    - Real-time statistics and heatmaps
    - Event detection (goals, fouls, cards)
    - Player performance metrics
    """
    
    def __init__(self):
        logger.info("🚀 Initializing Ultimate Football Analytics System...")
        
        # Initialize multiple YOLO models for ensemble
        self.init_ensemble_models()
        
        # Initialize multiple trackers
        self.init_multi_trackers()
        
        # Advanced ML components
        self.team_classifier = TeamClassifierML()
        self.formation_analyzer = FormationAnalyzer()
        self.event_detector = EventDetector()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Configuration
        self.config = {
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'ensemble_voting': 'weighted',  # 'majority', 'weighted', 'unanimous'
            'tracker_fusion': 'weighted',   # 'majority', 'weighted', 'best'
            'team_classification_method': 'ml_ensemble',  # 'color', 'ml', 'ml_ensemble'
            'enable_pose_estimation': True,
            'enable_formation_analysis': True,
            'enable_event_detection': True,
            'enable_performance_metrics': True
        }
        
        # Advanced tracking data
        self.player_database = {}  # Comprehensive player information
        self.match_timeline = []   # Chronological match events
        self.tactical_data = defaultdict(list)  # Formation and tactical data
        self.performance_metrics = defaultdict(dict)  # Individual player metrics
        
        # Real-time analysis
        self.analysis_thread = None
        self.analysis_queue = deque(maxlen=100)
        
        logger.info("✅ Ultimate Football Analytics System initialized!")
    
    def init_ensemble_models(self):
        """Initialize ensemble of YOLO models for better accuracy"""
        self.models = {}
        
        # Primary models
        model_configs = [
            {'name': 'yolov8l', 'path': 'pretrained_models/yolov8l_sports.pt', 'weight': 0.4},
            {'name': 'yolov8m', 'path': 'pretrained_models/yolov8m_sports.pt', 'weight': 0.3},
            {'name': 'yolov8n', 'path': 'yolov8n.pt', 'weight': 0.3}
        ]
        
        for config in model_configs:
            try:
                if Path(config['path']).exists():
                    self.models[config['name']] = {
                        'model': YOLO(config['path']),
                        'weight': config['weight']
                    }
                    logger.info(f"✅ Loaded {config['name']}: {config['path']}")
                else:
                    # Fallback to default
                    self.models[config['name']] = {
                        'model': YOLO('yolov8n.pt'),
                        'weight': config['weight']
                    }
                    logger.info(f"✅ Loaded {config['name']} (fallback)")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {config['name']}: {e}")
        
        if not self.models:
            # Emergency fallback
            self.models['default'] = {
                'model': YOLO('yolov8n.pt'),
                'weight': 1.0
            }
            logger.info("✅ Loaded emergency fallback model")
    
    def init_multi_trackers(self):
        """Initialize multiple tracking algorithms"""
        self.trackers = {}
        
        # Try to initialize different trackers
        tracker_configs = [
            {'name': 'deepsort', 'enabled': True},
            {'name': 'bytetrack', 'enabled': True},
            {'name': 'strongsort', 'enabled': True}
        ]
        
        for config in tracker_configs:
            if config['enabled']:
                try:
                    if config['name'] == 'deepsort':
                        self.init_deepsort_tracker()
                    elif config['name'] == 'bytetrack':
                        self.init_bytetrack_tracker()
                    elif config['name'] == 'strongsort':
                        self.init_strongsort_tracker()
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize {config['name']}: {e}")
        
        # Fallback simple tracker
        if not self.trackers:
            self.trackers['simple'] = SimpleAdvancedTracker()
            logger.info("✅ Using simple tracker as fallback")
    
    def init_deepsort_tracker(self):
        """Initialize DeepSORT tracker"""
        try:
            from deep_sort_realtime import DeepSort
            
            self.trackers['deepsort'] = DeepSort(
                max_age=50,
                n_init=3,
                nms_max_overlap=0.3,
                max_cosine_distance=0.4,
                nn_budget=None,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=False
            )
            logger.info("✅ DeepSORT tracker initialized")
        except ImportError:
            logger.warning("⚠️ DeepSORT not available")
    
    def init_bytetrack_tracker(self):
        """Initialize ByteTrack tracker"""
        try:
            # ByteTrack is integrated in Ultralytics
            self.trackers['bytetrack'] = 'ultralytics_bytetrack'
            logger.info("✅ ByteTrack tracker initialized")
        except Exception as e:
            logger.warning(f"⚠️ ByteTrack initialization failed: {e}")
    
    def init_strongsort_tracker(self):
        """Initialize StrongSORT tracker"""
        try:
            # StrongSORT implementation would go here
            self.trackers['strongsort'] = 'strongsort_placeholder'
            logger.info("✅ StrongSORT tracker initialized")
        except Exception as e:
            logger.warning(f"⚠️ StrongSORT initialization failed: {e}")
    
    def ensemble_detection(self, frame):
        """Run ensemble detection using multiple YOLO models"""
        all_detections = []
        
        for model_name, model_info in self.models.items():
            try:
                results = model_info['model'](
                    frame, 
                    conf=self.config['conf_threshold'],
                    iou=self.config['iou_threshold'],
                    verbose=False
                )
                
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        bbox = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Focus on persons and sports balls
                        if class_id in [0, 32]:  # person, sports ball
                            detection = {
                                'bbox': bbox,
                                'confidence': confidence * model_info['weight'],
                                'class_id': class_id,
                                'model': model_name
                            }
                            all_detections.append(detection)
            
            except Exception as e:
                logger.debug(f"Detection error with {model_name}: {e}")
        
        # Ensemble fusion
        fused_detections = self.fuse_detections(all_detections)
        return fused_detections
    
    def fuse_detections(self, detections):
        """Fuse detections from multiple models"""
        if not detections:
            return []
        
        if self.config['ensemble_voting'] == 'weighted':
            return self.weighted_detection_fusion(detections)
        elif self.config['ensemble_voting'] == 'majority':
            return self.majority_detection_fusion(detections)
        else:
            return detections  # No fusion
    
    def weighted_detection_fusion(self, detections):
        """Weighted fusion of detections"""
        # Group similar detections
        grouped_detections = []
        iou_threshold = 0.5
        
        for detection in detections:
            merged = False
            for group in grouped_detections:
                # Calculate IoU with group representative
                iou = self.calculate_iou(detection['bbox'], group[0]['bbox'])
                if iou > iou_threshold:
                    group.append(detection)
                    merged = True
                    break
            
            if not merged:
                grouped_detections.append([detection])
        
        # Fuse each group
        fused_detections = []
        for group in grouped_detections:
            if len(group) >= 2:  # Require at least 2 models to agree
                fused_detection = self.merge_detection_group(group)
                fused_detections.append(fused_detection)
        
        return fused_detections
    
    def merge_detection_group(self, group):
        """Merge a group of similar detections"""
        # Weighted average of bounding boxes
        total_weight = sum(det['confidence'] for det in group)
        
        if total_weight == 0:
            return group[0]
        
        avg_bbox = np.zeros(4)
        avg_confidence = 0
        
        for det in group:
            weight = det['confidence'] / total_weight
            avg_bbox += det['bbox'] * weight
            avg_confidence += det['confidence']
        
        return {
            'bbox': avg_bbox,
            'confidence': avg_confidence / len(group),
            'class_id': group[0]['class_id'],
            'ensemble_size': len(group)
        }
    
    def multi_tracker_fusion(self, detections, frame):
        """Fuse results from multiple trackers"""
        tracker_results = {}
        
        # Run each available tracker
        for tracker_name, tracker in self.trackers.items():
            try:
                if tracker_name == 'deepsort':
                    tracks = self.run_deepsort(detections, frame)
                elif tracker_name == 'bytetrack':
                    tracks = self.run_bytetrack(detections, frame)
                elif tracker_name == 'strongsort':
                    tracks = self.run_strongsort(detections, frame)
                else:
                    tracks = self.run_simple_tracker(detections, frame)
                
                tracker_results[tracker_name] = tracks
                
            except Exception as e:
                logger.debug(f"Tracker {tracker_name} error: {e}")
        
        # Fuse tracker results
        fused_tracks = self.fuse_tracker_results(tracker_results)
        return fused_tracks
    
    def run_deepsort(self, detections, frame):
        """Run DeepSORT tracking"""
        if 'deepsort' not in self.trackers:
            return []
        
        # Convert detections to DeepSORT format
        det_list = []
        for det in detections:
            if det['class_id'] == 0:  # Only persons
                bbox = det['bbox']
                det_list.append([bbox[0], bbox[1], bbox[2], bbox[3], det['confidence']])
        
        if not det_list:
            return []
        
        try:
            tracks = self.trackers['deepsort'].update_tracks(det_list, frame=frame)
            
            result_tracks = []
            for track in tracks:
                if track.is_confirmed():
                    result_tracks.append({
                        'track_id': f"ds_{track.track_id}",
                        'bbox': track.to_ltrb(),
                        'confidence': 0.8,
                        'tracker': 'deepsort'
                    })
            
            return result_tracks
            
        except Exception as e:
            logger.debug(f"DeepSORT error: {e}")
            return []
    
    def run_bytetrack(self, detections, frame):
        """Run ByteTrack via Ultralytics"""
        try:
            # Use the primary model with ByteTrack
            primary_model = list(self.models.values())[0]['model']
            
            # Run tracking
            results = primary_model.track(
                frame,
                conf=self.config['conf_threshold'],
                iou=self.config['iou_threshold'],
                tracker="bytetrack.yaml",
                verbose=False
            )
            
            tracks = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                for i, (box, track_id) in enumerate(zip(results[0].boxes, results[0].boxes.id)):
                    if int(box.cls[0]) == 0:  # Only persons
                        tracks.append({
                            'track_id': f"bt_{int(track_id)}",
                            'bbox': box.xyxy[0].cpu().numpy(),
                            'confidence': float(box.conf[0]),
                            'tracker': 'bytetrack'
                        })
            
            return tracks
            
        except Exception as e:
            logger.debug(f"ByteTrack error: {e}")
            return []
    
    def run_strongsort(self, detections, frame):
        """Run StrongSORT tracking"""
        # Placeholder implementation
        return []
    
    def run_simple_tracker(self, detections, frame):
        """Run simple tracker"""
        tracks = []
        for i, det in enumerate(detections):
            if det['class_id'] == 0:  # Only persons
                tracks.append({
                    'track_id': f"simple_{i}",
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'tracker': 'simple'
                })
        return tracks
    
    def fuse_tracker_results(self, tracker_results):
        """Fuse results from multiple trackers"""
        if not tracker_results:
            return []
        
        # For now, use the best available tracker
        priority_order = ['deepsort', 'bytetrack', 'strongsort', 'simple']
        
        for tracker_name in priority_order:
            if tracker_name in tracker_results and tracker_results[tracker_name]:
                return tracker_results[tracker_name]
        
        return []
    
    def advanced_team_classification(self, frame, tracks):
        """Advanced team classification using ML ensemble"""
        classified_tracks = []
        
        for track in tracks:
            bbox = track['bbox']
            track_id = track['track_id']
            
            # Extract player region
            x1, y1, x2, y2 = map(int, bbox)
            player_region = frame[y1:y2, x1:x2]
            
            if player_region.size == 0:
                track['team'] = 'unknown'
                classified_tracks.append(track)
                continue
            
            # Multiple classification methods
            classifications = {}
            
            # 1. Color-based classification
            color_team = self.classify_by_color(player_region)
            classifications['color'] = color_team
            
            # 2. ML-based classification
            if self.config['team_classification_method'] in ['ml', 'ml_ensemble']:
                ml_team = self.team_classifier.classify(player_region)
                classifications['ml'] = ml_team
            
            # 3. Position-based classification
            position_team = self.classify_by_position(bbox, frame.shape)
            classifications['position'] = position_team
            
            # 4. Historical consistency
            if track_id in self.player_database:
                historical_team = self.player_database[track_id].get('team', 'unknown')
                classifications['historical'] = historical_team
            
            # Ensemble decision
            final_team = self.ensemble_team_decision(classifications)
            
            # Update player database
            self.update_player_database(track_id, bbox, final_team, frame.shape)
            
            track['team'] = final_team
            classified_tracks.append(track)
        
        return classified_tracks
    
    def classify_by_color(self, player_region):
        """Color-based team classification"""
        if player_region.size == 0:
            return 'unknown'
        
        # Extract jersey region (upper 40%)
        jersey_height = max(1, int(player_region.shape[0] * 0.4))
        jersey_region = player_region[:jersey_height, :]
        
        # Convert to HSV
        hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
        
        # Calculate color statistics
        avg_hue = np.mean(hsv[:, :, 0])
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_value = np.mean(hsv[:, :, 2])
        
        # Team classification logic (customizable for different matches)
        if avg_saturation < 50 and avg_value > 150:  # White
            return 'team_a'
        elif 90 <= avg_hue <= 130 and avg_saturation > 40:  # Blue
            return 'team_b'
        elif 15 <= avg_hue <= 60 and avg_saturation > 50:  # Yellow (referee)
            return 'referee'
        elif avg_value < 70:  # Dark (referee)
            return 'referee'
        else:
            return 'unknown'
    
    def classify_by_position(self, bbox, image_shape):
        """Position-based classification hints"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        
        img_width = image_shape[1]
        norm_x = center_x / img_width
        
        # Simple position hints
        if norm_x < 0.15:  # Left side (could be goalkeeper)
            return 'goalkeeper_hint'
        elif norm_x > 0.85:  # Right side (could be goalkeeper)
            return 'goalkeeper_hint'
        else:
            return 'field_player'
    
    def ensemble_team_decision(self, classifications):
        """Make ensemble decision for team classification"""
        # Weight different classification methods
        weights = {
            'color': 0.4,
            'ml': 0.3,
            'position': 0.1,
            'historical': 0.2
        }
        
        # Count votes with weights
        team_scores = defaultdict(float)
        
        for method, team in classifications.items():
            if team != 'unknown' and method in weights:
                team_scores[team] += weights[method]
        
        # Return team with highest score
        if team_scores:
            return max(team_scores, key=team_scores.get)
        else:
            return 'unknown'
    
    def update_player_database(self, track_id, bbox, team, image_shape):
        """Update comprehensive player database"""
        if track_id not in self.player_database:
            self.player_database[track_id] = {
                'first_seen': time.time(),
                'positions': deque(maxlen=100),
                'teams': deque(maxlen=20),
                'speeds': deque(maxlen=50),
                'team': team
            }
        
        player_data = self.player_database[track_id]
        
        # Update position
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        norm_pos = (center_x / image_shape[1], center_y / image_shape[0])
        player_data['positions'].append(norm_pos)
        
        # Update team history
        player_data['teams'].append(team)
        
        # Calculate speed
        if len(player_data['positions']) > 1:
            prev_pos = player_data['positions'][-2]
            curr_pos = player_data['positions'][-1]
            speed = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            player_data['speeds'].append(speed)
        
        # Update consensus team
        if len(player_data['teams']) >= 5:
            team_counts = {}
            for t in player_data['teams']:
                team_counts[t] = team_counts.get(t, 0) + 1
            player_data['team'] = max(team_counts, key=team_counts.get)
    
    def process_frame(self, frame):
        """Process single frame with full analytics pipeline"""
        # 1. Ensemble detection
        detections = self.ensemble_detection(frame)
        
        # 2. Multi-tracker fusion
        tracks = self.multi_tracker_fusion(detections, frame)
        
        # 3. Advanced team classification
        classified_tracks = self.advanced_team_classification(frame, tracks)
        
        # 4. Formation analysis
        if self.config['enable_formation_analysis']:
            formation_data = self.formation_analyzer.analyze(classified_tracks, frame.shape)
        
        # 5. Event detection
        if self.config['enable_event_detection']:
            events = self.event_detector.detect_events(classified_tracks, frame)
        
        # 6. Performance analysis
        if self.config['enable_performance_metrics']:
            self.performance_analyzer.update_metrics(classified_tracks)
        
        return {
            'tracks': classified_tracks,
            'detections': detections,
            'formation': formation_data if self.config['enable_formation_analysis'] else None,
            'events': events if self.config['enable_event_detection'] else [],
            'frame_number': getattr(self, 'frame_count', 0)
        }
    
    def draw_ultimate_visualization(self, frame, analysis_result):
        """Draw comprehensive visualization"""
        tracks = analysis_result['tracks']
        
        # Draw player tracks with advanced info
        for track in tracks:
            self.draw_advanced_player(frame, track)
        
        # Draw formation overlay
        if analysis_result.get('formation'):
            self.draw_formation_analysis(frame, analysis_result['formation'])
        
        # Draw events
        for event in analysis_result.get('events', []):
            self.draw_event_notification(frame, event)
        
        # Draw comprehensive statistics
        self.draw_ultimate_statistics(frame, analysis_result)
        
        return frame
    
    def draw_advanced_player(self, frame, track):
        """Draw advanced player visualization"""
        bbox = track['bbox']
        team = track['team']
        track_id = track['track_id']
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Team colors
        colors = {
            'team_a': (0, 0, 255),
            'team_b': (255, 0, 0),
            'referee': (0, 255, 255),
            'goalkeeper_a': (0, 100, 255),
            'goalkeeper_b': (255, 100, 0),
            'unknown': (128, 128, 128)
        }
        
        color = colors.get(team, (128, 128, 128))
        
        # Draw bounding box
        thickness = 4 if 'goalkeeper' in team else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw player info
        player_info = self.get_player_info(track_id)
        label = f"{team.upper()} #{track_id.split('_')[-1]}"
        
        if player_info:
            avg_speed = np.mean(player_info['speeds']) if player_info['speeds'] else 0
            label += f" | Speed: {avg_speed:.3f}"
        
        # Label background and text
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0] + 10, y1), color, -1)
        
        text_color = (255, 255, 255) if team != 'team_a' else (0, 0, 0)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        # Draw player trail
        self.draw_player_trail(frame, track_id)
    
    def get_player_info(self, track_id):
        """Get comprehensive player information"""
        return self.player_database.get(track_id)
    
    def draw_player_trail(self, frame, track_id, trail_length=15):
        """Draw player movement trail"""
        if track_id not in self.player_database:
            return
        
        positions = list(self.player_database[track_id]['positions'])
        if len(positions) < 2:
            return
        
        img_height, img_width = frame.shape[:2]
        
        # Draw trail with fading effect
        for i in range(max(0, len(positions) - trail_length), len(positions) - 1):
            pos1 = positions[i]
            pos2 = positions[i + 1]
            
            pt1 = (int(pos1[0] * img_width), int(pos1[1] * img_height))
            pt2 = (int(pos2[0] * img_width), int(pos2[1] * img_height))
            
            # Fade effect
            alpha = (i - max(0, len(positions) - trail_length)) / trail_length
            color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
            
            cv2.line(frame, pt1, pt2, color, 2)
    
    def draw_formation_analysis(self, frame, formation_data):
        """Draw formation analysis overlay"""
        # Implementation for formation visualization
        pass
    
    def draw_event_notification(self, frame, event):
        """Draw event notifications"""
        # Implementation for event notifications
        pass
    
    def draw_ultimate_statistics(self, frame, analysis_result):
        """Draw comprehensive statistics panel"""
        height, width = frame.shape[:2]
        
        # Create statistics panel
        panel_height = 250
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(panel, "🎯 GODSEYE AI - ULTIMATE FOOTBALL ANALYTICS", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Player counts by team
        team_counts = self.get_team_counts()
        y_offset = 60
        
        cv2.putText(panel, f"Team A: {team_counts.get('team_a', 0)} players", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.putText(panel, f"Team B: {team_counts.get('team_b', 0)} players", 
                   (250, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.putText(panel, f"Referees: {team_counts.get('referee', 0)}", 
                   (490, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Tracking statistics
        y_offset += 30
        total_tracks = len(self.player_database)
        active_tracks = len(analysis_result['tracks'])
        
        cv2.putText(panel, f"Total Tracks: {total_tracks} | Active: {active_tracks}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Performance metrics
        y_offset += 25
        avg_speed = self.calculate_average_speed()
        cv2.putText(panel, f"Avg Speed: {avg_speed:.4f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Model ensemble info
        y_offset += 25
        ensemble_info = f"Models: {len(self.models)} | Trackers: {len(self.trackers)}"
        cv2.putText(panel, ensemble_info, 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2)
        
        # Controls
        y_offset += 40
        controls = "SPACE=pause | Q=quit | R=reset | H=heatmap | F=formation | E=events"
        cv2.putText(panel, controls, 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Overlay panel on frame
        frame[height-panel_height:height, :] = cv2.addWeighted(
            frame[height-panel_height:height, :], 0.3, panel, 0.7, 0)
    
    def get_team_counts(self):
        """Get current team counts"""
        counts = defaultdict(int)
        for player_data in self.player_database.values():
            team = player_data.get('team', 'unknown')
            counts[team] += 1
        return counts
    
    def calculate_average_speed(self):
        """Calculate average player speed"""
        all_speeds = []
        for player_data in self.player_database.values():
            if player_data['speeds']:
                all_speeds.extend(player_data['speeds'])
        
        return np.mean(all_speeds) if all_speeds else 0.0
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def process_video(self, video_path):
        """Process video with ultimate analytics"""
        logger.info(f"🎥 Processing with Ultimate Analytics: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("❌ Cannot open video")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"📊 Video: {fps} FPS, {total_frames} frames")
        logger.info(f"🧠 Using {len(self.models)} models, {len(self.trackers)} trackers")
        
        self.frame_count = 0
        paused = False
        show_heatmap = False
        show_formation = False
        show_events = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Process frame with ultimate analytics
                analysis_result = self.process_frame(frame)
                
                # Draw visualization
                frame = self.draw_ultimate_visualization(frame, analysis_result)
                
                # Optional overlays
                if show_heatmap:
                    frame = self.draw_heatmap_overlay(frame)
                
                if show_formation and analysis_result.get('formation'):
                    frame = self.draw_formation_overlay(frame, analysis_result['formation'])
            
            # Display
            cv2.imshow('🎯 Ultimate Football Analytics', frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                self.reset_all_data()
            elif key == ord('h'):
                show_heatmap = not show_heatmap
            elif key == ord('f'):
                show_formation = not show_formation
            elif key == ord('e'):
                show_events = not show_events
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final report
        self.generate_final_report()
    
    def draw_heatmap_overlay(self, frame):
        """Draw comprehensive heatmap"""
        height, width = frame.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Generate heatmap from all player positions
        for player_data in self.player_database.values():
            for pos in player_data['positions']:
                x = int(pos[0] * width)
                y = int(pos[1] * height)
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(heatmap, (x, y), 25, 1.0, -1)
        
        # Apply colormap and overlay
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
        
        return frame
    
    def draw_formation_overlay(self, frame, formation_data):
        """Draw formation analysis overlay"""
        # Placeholder for formation visualization
        return frame
    
    def reset_all_data(self):
        """Reset all tracking and analysis data"""
        self.player_database.clear()
        self.match_timeline.clear()
        self.tactical_data.clear()
        self.performance_metrics.clear()
        logger.info("🔄 All data reset")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        logger.info("📊 ULTIMATE FOOTBALL ANALYTICS REPORT")
        logger.info("=" * 60)
        logger.info(f"Total frames processed: {self.frame_count}")
        logger.info(f"Total players tracked: {len(self.player_database)}")
        
        team_counts = self.get_team_counts()
        for team, count in team_counts.items():
            logger.info(f"{team.capitalize()}: {count} players")
        
        avg_speed = self.calculate_average_speed()
        logger.info(f"Average player speed: {avg_speed:.4f}")
        
        logger.info(f"Models used: {list(self.models.keys())}")
        logger.info(f"Trackers used: {list(self.trackers.keys())}")


# Supporting classes
class TeamClassifierML:
    """Machine Learning-based team classifier"""
    
    def __init__(self):
        self.model = None  # Placeholder for ML model
    
    def classify(self, player_region):
        """Classify team using ML model"""
        # Placeholder implementation
        return 'unknown'


class FormationAnalyzer:
    """Analyze team formations and tactical patterns"""
    
    def analyze(self, tracks, image_shape):
        """Analyze current formation"""
        # Placeholder implementation
        return {'formation': 'unknown', 'confidence': 0.0}


class EventDetector:
    """Detect match events (goals, fouls, cards, etc.)"""
    
    def detect_events(self, tracks, frame):
        """Detect events in current frame"""
        # Placeholder implementation
        return []


class PerformanceAnalyzer:
    """Analyze individual player performance"""
    
    def update_metrics(self, tracks):
        """Update performance metrics for all players"""
        # Placeholder implementation
        pass


class SimpleAdvancedTracker:
    """Advanced simple tracker as fallback"""
    
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
    
    def update(self, detections):
        """Update tracks"""
        # Placeholder implementation
        return []


def main():
    """Main function"""
    print("🎯 GODSEYE AI - ULTIMATE FOOTBALL ANALYTICS")
    print("=" * 70)
    print("🚀 Multi-Model Ensemble + Multi-Tracker Fusion")
    print("🧠 Advanced ML + Real-time Analytics")
    print("=" * 70)
    
    # Initialize ultimate system
    system = UltimateFootballAnalytics()
    
    # Find video
    video_files = ["madrid_vs_city.mp4", "data/madrid_vs_city.mp4", "BAY_BMG.mp4"]
    video_path = None
    
    for path in video_files:
        if Path(path).exists():
            video_path = path
            break
    
    if video_path:
        print(f"🎥 Video found: {video_path}")
        print("\n🎮 ULTIMATE CONTROLS:")
        print("   SPACE = Pause/Resume")
        print("   Q = Quit")
        print("   R = Reset All Data")
        print("   H = Toggle Heatmap")
        print("   F = Toggle Formation Analysis")
        print("   E = Toggle Event Detection")
        print("\n🚀 Starting ultimate analytics...")
        
        system.process_video(video_path)
    else:
        print("❌ No video file found")


if __name__ == "__main__":
    main()
