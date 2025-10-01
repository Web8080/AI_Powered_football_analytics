#!/usr/bin/env python3
"""
Advanced Feature Engineering for Football Analytics
==================================================

Senior Data Scientist Approach:
- Spatial feature extraction
- Temporal feature engineering
- Team formation analysis
- Player movement patterns
- Ball trajectory analysis
- Advanced statistical features

Author: Victor
Date: 2025
"""

import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class SpatialFeatureExtractor:
    """
    Extract spatial features from player positions and field geometry
    """
    
    def __init__(self, field_width: float = 105.0, field_height: float = 68.0):
        self.field_width = field_width
        self.field_height = field_height
        
        # Field zones (defensive, midfield, attacking)
        self.field_zones = {
            'defensive_third': (0, 0.33),
            'midfield': (0.33, 0.67),
            'attacking_third': (0.67, 1.0)
        }
        
        # Field sectors for detailed analysis
        self.field_sectors = self._create_field_sectors()
    
    def _create_field_sectors(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Create detailed field sectors"""
        sectors = {}
        
        # Divide field into 18 sectors (3x6 grid)
        for i in range(3):  # Vertical zones
            for j in range(6):  # Horizontal zones
                x_start = j / 6
                x_end = (j + 1) / 6
                y_start = i / 3
                y_end = (i + 1) / 3
                
                sector_name = f"sector_{i}_{j}"
                sectors[sector_name] = (x_start, y_start, x_end, y_end)
        
        return sectors
    
    def extract_position_features(self, bbox: List[float], image_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Extract comprehensive position features
        """
        x1, y1, x2, y2 = bbox
        img_h, img_w = image_shape[:2]
        
        # Normalize coordinates
        x_center = (x1 + x2) / 2 / img_w
        y_center = (y1 + y2) / 2 / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        
        # Basic position features
        features = {
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height,
            'area': width * height,
            'aspect_ratio': width / height if height > 0 else 0,
        }
        
        # Field position features
        features.update(self._get_field_position_features(x_center, y_center))
        
        # Distance features
        features.update(self._get_distance_features(x_center, y_center))
        
        # Field sector features
        features.update(self._get_sector_features(x_center, y_center))
        
        return features
    
    def _get_field_position_features(self, x: float, y: float) -> Dict[str, Any]:
        """Get field position classification features"""
        features = {}
        
        # Field zone classification
        for zone_name, (y_start, y_end) in self.field_zones.items():
            features[f'in_{zone_name}'] = 1 if y_start <= y <= y_end else 0
        
        # Side of field (left/right)
        features['on_left_side'] = 1 if x < 0.5 else 0
        features['on_right_side'] = 1 if x > 0.5 else 0
        
        # Field quadrant
        if x < 0.5 and y < 0.33:
            features['field_quadrant'] = 0  # Left defensive
        elif x >= 0.5 and y < 0.33:
            features['field_quadrant'] = 1  # Right defensive
        elif x < 0.5 and y >= 0.67:
            features['field_quadrant'] = 2  # Left attacking
        elif x >= 0.5 and y >= 0.67:
            features['field_quadrant'] = 3  # Right attacking
        else:
            features['field_quadrant'] = 4  # Midfield
        
        return features
    
    def _get_distance_features(self, x: float, y: float) -> Dict[str, float]:
        """Get distance-based features"""
        features = {}
        
        # Distance from field center
        features['distance_from_center'] = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        
        # Distance from goal lines
        features['distance_from_own_goal'] = y
        features['distance_from_opponent_goal'] = 1 - y
        
        # Distance from sidelines
        features['distance_from_left_sideline'] = x
        features['distance_from_right_sideline'] = 1 - x
        
        # Distance from corners
        corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
        min_corner_distance = min([np.sqrt((x - cx)**2 + (y - cy)**2) for cx, cy in corners])
        features['distance_from_nearest_corner'] = min_corner_distance
        
        return features
    
    def _get_sector_features(self, x: float, y: float) -> Dict[str, int]:
        """Get field sector features"""
        features = {}
        
        for sector_name, (x_start, y_start, x_end, y_end) in self.field_sectors.items():
            features[f'in_{sector_name}'] = 1 if (x_start <= x <= x_end and y_start <= y <= y_end) else 0
        
        return features

class TemporalFeatureExtractor:
    """
    Extract temporal features from player movement history
    """
    
    def __init__(self, history_length: int = 30):
        self.history_length = history_length
    
    def extract_movement_features(self, position_history: List[Dict]) -> Dict[str, float]:
        """
        Extract comprehensive movement features
        """
        if len(position_history) < 2:
            return self._get_default_movement_features()
        
        # Extract positions
        positions = [(p['x_center'], p['y_center']) for p in position_history[-self.history_length:]]
        timestamps = [p.get('timestamp', i) for i, p in enumerate(position_history[-self.history_length:])]
        
        features = {}
        
        # Speed features
        features.update(self._get_speed_features(positions, timestamps))
        
        # Acceleration features
        features.update(self._get_acceleration_features(positions, timestamps))
        
        # Direction features
        features.update(self._get_direction_features(positions))
        
        # Movement pattern features
        features.update(self._get_movement_pattern_features(positions))
        
        # Statistical features
        features.update(self._get_statistical_features(positions))
        
        return features
    
    def _get_speed_features(self, positions: List[Tuple[float, float]], 
                           timestamps: List[float]) -> Dict[str, float]:
        """Calculate speed-related features"""
        features = {}
        
        if len(positions) < 2:
            return {'avg_speed': 0, 'max_speed': 0, 'speed_variance': 0}
        
        speeds = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dt = timestamps[i] - timestamps[i-1] if len(timestamps) > i else 1
            
            if dt > 0:
                speed = np.sqrt(dx**2 + dy**2) / dt
                speeds.append(speed)
        
        if speeds:
            features.update({
                'avg_speed': np.mean(speeds),
                'max_speed': np.max(speeds),
                'min_speed': np.min(speeds),
                'speed_variance': np.var(speeds),
                'speed_std': np.std(speeds),
                'speed_range': np.max(speeds) - np.min(speeds),
            })
        else:
            features.update({
                'avg_speed': 0, 'max_speed': 0, 'min_speed': 0,
                'speed_variance': 0, 'speed_std': 0, 'speed_range': 0
            })
        
        return features
    
    def _get_acceleration_features(self, positions: List[Tuple[float, float]], 
                                  timestamps: List[float]) -> Dict[str, float]:
        """Calculate acceleration-related features"""
        features = {}
        
        if len(positions) < 3:
            return {'avg_acceleration': 0, 'max_acceleration': 0, 'acceleration_variance': 0}
        
        accelerations = []
        for i in range(2, len(positions)):
            # Calculate speed at current and previous time
            dx1 = positions[i-1][0] - positions[i-2][0]
            dy1 = positions[i-1][1] - positions[i-2][1]
            dt1 = timestamps[i-1] - timestamps[i-2] if len(timestamps) > i-1 else 1
            
            dx2 = positions[i][0] - positions[i-1][0]
            dy2 = positions[i][1] - positions[i-1][1]
            dt2 = timestamps[i] - timestamps[i-1] if len(timestamps) > i else 1
            
            if dt1 > 0 and dt2 > 0:
                speed1 = np.sqrt(dx1**2 + dy1**2) / dt1
                speed2 = np.sqrt(dx2**2 + dy2**2) / dt2
                acceleration = (speed2 - speed1) / dt2
                accelerations.append(acceleration)
        
        if accelerations:
            features.update({
                'avg_acceleration': np.mean(accelerations),
                'max_acceleration': np.max(accelerations),
                'min_acceleration': np.min(accelerations),
                'acceleration_variance': np.var(accelerations),
                'acceleration_std': np.std(accelerations),
            })
        else:
            features.update({
                'avg_acceleration': 0, 'max_acceleration': 0, 'min_acceleration': 0,
                'acceleration_variance': 0, 'acceleration_std': 0
            })
        
        return features
    
    def _get_direction_features(self, positions: List[Tuple[float, float]]) -> Dict[str, float]:
        """Calculate direction-related features"""
        features = {}
        
        if len(positions) < 2:
            return {'avg_direction': 0, 'direction_variance': 0, 'direction_consistency': 0}
        
        directions = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            direction = np.arctan2(dy, dx)
            directions.append(direction)
        
        if directions:
            # Convert to unit circle for proper averaging
            cos_dirs = [np.cos(d) for d in directions]
            sin_dirs = [np.sin(d) for d in directions]
            
            avg_cos = np.mean(cos_dirs)
            avg_sin = np.mean(sin_dirs)
            avg_direction = np.arctan2(avg_sin, avg_cos)
            
            # Direction consistency (how consistent the direction is)
            direction_consistency = np.sqrt(avg_cos**2 + avg_sin**2)
            
            features.update({
                'avg_direction': avg_direction,
                'direction_variance': np.var(directions),
                'direction_consistency': direction_consistency,
                'direction_std': np.std(directions),
            })
        else:
            features.update({
                'avg_direction': 0, 'direction_variance': 0, 'direction_consistency': 0,
                'direction_std': 0
            })
        
        return features
    
    def _get_movement_pattern_features(self, positions: List[Tuple[float, float]]) -> Dict[str, float]:
        """Calculate movement pattern features"""
        features = {}
        
        if len(positions) < 3:
            return {'movement_linearity': 0, 'movement_curvature': 0, 'movement_efficiency': 0}
        
        # Calculate total distance traveled
        total_distance = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        # Calculate straight-line distance
        start_pos = positions[0]
        end_pos = positions[-1]
        straight_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # Movement efficiency (straight distance / total distance)
        movement_efficiency = straight_distance / total_distance if total_distance > 0 else 0
        
        # Movement linearity (how close to a straight line)
        if len(positions) >= 3:
            # Fit a line to the positions
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            
            # Calculate R-squared for linearity
            if len(set(x_coords)) > 1:  # Check if x coordinates vary
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_coords, y_coords)
                movement_linearity = r_value**2
            else:
                movement_linearity = 0
        else:
            movement_linearity = 0
        
        # Movement curvature (how much the path curves)
        if len(positions) >= 3:
            # Calculate curvature using three consecutive points
            curvatures = []
            for i in range(1, len(positions) - 1):
                p1, p2, p3 = positions[i-1], positions[i], positions[i+1]
                
                # Calculate curvature using the formula for discrete points
                a = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                b = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
                c = np.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2)
                
                if a > 0 and b > 0 and c > 0:
                    # Heron's formula for area
                    s = (a + b + c) / 2
                    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                    curvature = 4 * area / (a * b * c) if a * b * c > 0 else 0
                    curvatures.append(curvature)
            
            movement_curvature = np.mean(curvatures) if curvatures else 0
        else:
            movement_curvature = 0
        
        features.update({
            'movement_linearity': movement_linearity,
            'movement_curvature': movement_curvature,
            'movement_efficiency': movement_efficiency,
            'total_distance_traveled': total_distance,
            'straight_line_distance': straight_distance,
        })
        
        return features
    
    def _get_statistical_features(self, positions: List[Tuple[float, float]]) -> Dict[str, float]:
        """Calculate statistical features of movement"""
        features = {}
        
        if len(positions) < 2:
            return {'position_variance_x': 0, 'position_variance_y': 0, 'movement_entropy': 0}
        
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        
        features.update({
            'position_mean_x': np.mean(x_coords),
            'position_mean_y': np.mean(y_coords),
            'position_variance_x': np.var(x_coords),
            'position_variance_y': np.var(y_coords),
            'position_std_x': np.std(x_coords),
            'position_std_y': np.std(y_coords),
            'position_range_x': np.max(x_coords) - np.min(x_coords),
            'position_range_y': np.max(y_coords) - np.min(y_coords),
        })
        
        # Movement entropy (how random the movement is)
        if len(positions) >= 3:
            # Calculate direction changes
            direction_changes = []
            for i in range(1, len(positions) - 1):
                dx1 = positions[i][0] - positions[i-1][0]
                dy1 = positions[i][1] - positions[i-1][1]
                dx2 = positions[i+1][0] - positions[i][0]
                dy2 = positions[i+1][1] - positions[i][1]
                
                if dx1 != 0 or dy1 != 0:
                    angle1 = np.arctan2(dy1, dx1)
                    angle2 = np.arctan2(dy2, dx2)
                    angle_diff = abs(angle2 - angle1)
                    direction_changes.append(min(angle_diff, 2*np.pi - angle_diff))
            
            if direction_changes:
                # Calculate entropy of direction changes
                hist, _ = np.histogram(direction_changes, bins=10, density=True)
                hist = hist[hist > 0]  # Remove zero bins
                entropy = -np.sum(hist * np.log2(hist))
                features['movement_entropy'] = entropy
            else:
                features['movement_entropy'] = 0
        else:
            features['movement_entropy'] = 0
        
        return features
    
    def _get_default_movement_features(self) -> Dict[str, float]:
        """Return default values when insufficient data"""
        return {
            'avg_speed': 0, 'max_speed': 0, 'min_speed': 0,
            'speed_variance': 0, 'speed_std': 0, 'speed_range': 0,
            'avg_acceleration': 0, 'max_acceleration': 0, 'min_acceleration': 0,
            'acceleration_variance': 0, 'acceleration_std': 0,
            'avg_direction': 0, 'direction_variance': 0, 'direction_consistency': 0,
            'direction_std': 0, 'movement_linearity': 0, 'movement_curvature': 0,
            'movement_efficiency': 0, 'total_distance_traveled': 0,
            'straight_line_distance': 0, 'position_mean_x': 0, 'position_mean_y': 0,
            'position_variance_x': 0, 'position_variance_y': 0,
            'position_std_x': 0, 'position_std_y': 0,
            'position_range_x': 0, 'position_range_y': 0, 'movement_entropy': 0
        }

class TeamFormationAnalyzer:
    """
    Analyze team formations and tactical patterns
    """
    
    def __init__(self):
        self.formation_patterns = {
            '4-4-2': [(0.1, 0.2), (0.3, 0.2), (0.7, 0.2), (0.9, 0.2),
                     (0.1, 0.5), (0.3, 0.5), (0.7, 0.5), (0.9, 0.5),
                     (0.3, 0.8), (0.7, 0.8)],
            '4-3-3': [(0.1, 0.2), (0.3, 0.2), (0.7, 0.2), (0.9, 0.2),
                     (0.2, 0.5), (0.5, 0.5), (0.8, 0.5),
                     (0.2, 0.8), (0.5, 0.8), (0.8, 0.8)],
            '3-5-2': [(0.2, 0.2), (0.5, 0.2), (0.8, 0.2),
                     (0.1, 0.4), (0.3, 0.4), (0.5, 0.4), (0.7, 0.4), (0.9, 0.4),
                     (0.3, 0.8), (0.7, 0.8)],
        }
    
    def analyze_formation(self, team_positions: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Analyze team formation from player positions
        """
        if len(team_positions) < 8:  # Need at least 8 outfield players
            return {'formation': 'unknown', 'confidence': 0.0}
        
        best_formation = 'unknown'
        best_score = 0.0
        
        for formation_name, pattern_positions in self.formation_patterns.items():
            if len(pattern_positions) <= len(team_positions):
                score = self._calculate_formation_score(team_positions, pattern_positions)
                if score > best_score:
                    best_score = score
                    best_formation = formation_name
        
        return {
            'formation': best_formation,
            'confidence': best_score,
            'player_count': len(team_positions)
        }
    
    def _calculate_formation_score(self, actual_positions: List[Tuple[float, float]], 
                                  pattern_positions: List[Tuple[float, float]]) -> float:
        """Calculate how well actual positions match a formation pattern"""
        if len(actual_positions) < len(pattern_positions):
            return 0.0
        
        # Use Hungarian algorithm for optimal assignment
        from scipy.optimize import linear_sum_assignment
        
        # Calculate distance matrix
        distances = np.zeros((len(actual_positions), len(pattern_positions)))
        for i, actual_pos in enumerate(actual_positions):
            for j, pattern_pos in enumerate(pattern_positions):
                distances[i, j] = np.sqrt((actual_pos[0] - pattern_pos[0])**2 + 
                                        (actual_pos[1] - pattern_pos[1])**2)
        
        # Find optimal assignment
        row_indices, col_indices = linear_sum_assignment(distances)
        
        # Calculate score (lower distance = higher score)
        total_distance = distances[row_indices, col_indices].sum()
        max_possible_distance = np.sqrt(2) * len(pattern_positions)  # Diagonal of unit square
        score = 1 - (total_distance / max_possible_distance)
        
        return max(0, score)
    
    def analyze_team_spread(self, team_positions: List[Tuple[float, float]]) -> Dict[str, float]:
        """Analyze how spread out the team is"""
        if len(team_positions) < 2:
            return {'spread_x': 0, 'spread_y': 0, 'spread_ratio': 0}
        
        x_coords = [pos[0] for pos in team_positions]
        y_coords = [pos[1] for pos in team_positions]
        
        spread_x = np.std(x_coords)
        spread_y = np.std(y_coords)
        spread_ratio = spread_x / spread_y if spread_y > 0 else 0
        
        return {
            'spread_x': spread_x,
            'spread_y': spread_y,
            'spread_ratio': spread_ratio,
            'center_x': np.mean(x_coords),
            'center_y': np.mean(y_coords)
        }

class BallTrajectoryAnalyzer:
    """
    Analyze ball trajectory and movement patterns
    """
    
    def __init__(self):
        self.trajectory_features = {}
    
    def analyze_ball_movement(self, ball_positions: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Analyze ball movement patterns
        ball_positions: List of (x, y, timestamp)
        """
        if len(ball_positions) < 2:
            return self._get_default_ball_features()
        
        # Extract coordinates and timestamps
        x_coords = [pos[0] for pos in ball_positions]
        y_coords = [pos[1] for pos in ball_positions]
        timestamps = [pos[2] for pos in ball_positions]
        
        features = {}
        
        # Speed analysis
        features.update(self._analyze_ball_speed(x_coords, y_coords, timestamps))
        
        # Direction analysis
        features.update(self._analyze_ball_direction(x_coords, y_coords))
        
        # Trajectory analysis
        features.update(self._analyze_ball_trajectory(x_coords, y_coords))
        
        # Field position analysis
        features.update(self._analyze_ball_field_position(x_coords, y_coords))
        
        return features
    
    def _analyze_ball_speed(self, x_coords: List[float], y_coords: List[float], 
                           timestamps: List[float]) -> Dict[str, float]:
        """Analyze ball speed patterns"""
        features = {}
        
        speeds = []
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            dt = timestamps[i] - timestamps[i-1] if len(timestamps) > i else 1
            
            if dt > 0:
                speed = np.sqrt(dx**2 + dy**2) / dt
                speeds.append(speed)
        
        if speeds:
            features.update({
                'ball_avg_speed': np.mean(speeds),
                'ball_max_speed': np.max(speeds),
                'ball_speed_variance': np.var(speeds),
                'ball_speed_std': np.std(speeds),
            })
        else:
            features.update({
                'ball_avg_speed': 0, 'ball_max_speed': 0,
                'ball_speed_variance': 0, 'ball_speed_std': 0
            })
        
        return features
    
    def _analyze_ball_direction(self, x_coords: List[float], y_coords: List[float]) -> Dict[str, float]:
        """Analyze ball direction patterns"""
        features = {}
        
        if len(x_coords) < 2:
            return {'ball_direction_consistency': 0, 'ball_direction_variance': 0}
        
        directions = []
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            direction = np.arctan2(dy, dx)
            directions.append(direction)
        
        if directions:
            # Calculate direction consistency
            cos_dirs = [np.cos(d) for d in directions]
            sin_dirs = [np.sin(d) for d in directions]
            
            avg_cos = np.mean(cos_dirs)
            avg_sin = np.mean(sin_dirs)
            direction_consistency = np.sqrt(avg_cos**2 + avg_sin**2)
            
            features.update({
                'ball_direction_consistency': direction_consistency,
                'ball_direction_variance': np.var(directions),
                'ball_direction_std': np.std(directions),
            })
        else:
            features.update({
                'ball_direction_consistency': 0, 'ball_direction_variance': 0,
                'ball_direction_std': 0
            })
        
        return features
    
    def _analyze_ball_trajectory(self, x_coords: List[float], y_coords: List[float]) -> Dict[str, float]:
        """Analyze ball trajectory characteristics"""
        features = {}
        
        if len(x_coords) < 3:
            return {'ball_trajectory_linearity': 0, 'ball_trajectory_curvature': 0}
        
        # Calculate trajectory linearity
        if len(set(x_coords)) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_coords, y_coords)
            trajectory_linearity = r_value**2
        else:
            trajectory_linearity = 0
        
        # Calculate trajectory curvature
        curvatures = []
        for i in range(1, len(x_coords) - 1):
            p1 = (x_coords[i-1], y_coords[i-1])
            p2 = (x_coords[i], y_coords[i])
            p3 = (x_coords[i+1], y_coords[i+1])
            
            # Calculate curvature using three points
            a = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            b = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
            c = np.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2)
            
            if a > 0 and b > 0 and c > 0:
                s = (a + b + c) / 2
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                curvature = 4 * area / (a * b * c) if a * b * c > 0 else 0
                curvatures.append(curvature)
        
        trajectory_curvature = np.mean(curvatures) if curvatures else 0
        
        features.update({
            'ball_trajectory_linearity': trajectory_linearity,
            'ball_trajectory_curvature': trajectory_curvature,
        })
        
        return features
    
    def _analyze_ball_field_position(self, x_coords: List[float], y_coords: List[float]) -> Dict[str, float]:
        """Analyze ball field position patterns"""
        features = {}
        
        if not x_coords or not y_coords:
            return {'ball_field_coverage': 0, 'ball_zone_distribution': 0}
        
        # Calculate field coverage
        x_range = np.max(x_coords) - np.min(x_coords)
        y_range = np.max(y_coords) - np.min(y_coords)
        field_coverage = x_range * y_range
        
        # Calculate zone distribution
        defensive_zone = sum(1 for y in y_coords if y < 0.33)
        midfield_zone = sum(1 for y in y_coords if 0.33 <= y <= 0.67)
        attacking_zone = sum(1 for y in y_coords if y > 0.67)
        
        total_positions = len(y_coords)
        zone_distribution = max(defensive_zone, midfield_zone, attacking_zone) / total_positions if total_positions > 0 else 0
        
        features.update({
            'ball_field_coverage': field_coverage,
            'ball_zone_distribution': zone_distribution,
            'ball_defensive_zone_time': defensive_zone / total_positions if total_positions > 0 else 0,
            'ball_midfield_zone_time': midfield_zone / total_positions if total_positions > 0 else 0,
            'ball_attacking_zone_time': attacking_zone / total_positions if total_positions > 0 else 0,
        })
        
        return features
    
    def _get_default_ball_features(self) -> Dict[str, float]:
        """Return default ball features when insufficient data"""
        return {
            'ball_avg_speed': 0, 'ball_max_speed': 0, 'ball_speed_variance': 0, 'ball_speed_std': 0,
            'ball_direction_consistency': 0, 'ball_direction_variance': 0, 'ball_direction_std': 0,
            'ball_trajectory_linearity': 0, 'ball_trajectory_curvature': 0,
            'ball_field_coverage': 0, 'ball_zone_distribution': 0,
            'ball_defensive_zone_time': 0, 'ball_midfield_zone_time': 0, 'ball_attacking_zone_time': 0
        }

class AdvancedFeatureEngineer:
    """
    Main feature engineering class that combines all feature extractors
    """
    
    def __init__(self):
        self.spatial_extractor = SpatialFeatureExtractor()
        self.temporal_extractor = TemporalFeatureExtractor()
        self.formation_analyzer = TeamFormationAnalyzer()
        self.ball_analyzer = BallTrajectoryAnalyzer()
        
        # Feature scalers
        self.scalers = {
            'spatial': StandardScaler(),
            'temporal': StandardScaler(),
            'team': StandardScaler(),
            'ball': StandardScaler()
        }
        
        # Feature encoders
        self.encoders = {
            'field_quadrant': LabelEncoder(),
            'formation': LabelEncoder()
        }
    
    def extract_comprehensive_features(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comprehensive features from frame data
        """
        features = {}
        
        # Extract spatial features for each detection
        if 'detections' in frame_data:
            spatial_features = []
            for detection in frame_data['detections']:
                bbox = detection.get('bbox', [0, 0, 0, 0])
                image_shape = frame_data.get('image_shape', (640, 640))
                spatial_feat = self.spatial_extractor.extract_position_features(bbox, image_shape)
                spatial_features.append(spatial_feat)
            
            features['spatial_features'] = spatial_features
        
        # Extract temporal features if tracking history is available
        if 'tracking_history' in frame_data:
            temporal_features = []
            for track_id, history in frame_data['tracking_history'].items():
                temporal_feat = self.temporal_extractor.extract_movement_features(history)
                temporal_features.append(temporal_feat)
            
            features['temporal_features'] = temporal_features
        
        # Analyze team formations
        if 'team_positions' in frame_data:
            team_a_positions = frame_data['team_positions'].get('team_a', [])
            team_b_positions = frame_data['team_positions'].get('team_b', [])
            
            if team_a_positions:
                formation_a = self.formation_analyzer.analyze_formation(team_a_positions)
                spread_a = self.formation_analyzer.analyze_team_spread(team_a_positions)
                features['team_a_formation'] = formation_a
                features['team_a_spread'] = spread_a
            
            if team_b_positions:
                formation_b = self.formation_analyzer.analyze_formation(team_b_positions)
                spread_b = self.formation_analyzer.analyze_team_spread(team_b_positions)
                features['team_b_formation'] = formation_b
                features['team_b_spread'] = spread_b
        
        # Analyze ball trajectory
        if 'ball_positions' in frame_data:
            ball_features = self.ball_analyzer.analyze_ball_movement(frame_data['ball_positions'])
            features['ball_features'] = ball_features
        
        return features
    
    def normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize features using fitted scalers
        """
        normalized_features = {}
        
        for feature_type, scaler in self.scalers.items():
            if feature_type in features:
                if isinstance(features[feature_type], list):
                    # Handle list of feature dictionaries
                    normalized_list = []
                    for feat_dict in features[feature_type]:
                        if isinstance(feat_dict, dict):
                            # Convert to array, normalize, convert back
                            feat_array = np.array(list(feat_dict.values())).reshape(1, -1)
                            normalized_array = scaler.transform(feat_array)
                            normalized_dict = dict(zip(feat_dict.keys(), normalized_array[0]))
                            normalized_list.append(normalized_dict)
                        else:
                            normalized_list.append(feat_dict)
                    normalized_features[feature_type] = normalized_list
                else:
                    # Handle single feature dictionary
                    if isinstance(features[feature_type], dict):
                        feat_array = np.array(list(features[feature_type].values())).reshape(1, -1)
                        normalized_array = scaler.transform(feat_array)
                        normalized_dict = dict(zip(features[feature_type].keys(), normalized_array[0]))
                        normalized_features[feature_type] = normalized_dict
                    else:
                        normalized_features[feature_type] = features[feature_type]
        
        return normalized_features
    
    def fit_scalers(self, training_features: List[Dict[str, Any]]):
        """
        Fit scalers on training data
        """
        for feature_type in self.scalers.keys():
            feature_data = []
            
            for sample in training_features:
                if feature_type in sample:
                    if isinstance(sample[feature_type], list):
                        for feat_dict in sample[feature_type]:
                            if isinstance(feat_dict, dict):
                                feature_data.append(list(feat_dict.values()))
                    elif isinstance(sample[feature_type], dict):
                        feature_data.append(list(sample[feature_type].values()))
            
            if feature_data:
                self.scalers[feature_type].fit(feature_data)
                logger.info(f"Fitted scaler for {feature_type} features on {len(feature_data)} samples")

def test_feature_engineering():
    """
    Test the feature engineering pipeline
    """
    # Create sample data
    sample_frame_data = {
        'detections': [
            {'bbox': [0.1, 0.2, 0.15, 0.3], 'class': 0},  # team_a_player
            {'bbox': [0.3, 0.4, 0.35, 0.5], 'class': 2},  # team_b_player
            {'bbox': [0.5, 0.6, 0.55, 0.7], 'class': 5},  # ball
        ],
        'image_shape': (640, 640),
        'tracking_history': {
            'player_1': [
                {'x_center': 0.1, 'y_center': 0.2, 'timestamp': 0},
                {'x_center': 0.12, 'y_center': 0.22, 'timestamp': 1},
                {'x_center': 0.14, 'y_center': 0.24, 'timestamp': 2},
            ]
        },
        'team_positions': {
            'team_a': [(0.1, 0.2), (0.3, 0.2), (0.5, 0.2), (0.7, 0.2)],
            'team_b': [(0.1, 0.8), (0.3, 0.8), (0.5, 0.8), (0.7, 0.8)]
        },
        'ball_positions': [(0.5, 0.6, 0), (0.52, 0.62, 1), (0.54, 0.64, 2)]
    }
    
    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer()
    
    # Extract features
    features = feature_engineer.extract_comprehensive_features(sample_frame_data)
    
    print("Extracted features:")
    for key, value in features.items():
        print(f"{key}: {type(value)} - {len(value) if isinstance(value, list) else 'single'}")
    
    # Test individual extractors
    print("\nTesting spatial feature extraction...")
    spatial_feat = feature_engineer.spatial_extractor.extract_position_features([0.1, 0.2, 0.15, 0.3], (640, 640))
    print(f"Spatial features: {len(spatial_feat)} features extracted")
    
    print("\nTesting temporal feature extraction...")
    temporal_feat = feature_engineer.temporal_extractor.extract_movement_features(sample_frame_data['tracking_history']['player_1'])
    print(f"Temporal features: {len(temporal_feat)} features extracted")
    
    print("\nTesting formation analysis...")
    formation_feat = feature_engineer.formation_analyzer.analyze_formation(sample_frame_data['team_positions']['team_a'])
    print(f"Formation analysis: {formation_feat}")
    
    print("\nTesting ball trajectory analysis...")
    ball_feat = feature_engineer.ball_analyzer.analyze_ball_movement(sample_frame_data['ball_positions'])
    print(f"Ball features: {len(ball_feat)} features extracted")

if __name__ == "__main__":
    test_feature_engineering()

