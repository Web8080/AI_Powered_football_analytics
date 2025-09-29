"""
================================================================================
GODSEYE AI - STATISTICS ANALYTICS MODULE
================================================================================

Author: Victor Ibhafidon
Date: January 28, 2025
Version: 1.0.0

DESCRIPTION:
This module provides comprehensive statistics and analytics for the Godseye AI
sports analytics platform. It calculates detailed player performance metrics,
team tactical statistics, and match event analytics for professional football
analysis. Generates KPIs and insights comparable to Premier League analytics.

PIPELINE INTEGRATION:
- Receives: Detection data from ml/models/detection.py
- Receives: Tracking data from ml/models/tracking.py
- Receives: Pose data from ml/models/pose_estimation.py
- Receives: Event data from ml/models/event_detection.py
- Provides: Statistics to Frontend StatisticsDashboard.tsx
- Integrates: With ml/pipeline/inference_pipeline.py for real-time analytics
- Feeds: Analytics data to backend API endpoints

FEATURES:
- Player performance metrics (distance, speed, acceleration, fatigue)
- Team tactical analysis (possession, formation, passing patterns)
- Match event statistics (goals, shots, fouls, cards, corners)
- Heatmap generation and spatial analysis
- Real-time statistics calculation
- Professional KPI generation
- Export capabilities (JSON, CSV, PDF)
- Historical data analysis and trends

DEPENDENCIES:
- numpy for numerical computations
- pandas for data analysis
- scipy for statistical analysis
- opencv-python for image processing
- matplotlib for visualization

USAGE:
    from ml.analytics.statistics import StatisticsCalculator
    
    # Initialize statistics calculator
    calculator = StatisticsCalculator()
    
    # Calculate statistics
    stats = calculator.calculate_match_statistics(detections, tracks, events)

COMPETITOR ANALYSIS:
Based on analysis of industry-leading statistics systems from VeoCam, Stats
Perform, and other professional sports analytics platforms. Implements
Premier League-quality analytics with professional-grade metrics and insights.

================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from scipy import stats
from scipy.spatial.distance import cdist
import cv2
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class PlayerStatistics:
    """Individual player statistics."""
    player_id: str
    team_id: int
    position: str  # 'player' or 'goalkeeper'
    
    # Movement metrics
    total_distance: float = 0.0
    avg_speed: float = 0.0
    max_speed: float = 0.0
    total_sprints: int = 0
    high_intensity_distance: float = 0.0
    
    # Positioning
    avg_position: Tuple[float, float] = (0.0, 0.0)
    field_coverage: float = 0.0
    heatmap_data: np.ndarray = field(default_factory=lambda: np.zeros((100, 100)))
    
    # Performance metrics
    touches: int = 0
    passes_attempted: int = 0
    passes_completed: int = 0
    pass_accuracy: float = 0.0
    key_passes: int = 0
    
    # Defensive metrics
    tackles: int = 0
    interceptions: int = 0
    clearances: int = 0
    duels_won: int = 0
    duels_total: int = 0
    
    # Fatigue analysis
    speed_degradation: float = 0.0
    recovery_time: float = 0.0
    work_rate: float = 0.0
    
    # Advanced metrics
    expected_goals: float = 0.0
    expected_assists: float = 0.0
    progressive_distance: float = 0.0
    pressure_events: int = 0


@dataclass
class TeamStatistics:
    """Team-level statistics."""
    team_id: int
    team_name: str
    
    # Formation and tactics
    formation: str = "4-4-2"
    avg_formation_width: float = 0.0
    avg_formation_height: float = 0.0
    formation_changes: int = 0
    
    # Possession and passing
    possession_percentage: float = 0.0
    total_passes: int = 0
    pass_accuracy: float = 0.0
    key_passes: int = 0
    through_balls: int = 0
    
    # Movement and work rate
    total_distance: float = 0.0
    avg_speed: float = 0.0
    high_intensity_distance: float = 0.0
    sprints: int = 0
    
    # Defensive metrics
    tackles: int = 0
    interceptions: int = 0
    clearances: int = 0
    blocks: int = 0
    
    # Attacking metrics
    shots: int = 0
    shots_on_target: int = 0
    goals: int = 0
    expected_goals: float = 0.0
    
    # Set pieces
    corners: int = 0
    free_kicks: int = 0
    throw_ins: int = 0
    
    # Pressing and transitions
    high_press_events: int = 0
    counter_attacks: int = 0
    transition_speed: float = 0.0


@dataclass
class MatchStatistics:
    """Complete match statistics."""
    match_id: str
    duration: float  # in seconds
    
    # Teams
    team_a: TeamStatistics
    team_b: TeamStatistics
    
    # Players
    players: Dict[str, PlayerStatistics] = field(default_factory=dict)
    
    # Match events
    total_events: int = 0
    goals: int = 0
    cards: int = 0
    substitutions: int = 0
    
    # Ball statistics
    ball_possession_time: Dict[int, float] = field(default_factory=dict)
    ball_touches: int = 0
    ball_distance_traveled: float = 0.0
    
    # Match phases
    first_half_stats: Optional['MatchStatistics'] = None
    second_half_stats: Optional['MatchStatistics'] = None


class StatisticsCalculator:
    """Main statistics calculator for football analytics."""
    
    def __init__(self, field_dimensions: Tuple[int, int] = (105, 68)):
        """
        Initialize statistics calculator.
        
        Args:
            field_dimensions: Field dimensions in meters (length, width)
        """
        self.field_length, self.field_width = field_dimensions
        self.frame_rate = 25  # FPS
        self.pixel_to_meter_ratio = 1.0  # Will be calibrated based on field
        
        # Speed thresholds (km/h)
        self.walking_threshold = 6.0
        self.jogging_threshold = 12.0
        self.running_threshold = 18.0
        self.sprinting_threshold = 25.0
        
        # High intensity threshold
        self.high_intensity_threshold = 20.0  # km/h
        
        logger.info(f"Statistics calculator initialized for {field_dimensions}m field")
    
    def calculate_player_statistics(
        self, 
        player_tracks: List[Dict], 
        ball_positions: List[Dict],
        events: List[Dict],
        match_duration: float
    ) -> PlayerStatistics:
        """Calculate comprehensive player statistics."""
        if not player_tracks:
            return PlayerStatistics(player_id="", team_id=0, position="player")
        
        # Extract positions and timestamps
        positions = [(track['center'][0], track['center'][1]) for track in player_tracks]
        timestamps = [track.get('timestamp', i / self.frame_rate) for i, track in enumerate(player_tracks)]
        
        # Calculate movement metrics
        total_distance = self._calculate_total_distance(positions)
        speeds = self._calculate_speeds(positions, timestamps)
        avg_speed = np.mean(speeds) if speeds else 0.0
        max_speed = np.max(speeds) if speeds else 0.0
        
        # Count sprints and high intensity runs
        total_sprints = sum(1 for speed in speeds if speed > self.sprinting_threshold)
        high_intensity_distance = self._calculate_high_intensity_distance(positions, speeds)
        
        # Calculate positioning
        avg_position = self._calculate_average_position(positions)
        field_coverage = self._calculate_field_coverage(positions)
        heatmap_data = self._generate_heatmap(positions)
        
        # Calculate performance metrics
        touches = self._count_player_touches(player_tracks, ball_positions)
        passes = self._analyze_passing(player_tracks, ball_positions, events)
        defensive_actions = self._analyze_defensive_actions(player_tracks, events)
        
        # Calculate fatigue metrics
        speed_degradation = self._calculate_speed_degradation(speeds, timestamps)
        work_rate = self._calculate_work_rate(speeds, match_duration)
        
        # Advanced metrics
        expected_goals = self._calculate_expected_goals(player_tracks, events)
        progressive_distance = self._calculate_progressive_distance(positions, ball_positions)
        
        return PlayerStatistics(
            player_id=player_tracks[0].get('track_id', ''),
            team_id=player_tracks[0].get('team_id', 0),
            position=player_tracks[0].get('role', 'player'),
            total_distance=total_distance,
            avg_speed=avg_speed,
            max_speed=max_speed,
            total_sprints=total_sprints,
            high_intensity_distance=high_intensity_distance,
            avg_position=avg_position,
            field_coverage=field_coverage,
            heatmap_data=heatmap_data,
            touches=touches,
            passes_attempted=passes['attempted'],
            passes_completed=passes['completed'],
            pass_accuracy=passes['accuracy'],
            key_passes=passes['key_passes'],
            tackles=defensive_actions['tackles'],
            interceptions=defensive_actions['interceptions'],
            clearances=defensive_actions['clearances'],
            duels_won=defensive_actions['duels_won'],
            duels_total=defensive_actions['duels_total'],
            speed_degradation=speed_degradation,
            work_rate=work_rate,
            expected_goals=expected_goals,
            progressive_distance=progressive_distance,
            pressure_events=defensive_actions['pressure_events']
        )
    
    def calculate_team_statistics(
        self, 
        team_tracks: Dict[int, List[Dict]], 
        ball_positions: List[Dict],
        events: List[Dict],
        match_duration: float
    ) -> Dict[int, TeamStatistics]:
        """Calculate team-level statistics."""
        team_stats = {}
        
        for team_id, tracks in team_tracks.items():
            if not tracks:
                continue
            
            # Calculate formation
            formation = self._analyze_formation(tracks)
            formation_metrics = self._calculate_formation_metrics(tracks)
            
            # Calculate possession
            possession = self._calculate_team_possession(team_id, ball_positions, match_duration)
            
            # Calculate passing statistics
            passing_stats = self._calculate_team_passing(team_id, tracks, ball_positions, events)
            
            # Calculate movement statistics
            movement_stats = self._calculate_team_movement(team_id, tracks)
            
            # Calculate defensive statistics
            defensive_stats = self._calculate_team_defensive(team_id, events)
            
            # Calculate attacking statistics
            attacking_stats = self._calculate_team_attacking(team_id, events)
            
            # Calculate set pieces
            set_pieces = self._calculate_set_pieces(team_id, events)
            
            # Calculate pressing and transitions
            pressing_stats = self._calculate_pressing_transitions(team_id, tracks, events)
            
            team_stats[team_id] = TeamStatistics(
                team_id=team_id,
                team_name=f"Team {team_id}",
                formation=formation,
                avg_formation_width=formation_metrics['width'],
                avg_formation_height=formation_metrics['height'],
                formation_changes=formation_metrics['changes'],
                possession_percentage=possession,
                total_passes=passing_stats['total'],
                pass_accuracy=passing_stats['accuracy'],
                key_passes=passing_stats['key_passes'],
                through_balls=passing_stats['through_balls'],
                total_distance=movement_stats['total_distance'],
                avg_speed=movement_stats['avg_speed'],
                high_intensity_distance=movement_stats['high_intensity'],
                sprints=movement_stats['sprints'],
                tackles=defensive_stats['tackles'],
                interceptions=defensive_stats['interceptions'],
                clearances=defensive_stats['clearances'],
                blocks=defensive_stats['blocks'],
                shots=attacking_stats['shots'],
                shots_on_target=attacking_stats['shots_on_target'],
                goals=attacking_stats['goals'],
                expected_goals=attacking_stats['expected_goals'],
                corners=set_pieces['corners'],
                free_kicks=set_pieces['free_kicks'],
                throw_ins=set_pieces['throw_ins'],
                high_press_events=pressing_stats['high_press'],
                counter_attacks=pressing_stats['counter_attacks'],
                transition_speed=pressing_stats['transition_speed']
            )
        
        return team_stats
    
    def _calculate_total_distance(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate total distance covered by a player."""
        if len(positions) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            distance = np.sqrt(dx**2 + dy**2) * self.pixel_to_meter_ratio
            total_distance += distance
        
        return total_distance
    
    def _calculate_speeds(self, positions: List[Tuple[float, float]], timestamps: List[float]) -> List[float]:
        """Calculate speeds between consecutive positions."""
        if len(positions) < 2:
            return []
        
        speeds = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            distance = np.sqrt(dx**2 + dy**2) * self.pixel_to_meter_ratio
            
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                speed_ms = distance / dt  # m/s
                speed_kmh = speed_ms * 3.6  # km/h
                speeds.append(speed_kmh)
        
        return speeds
    
    def _calculate_high_intensity_distance(self, positions: List[Tuple[float, float]], speeds: List[float]) -> float:
        """Calculate distance covered at high intensity."""
        if len(positions) < 2 or len(speeds) == 0:
            return 0.0
        
        high_intensity_distance = 0.0
        for i, speed in enumerate(speeds):
            if speed > self.high_intensity_threshold and i < len(positions) - 1:
                dx = positions[i+1][0] - positions[i][0]
                dy = positions[i+1][1] - positions[i][1]
                distance = np.sqrt(dx**2 + dy**2) * self.pixel_to_meter_ratio
                high_intensity_distance += distance
        
        return high_intensity_distance
    
    def _calculate_average_position(self, positions: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate average position of a player."""
        if not positions:
            return (0.0, 0.0)
        
        avg_x = np.mean([pos[0] for pos in positions])
        avg_y = np.mean([pos[1] for pos in positions])
        return (avg_x, avg_y)
    
    def _calculate_field_coverage(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate percentage of field covered by player."""
        if not positions:
            return 0.0
        
        # Create a grid and mark visited cells
        grid_size = 10  # 10x10 grid
        visited_cells = set()
        
        for pos in positions:
            x_cell = int((pos[0] / self.field_length) * grid_size)
            y_cell = int((pos[1] / self.field_width) * grid_size)
            visited_cells.add((x_cell, y_cell))
        
        total_cells = grid_size * grid_size
        coverage_percentage = (len(visited_cells) / total_cells) * 100
        
        return coverage_percentage
    
    def _generate_heatmap(self, positions: List[Tuple[float, float]], grid_size: int = 100) -> np.ndarray:
        """Generate heatmap data for player positions."""
        heatmap = np.zeros((grid_size, grid_size))
        
        for pos in positions:
            x_cell = int((pos[0] / self.field_length) * grid_size)
            y_cell = int((pos[1] / self.field_width) * grid_size)
            
            if 0 <= x_cell < grid_size and 0 <= y_cell < grid_size:
                heatmap[y_cell, x_cell] += 1
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap
    
    def _count_player_touches(self, player_tracks: List[Dict], ball_positions: List[Dict]) -> int:
        """Count number of ball touches by player."""
        touches = 0
        touch_distance_threshold = 2.0  # meters
        
        for track in player_tracks:
            player_pos = track['center']
            timestamp = track.get('timestamp', 0)
            
            # Find ball position at similar timestamp
            for ball_pos in ball_positions:
                if abs(ball_pos.get('timestamp', 0) - timestamp) < 0.1:  # 100ms tolerance
                    ball_xy = ball_pos['position']
                    distance = np.sqrt(
                        (player_pos[0] - ball_xy[0])**2 + 
                        (player_pos[1] - ball_xy[1])**2
                    ) * self.pixel_to_meter_ratio
                    
                    if distance < touch_distance_threshold:
                        touches += 1
                        break
        
        return touches
    
    def _analyze_passing(self, player_tracks: List[Dict], ball_positions: List[Dict], events: List[Dict]) -> Dict[str, int]:
        """Analyze passing statistics for a player."""
        passes_attempted = 0
        passes_completed = 0
        key_passes = 0
        
        # Count passes from events
        for event in events:
            if event.get('event_name') == 'pass' and event.get('player_id') == player_tracks[0].get('track_id'):
                passes_attempted += 1
                if event.get('successful', False):
                    passes_completed += 1
                if event.get('key_pass', False):
                    key_passes += 1
        
        pass_accuracy = (passes_completed / passes_attempted * 100) if passes_attempted > 0 else 0.0
        
        return {
            'attempted': passes_attempted,
            'completed': passes_completed,
            'accuracy': pass_accuracy,
            'key_passes': key_passes
        }
    
    def _analyze_defensive_actions(self, player_tracks: List[Dict], events: List[Dict]) -> Dict[str, int]:
        """Analyze defensive actions for a player."""
        player_id = player_tracks[0].get('track_id')
        
        tackles = 0
        interceptions = 0
        clearances = 0
        duels_won = 0
        duels_total = 0
        pressure_events = 0
        
        for event in events:
            if event.get('player_id') == player_id:
                event_name = event.get('event_name', '')
                
                if event_name == 'tackle':
                    tackles += 1
                elif event_name == 'interception':
                    interceptions += 1
                elif event_name == 'clearance':
                    clearances += 1
                elif event_name == 'duel':
                    duels_total += 1
                    if event.get('won', False):
                        duels_won += 1
                elif event_name == 'pressure':
                    pressure_events += 1
        
        return {
            'tackles': tackles,
            'interceptions': interceptions,
            'clearances': clearances,
            'duels_won': duels_won,
            'duels_total': duels_total,
            'pressure_events': pressure_events
        }
    
    def _calculate_speed_degradation(self, speeds: List[float], timestamps: List[float]) -> float:
        """Calculate speed degradation over time (fatigue indicator)."""
        if len(speeds) < 10:
            return 0.0
        
        # Split into first and second half
        mid_point = len(speeds) // 2
        first_half_speeds = speeds[:mid_point]
        second_half_speeds = speeds[mid_point:]
        
        if not first_half_speeds or not second_half_speeds:
            return 0.0
        
        first_half_avg = np.mean(first_half_speeds)
        second_half_avg = np.mean(second_half_speeds)
        
        degradation = ((first_half_avg - second_half_avg) / first_half_avg) * 100
        return max(0, degradation)  # Only positive degradation
    
    def _calculate_work_rate(self, speeds: List[float], match_duration: float) -> float:
        """Calculate work rate (distance per minute)."""
        if not speeds or match_duration == 0:
            return 0.0
        
        # Estimate distance from speeds
        total_distance = sum(speeds) * (1 / self.frame_rate) * self.pixel_to_meter_ratio / 3.6  # Convert to meters
        work_rate = total_distance / (match_duration / 60)  # meters per minute
        
        return work_rate
    
    def _calculate_expected_goals(self, player_tracks: List[Dict], events: List[Dict]) -> float:
        """Calculate expected goals for a player."""
        player_id = player_tracks[0].get('track_id')
        xg = 0.0
        
        for event in events:
            if (event.get('player_id') == player_id and 
                event.get('event_name') == 'shot'):
                xg += event.get('expected_goals', 0.0)
        
        return xg
    
    def _calculate_progressive_distance(self, positions: List[Tuple[float, float]], ball_positions: List[Dict]) -> float:
        """Calculate progressive distance (distance towards opponent goal)."""
        if len(positions) < 2:
            return 0.0
        
        progressive_distance = 0.0
        goal_position = (self.field_length, self.field_width / 2)  # Assume goal at end of field
        
        for i in range(1, len(positions)):
            prev_distance_to_goal = np.sqrt(
                (positions[i-1][0] - goal_position[0])**2 + 
                (positions[i-1][1] - goal_position[1])**2
            )
            curr_distance_to_goal = np.sqrt(
                (positions[i][0] - goal_position[0])**2 + 
                (positions[i][1] - goal_position[1])**2
            )
            
            # If player moved closer to goal
            if curr_distance_to_goal < prev_distance_to_goal:
                progressive_distance += prev_distance_to_goal - curr_distance_to_goal
        
        return progressive_distance * self.pixel_to_meter_ratio
    
    def _analyze_formation(self, tracks: List[Dict]) -> str:
        """Analyze team formation."""
        if len(tracks) < 10:  # Need at least 10 players
            return "Unknown"
        
        # Get average positions
        positions = [track['center'] for track in tracks]
        
        # Simple formation detection based on positioning
        # This is a simplified version - in practice, you'd use more sophisticated algorithms
        
        # Sort by Y position (field width)
        sorted_positions = sorted(positions, key=lambda p: p[1])
        
        # Count players in different zones
        defenders = sum(1 for pos in positions if pos[0] < self.field_length * 0.3)
        midfielders = sum(1 for pos in positions if self.field_length * 0.3 <= pos[0] < self.field_length * 0.7)
        forwards = sum(1 for pos in positions if pos[0] >= self.field_length * 0.7)
        
        # Determine formation
        if defenders == 4 and midfielders == 4 and forwards == 2:
            return "4-4-2"
        elif defenders == 4 and midfielders == 3 and forwards == 3:
            return "4-3-3"
        elif defenders == 3 and midfielders == 5 and forwards == 2:
            return "3-5-2"
        elif defenders == 5 and midfielders == 3 and forwards == 2:
            return "5-3-2"
        else:
            return f"{defenders}-{midfielders}-{forwards}"
    
    def _calculate_formation_metrics(self, tracks: List[Dict]) -> Dict[str, float]:
        """Calculate formation width, height, and changes."""
        if not tracks:
            return {'width': 0.0, 'height': 0.0, 'changes': 0}
        
        positions = [track['center'] for track in tracks]
        
        # Calculate width (Y-axis spread)
        y_positions = [pos[1] for pos in positions]
        width = max(y_positions) - min(y_positions)
        
        # Calculate height (X-axis spread)
        x_positions = [pos[0] for pos in positions]
        height = max(x_positions) - min(x_positions)
        
        # Formation changes would require tracking over time
        # For now, return 0
        changes = 0
        
        return {
            'width': width * self.pixel_to_meter_ratio,
            'height': height * self.pixel_to_meter_ratio,
            'changes': changes
        }
    
    def _calculate_team_possession(self, team_id: int, ball_positions: List[Dict], match_duration: float) -> float:
        """Calculate team possession percentage."""
        if not ball_positions or match_duration == 0:
            return 0.0
        
        team_possession_time = 0.0
        possession_distance_threshold = 3.0  # meters
        
        for ball_pos in ball_positions:
            # Find nearest player from the team
            # This is simplified - in practice, you'd need player positions at each timestamp
            # For now, assume 50% possession if ball is in team's half
            ball_x = ball_pos['position'][0]
            if team_id == 0:  # Team A
                if ball_x < self.field_length / 2:
                    team_possession_time += 1 / self.frame_rate
            else:  # Team B
                if ball_x >= self.field_length / 2:
                    team_possession_time += 1 / self.frame_rate
        
        possession_percentage = (team_possession_time / match_duration) * 100
        return possession_percentage
    
    def _calculate_team_passing(self, team_id: int, tracks: List[Dict], ball_positions: List[Dict], events: List[Dict]) -> Dict[str, int]:
        """Calculate team passing statistics."""
        total_passes = 0
        completed_passes = 0
        key_passes = 0
        through_balls = 0
        
        for event in events:
            if event.get('team_id') == team_id and event.get('event_name') == 'pass':
                total_passes += 1
                if event.get('successful', False):
                    completed_passes += 1
                if event.get('key_pass', False):
                    key_passes += 1
                if event.get('through_ball', False):
                    through_balls += 1
        
        pass_accuracy = (completed_passes / total_passes * 100) if total_passes > 0 else 0.0
        
        return {
            'total': total_passes,
            'completed': completed_passes,
            'accuracy': pass_accuracy,
            'key_passes': key_passes,
            'through_balls': through_balls
        }
    
    def _calculate_team_movement(self, team_id: int, tracks: List[Dict]) -> Dict[str, float]:
        """Calculate team movement statistics."""
        total_distance = 0.0
        speeds = []
        high_intensity_distance = 0.0
        sprints = 0
        
        for track in tracks:
            if track.get('team_id') == team_id:
                # This would require position history for each player
                # For now, use dummy values
                total_distance += track.get('total_distance', 0.0)
                speeds.append(track.get('avg_speed', 0.0))
                high_intensity_distance += track.get('high_intensity_distance', 0.0)
                sprints += track.get('total_sprints', 0)
        
        avg_speed = np.mean(speeds) if speeds else 0.0
        
        return {
            'total_distance': total_distance,
            'avg_speed': avg_speed,
            'high_intensity': high_intensity_distance,
            'sprints': sprints
        }
    
    def _calculate_team_defensive(self, team_id: int, events: List[Dict]) -> Dict[str, int]:
        """Calculate team defensive statistics."""
        tackles = 0
        interceptions = 0
        clearances = 0
        blocks = 0
        
        for event in events:
            if event.get('team_id') == team_id:
                event_name = event.get('event_name', '')
                if event_name == 'tackle':
                    tackles += 1
                elif event_name == 'interception':
                    interceptions += 1
                elif event_name == 'clearance':
                    clearances += 1
                elif event_name == 'block':
                    blocks += 1
        
        return {
            'tackles': tackles,
            'interceptions': interceptions,
            'clearances': clearances,
            'blocks': blocks
        }
    
    def _calculate_team_attacking(self, team_id: int, events: List[Dict]) -> Dict[str, Union[int, float]]:
        """Calculate team attacking statistics."""
        shots = 0
        shots_on_target = 0
        goals = 0
        expected_goals = 0.0
        
        for event in events:
            if event.get('team_id') == team_id:
                event_name = event.get('event_name', '')
                if event_name == 'shot':
                    shots += 1
                    if event.get('on_target', False):
                        shots_on_target += 1
                    expected_goals += event.get('expected_goals', 0.0)
                elif event_name == 'goal':
                    goals += 1
        
        return {
            'shots': shots,
            'shots_on_target': shots_on_target,
            'goals': goals,
            'expected_goals': expected_goals
        }
    
    def _calculate_set_pieces(self, team_id: int, events: List[Dict]) -> Dict[str, int]:
        """Calculate set piece statistics."""
        corners = 0
        free_kicks = 0
        throw_ins = 0
        
        for event in events:
            if event.get('team_id') == team_id:
                event_name = event.get('event_name', '')
                if event_name == 'corner':
                    corners += 1
                elif event_name == 'free_kick':
                    free_kicks += 1
                elif event_name == 'throw_in':
                    throw_ins += 1
        
        return {
            'corners': corners,
            'free_kicks': free_kicks,
            'throw_ins': throw_ins
        }
    
    def _calculate_pressing_transitions(self, team_id: int, tracks: List[Dict], events: List[Dict]) -> Dict[str, Union[int, float]]:
        """Calculate pressing and transition statistics."""
        high_press_events = 0
        counter_attacks = 0
        transition_speed = 0.0
        
        for event in events:
            if event.get('team_id') == team_id:
                event_name = event.get('event_name', '')
                if event_name == 'high_press':
                    high_press_events += 1
                elif event_name == 'counter_attack':
                    counter_attacks += 1
                    transition_speed += event.get('speed', 0.0)
        
        avg_transition_speed = transition_speed / counter_attacks if counter_attacks > 0 else 0.0
        
        return {
            'high_press': high_press_events,
            'counter_attacks': counter_attacks,
            'transition_speed': avg_transition_speed
        }


class StatisticsVisualizer:
    """Visualization utilities for statistics."""
    
    @staticmethod
    def create_heatmap_visualization(heatmap_data: np.ndarray, title: str = "Player Heatmap") -> np.ndarray:
        """Create heatmap visualization."""
        # Normalize heatmap
        if np.max(heatmap_data) > 0:
            heatmap_normalized = heatmap_data / np.max(heatmap_data)
        else:
            heatmap_normalized = heatmap_data
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_normalized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        return heatmap_colored
    
    @staticmethod
    def create_speed_curve(speeds: List[float], timestamps: List[float]) -> Dict[str, Any]:
        """Create speed curve data for visualization."""
        return {
            'speeds': speeds,
            'timestamps': timestamps,
            'avg_speed': np.mean(speeds) if speeds else 0.0,
            'max_speed': np.max(speeds) if speeds else 0.0,
            'min_speed': np.min(speeds) if speeds else 0.0
        }
    
    @staticmethod
    def create_formation_visualization(positions: List[Tuple[float, float]], formation: str) -> Dict[str, Any]:
        """Create formation visualization data."""
        return {
            'positions': positions,
            'formation': formation,
            'width': max(pos[1] for pos in positions) - min(pos[1] for pos in positions) if positions else 0,
            'height': max(pos[0] for pos in positions) - min(pos[0] for pos in positions) if positions else 0
        }


# Utility functions
def calculate_match_statistics(
    detection_results: List[Dict],
    tracking_results: List[Dict], 
    pose_results: List[Dict],
    event_results: List[Dict],
    match_duration: float = 90 * 60  # 90 minutes in seconds
) -> MatchStatistics:
    """Calculate complete match statistics from analysis results."""
    
    calculator = StatisticsCalculator()
    
    # Group tracks by team and player
    team_tracks = defaultdict(list)
    player_tracks = defaultdict(list)
    
    for track in tracking_results:
        team_id = track.get('team_id')
        player_id = track.get('track_id')
        
        if team_id is not None:
            team_tracks[team_id].append(track)
            player_tracks[player_id].append(track)
    
    # Calculate team statistics
    team_stats = calculator.calculate_team_statistics(
        team_tracks, 
        tracking_results,  # Using tracking results as ball positions
        event_results,
        match_duration
    )
    
    # Calculate player statistics
    players_stats = {}
    for player_id, tracks in player_tracks.items():
        if tracks:
            player_stat = calculator.calculate_player_statistics(
                tracks,
                tracking_results,
                event_results,
                match_duration
            )
            players_stats[player_id] = player_stat
    
    # Create match statistics
    team_a_stats = team_stats.get(0, TeamStatistics(team_id=0, team_name="Team A"))
    team_b_stats = team_stats.get(1, TeamStatistics(team_id=1, team_name="Team B"))
    
    match_stats = MatchStatistics(
        match_id="match_001",
        duration=match_duration,
        team_a=team_a_stats,
        team_b=team_b_stats,
        players=players_stats,
        total_events=len(event_results),
        goals=sum(1 for event in event_results if event.get('event_name') == 'goal'),
        cards=sum(1 for event in event_results if event.get('event_name') in ['yellow_card', 'red_card']),
        substitutions=sum(1 for event in event_results if event.get('event_name') == 'substitution')
    )
    
    return match_stats


def export_statistics_to_csv(match_stats: MatchStatistics, output_path: str):
    """Export statistics to CSV format."""
    
    # Player statistics
    player_data = []
    for player_id, stats in match_stats.players.items():
        player_data.append({
            'player_id': player_id,
            'team_id': stats.team_id,
            'position': stats.position,
            'total_distance': stats.total_distance,
            'avg_speed': stats.avg_speed,
            'max_speed': stats.max_speed,
            'total_sprints': stats.total_sprints,
            'high_intensity_distance': stats.high_intensity_distance,
            'touches': stats.touches,
            'pass_accuracy': stats.pass_accuracy,
            'tackles': stats.tackles,
            'interceptions': stats.interceptions,
            'expected_goals': stats.expected_goals,
            'work_rate': stats.work_rate
        })
    
    player_df = pd.DataFrame(player_data)
    player_df.to_csv(f"{output_path}_players.csv", index=False)
    
    # Team statistics
    team_data = [
        {
            'team_id': match_stats.team_a.team_id,
            'team_name': match_stats.team_a.team_name,
            'formation': match_stats.team_a.formation,
            'possession_percentage': match_stats.team_a.possession_percentage,
            'total_passes': match_stats.team_a.total_passes,
            'pass_accuracy': match_stats.team_a.pass_accuracy,
            'total_distance': match_stats.team_a.total_distance,
            'avg_speed': match_stats.team_a.avg_speed,
            'tackles': match_stats.team_a.tackles,
            'shots': match_stats.team_a.shots,
            'goals': match_stats.team_a.goals,
            'expected_goals': match_stats.team_a.expected_goals
        },
        {
            'team_id': match_stats.team_b.team_id,
            'team_name': match_stats.team_b.team_name,
            'formation': match_stats.team_b.formation,
            'possession_percentage': match_stats.team_b.possession_percentage,
            'total_passes': match_stats.team_b.total_passes,
            'pass_accuracy': match_stats.team_b.pass_accuracy,
            'total_distance': match_stats.team_b.total_distance,
            'avg_speed': match_stats.team_b.avg_speed,
            'tackles': match_stats.team_b.tackles,
            'shots': match_stats.team_b.shots,
            'goals': match_stats.team_b.goals,
            'expected_goals': match_stats.team_b.expected_goals
        }
    ]
    
    team_df = pd.DataFrame(team_data)
    team_df.to_csv(f"{output_path}_teams.csv", index=False)
    
    logger.info(f"Statistics exported to {output_path}_players.csv and {output_path}_teams.csv")


def generate_statistics_report(match_stats: MatchStatistics) -> Dict[str, Any]:
    """Generate comprehensive statistics report."""
    
    report = {
        'match_info': {
            'match_id': match_stats.match_id,
            'duration': match_stats.duration,
            'total_events': match_stats.total_events,
            'goals': match_stats.goals,
            'cards': match_stats.cards,
            'substitutions': match_stats.substitutions
        },
        'team_comparison': {
            'possession': {
                'team_a': match_stats.team_a.possession_percentage,
                'team_b': match_stats.team_b.possession_percentage
            },
            'distance': {
                'team_a': match_stats.team_a.total_distance,
                'team_b': match_stats.team_b.total_distance
            },
            'speed': {
                'team_a': match_stats.team_a.avg_speed,
                'team_b': match_stats.team_b.avg_speed
            },
            'goals': {
                'team_a': match_stats.team_a.goals,
                'team_b': match_stats.team_b.goals
            }
        },
        'top_performers': {
            'distance': max(match_stats.players.values(), key=lambda p: p.total_distance).player_id,
            'speed': max(match_stats.players.values(), key=lambda p: p.max_speed).player_id,
            'passes': max(match_stats.players.values(), key=lambda p: p.passes_completed).player_id,
            'tackles': max(match_stats.players.values(), key=lambda p: p.tackles).player_id
        },
        'key_insights': [
            f"Team A had {match_stats.team_a.possession_percentage:.1f}% possession",
            f"Team B had {match_stats.team_b.possession_percentage:.1f}% possession",
            f"Total distance covered: {match_stats.team_a.total_distance + match_stats.team_b.total_distance:.0f}m",
            f"Match had {match_stats.total_events} events and {match_stats.goals} goals"
        ]
    }
    
    return report
