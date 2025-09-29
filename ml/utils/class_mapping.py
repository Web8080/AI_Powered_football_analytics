"""
Comprehensive class mapping utilities for Godseye AI sports analytics.
Handles professional football classification with team-specific detection.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class FootballClassMapper:
    """Professional football class mapping and utilities."""
    
    # Primary class definitions
    CLASSES = [
        'team_a_player',      # Team A outfield players
        'team_a_goalkeeper',  # Team A goalkeeper
        'team_b_player',      # Team B outfield players  
        'team_b_goalkeeper',  # Team B goalkeeper
        'referee',            # Main referee
        'ball',               # Football
        'other',              # Other objects/people outside play
        'staff'               # Medical staff, coaches, ball boys, etc.
    ]
    
    # Class to index mapping
    CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
    
    # Index to class mapping
    IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}
    
    # Team mapping
    TEAM_MAPPING = {
        0: 'team_a',  # Team A
        1: 'team_b'   # Team B
    }
    
    # Role mapping
    ROLE_MAPPING = {
        'player': ['team_a_player', 'team_b_player'],
        'goalkeeper': ['team_a_goalkeeper', 'team_b_goalkeeper'],
        'referee': ['referee'],
        'ball': ['ball'],
        'other': ['other'],
        'staff': ['staff']
    }
    
    # Color mapping for visualization
    CLASS_COLORS = {
        'team_a_player': (255, 0, 0),      # Red
        'team_a_goalkeeper': (200, 0, 0),  # Dark Red
        'team_b_player': (0, 0, 255),      # Blue
        'team_b_goalkeeper': (0, 0, 200),  # Dark Blue
        'referee': (0, 255, 0),            # Green
        'ball': (255, 255, 0),             # Yellow
        'other': (128, 128, 128),          # Gray
        'staff': (255, 0, 255)             # Magenta
    }
    
    # Team colors for visualization
    TEAM_COLORS = {
        0: (255, 0, 0),    # Team A - Red
        1: (0, 0, 255),    # Team B - Blue
        None: (128, 128, 128)  # No team - Gray
    }
    
    @classmethod
    def get_class_index(cls, class_name: str) -> int:
        """Get class index from class name."""
        return cls.CLASS_TO_IDX.get(class_name, -1)
    
    @classmethod
    def get_class_name(cls, class_index: int) -> str:
        """Get class name from class index."""
        return cls.IDX_TO_CLASS.get(class_index, 'unknown')
    
    @classmethod
    def get_team_from_class(cls, class_name: str) -> Optional[int]:
        """Get team ID from class name."""
        if class_name.startswith('team_a'):
            return 0
        elif class_name.startswith('team_b'):
            return 1
        else:
            return None
    
    @classmethod
    def get_team_from_index(cls, class_index: int) -> Optional[int]:
        """Get team ID from class index."""
        class_name = cls.get_class_name(class_index)
        return cls.get_team_from_class(class_name)
    
    @classmethod
    def get_role_from_class(cls, class_name: str) -> str:
        """Get role from class name."""
        if 'goalkeeper' in class_name:
            return 'goalkeeper'
        elif 'player' in class_name:
            return 'player'
        elif class_name == 'referee':
            return 'referee'
        elif class_name == 'ball':
            return 'ball'
        elif class_name == 'staff':
            return 'staff'
        else:
            return 'other'
    
    @classmethod
    def get_role_from_index(cls, class_index: int) -> str:
        """Get role from class index."""
        class_name = cls.get_class_name(class_index)
        return cls.get_role_from_class(class_name)
    
    @classmethod
    def get_class_color(cls, class_name: str) -> Tuple[int, int, int]:
        """Get color for class visualization."""
        return cls.CLASS_COLORS.get(class_name, (128, 128, 128))
    
    @classmethod
    def get_team_color(cls, team_id: Optional[int]) -> Tuple[int, int, int]:
        """Get color for team visualization."""
        return cls.TEAM_COLORS.get(team_id, (128, 128, 128))
    
    @classmethod
    def is_player(cls, class_name: str) -> bool:
        """Check if class is a player (including goalkeeper)."""
        return 'player' in class_name or 'goalkeeper' in class_name
    
    @classmethod
    def is_goalkeeper(cls, class_name: str) -> bool:
        """Check if class is a goalkeeper."""
        return 'goalkeeper' in class_name
    
    @classmethod
    def is_team_a(cls, class_name: str) -> bool:
        """Check if class belongs to Team A."""
        return class_name.startswith('team_a')
    
    @classmethod
    def is_team_b(cls, class_name: str) -> bool:
        """Check if class belongs to Team B."""
        return class_name.startswith('team_b')
    
    @classmethod
    def get_team_classes(cls, team_id: int) -> List[str]:
        """Get all classes for a specific team."""
        if team_id == 0:
            return ['team_a_player', 'team_a_goalkeeper']
        elif team_id == 1:
            return ['team_b_player', 'team_b_goalkeeper']
        else:
            return []
    
    @classmethod
    def get_team_indices(cls, team_id: int) -> List[int]:
        """Get all class indices for a specific team."""
        team_classes = cls.get_team_classes(team_id)
        return [cls.get_class_index(cls_name) for cls_name in team_classes]
    
    @classmethod
    def filter_by_team(cls, detections: List[Dict], team_id: int) -> List[Dict]:
        """Filter detections by team."""
        team_indices = cls.get_team_indices(team_id)
        return [det for det in detections if det.get('class_id') in team_indices]
    
    @classmethod
    def filter_by_role(cls, detections: List[Dict], role: str) -> List[Dict]:
        """Filter detections by role."""
        role_classes = cls.ROLE_MAPPING.get(role, [])
        role_indices = [cls.get_class_index(cls_name) for cls_name in role_classes]
        return [det for det in detections if det.get('class_id') in role_indices]
    
    @classmethod
    def get_team_statistics(cls, detections: List[Dict]) -> Dict[str, Any]:
        """Get team statistics from detections."""
        stats = {
            'team_a': {
                'players': 0,
                'goalkeepers': 0,
                'total': 0
            },
            'team_b': {
                'players': 0,
                'goalkeepers': 0,
                'total': 0
            },
            'referees': 0,
            'balls': 0,
            'other': 0,
            'staff': 0
        }
        
        for detection in detections:
            class_name = cls.get_class_name(detection.get('class_id', -1))
            
            if cls.is_team_a(class_name):
                if cls.is_goalkeeper(class_name):
                    stats['team_a']['goalkeepers'] += 1
                else:
                    stats['team_a']['players'] += 1
                stats['team_a']['total'] += 1
            elif cls.is_team_b(class_name):
                if cls.is_goalkeeper(class_name):
                    stats['team_b']['goalkeepers'] += 1
                else:
                    stats['team_b']['players'] += 1
                stats['team_b']['total'] += 1
            elif class_name == 'referee':
                stats['referees'] += 1
            elif class_name == 'ball':
                stats['balls'] += 1
            elif class_name == 'staff':
                stats['staff'] += 1
            else:
                stats['other'] += 1
        
        return stats
    
    @classmethod
    def validate_formation(cls, detections: List[Dict]) -> Dict[str, Any]:
        """Validate if detected formation is valid."""
        team_a_detections = cls.filter_by_team(detections, 0)
        team_b_detections = cls.filter_by_team(detections, 1)
        
        validation = {
            'team_a': {
                'valid': True,
                'players': len(cls.filter_by_role(team_a_detections, 'player')),
                'goalkeepers': len(cls.filter_by_role(team_a_detections, 'goalkeeper')),
                'issues': []
            },
            'team_b': {
                'valid': True,
                'players': len(cls.filter_by_role(team_b_detections, 'player')),
                'goalkeepers': len(cls.filter_by_role(team_b_detections, 'goalkeeper')),
                'issues': []
            }
        }
        
        # Validate Team A
        if validation['team_a']['goalkeepers'] != 1:
            validation['team_a']['valid'] = False
            validation['team_a']['issues'].append(f"Expected 1 goalkeeper, found {validation['team_a']['goalkeepers']}")
        
        if validation['team_a']['players'] < 10 or validation['team_a']['players'] > 10:
            validation['team_a']['valid'] = False
            validation['team_a']['issues'].append(f"Expected 10 players, found {validation['team_a']['players']}")
        
        # Validate Team B
        if validation['team_b']['goalkeepers'] != 1:
            validation['team_b']['valid'] = False
            validation['team_b']['issues'].append(f"Expected 1 goalkeeper, found {validation['team_b']['goalkeepers']}")
        
        if validation['team_b']['players'] < 10 or validation['team_b']['players'] > 10:
            validation['team_b']['valid'] = False
            validation['team_b']['issues'].append(f"Expected 10 players, found {validation['team_b']['players']}")
        
        return validation
    
    @classmethod
    def convert_legacy_format(cls, detections: List[Dict]) -> List[Dict]:
        """Convert legacy 4-class format to new 8-class format."""
        converted = []
        
        for detection in detections:
            old_class_id = detection.get('class_id', -1)
            old_class_name = ['player', 'ball', 'referee', 'other'][old_class_id] if old_class_id < 4 else 'unknown'
            
            # Convert to new format
            if old_class_name == 'player':
                # Assume all players are Team A for now (can be improved with team classification)
                new_class_id = cls.get_class_index('team_a_player')
                new_class_name = 'team_a_player'
            elif old_class_name == 'ball':
                new_class_id = cls.get_class_index('ball')
                new_class_name = 'ball'
            elif old_class_name == 'referee':
                new_class_id = cls.get_class_index('referee')
                new_class_name = 'referee'
            else:
                new_class_id = cls.get_class_index('other')
                new_class_name = 'other'
            
            converted_detection = detection.copy()
            converted_detection['class_id'] = new_class_id
            converted_detection['class_name'] = new_class_name
            converted_detection['team_id'] = cls.get_team_from_index(new_class_id)
            converted_detection['role'] = cls.get_role_from_index(new_class_id)
            
            converted.append(converted_detection)
        
        return converted
    
    @classmethod
    def get_class_info(cls, class_name: str) -> Dict[str, Any]:
        """Get comprehensive information about a class."""
        return {
            'name': class_name,
            'index': cls.get_class_index(class_name),
            'team_id': cls.get_team_from_class(class_name),
            'role': cls.get_role_from_class(class_name),
            'color': cls.get_class_color(class_name),
            'is_player': cls.is_player(class_name),
            'is_goalkeeper': cls.is_goalkeeper(class_name),
            'is_team_a': cls.is_team_a(class_name),
            'is_team_b': cls.is_team_b(class_name)
        }


# Convenience functions
def get_class_mapper() -> FootballClassMapper:
    """Get the class mapper instance."""
    return FootballClassMapper


def map_detection_to_team_role(detection: Dict) -> Dict[str, Any]:
    """Map detection to team and role information."""
    class_name = FootballClassMapper.get_class_name(detection.get('class_id', -1))
    
    return {
        'team_id': FootballClassMapper.get_team_from_class(class_name),
        'role': FootballClassMapper.get_role_from_class(class_name),
        'is_player': FootballClassMapper.is_player(class_name),
        'is_goalkeeper': FootballClassMapper.is_goalkeeper(class_name),
        'team_name': FootballClassMapper.TEAM_MAPPING.get(
            FootballClassMapper.get_team_from_class(class_name), 'unknown'
        )
    }


def create_team_summary(detections: List[Dict]) -> Dict[str, Any]:
    """Create a summary of team composition from detections."""
    stats = FootballClassMapper.get_team_statistics(detections)
    validation = FootballClassMapper.validate_formation(detections)
    
    return {
        'statistics': stats,
        'validation': validation,
        'total_detections': len(detections),
        'teams_present': [
            team for team in ['team_a', 'team_b'] 
            if stats[team]['total'] > 0
        ]
    }
