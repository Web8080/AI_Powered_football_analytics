#!/usr/bin/env python3
"""
===============================================================================
GODSEYE AI - COMPREHENSIVE FOOTBALL CLASSES
===============================================================================

Defines comprehensive classes for professional football analytics:

CORE CLASSES (Essential):
- Players (Team A/B with jersey numbers)
- Goalkeepers (Team A/B)
- Referees (Main + Assistant)
- Ball
- Goalposts
- Field markings

EXTENDED CLASSES (Advanced):
- Coaching staff
- Medical staff
- Ball boys
- Security personnel
- Equipment (cones, flags, etc.)
- Stadium elements

Author: Victor
Date: 2025
Version: 1.0.0
"""

class ComprehensiveFootballClasses:
    """Comprehensive football class definitions for professional analytics"""
    
    # CORE ESSENTIAL CLASSES (Minimum viable)
    CORE_CLASSES = [
        'team_a_player',      # 0 - Team A outfield players
        'team_a_goalkeeper',  # 1 - Team A goalkeeper
        'team_b_player',      # 2 - Team B outfield players
        'team_b_goalkeeper',  # 3 - Team B goalkeeper
        'referee',            # 4 - Main referee
        'assistant_referee',  # 5 - Assistant referees (linesmen)
        'ball',               # 6 - Football
        'goalpost',           # 7 - Goal posts
        'other',              # 8 - Other people/objects
    ]
    
    # EXTENDED CLASSES (Professional level)
    EXTENDED_CLASSES = [
        'team_a_player',      # 0
        'team_a_goalkeeper',  # 1
        'team_b_player',      # 2
        'team_b_goalkeeper',  # 3
        'referee',            # 4
        'assistant_referee',  # 5
        'ball',               # 6
        'goalpost',           # 7
        'coaching_staff',     # 8 - Coaches, managers
        'medical_staff',      # 9 - Physios, doctors
        'ball_boy',           # 10 - Ball boys/girls
        'security',           # 11 - Security personnel
        'equipment',          # 12 - Cones, flags, etc.
        'stadium_element',    # 13 - Ad boards, cameras, etc.
        'other',              # 14 - Unidentified
    ]
    
    # PROFESSIONAL CLASSES (Broadcast/Scouting level)
    PROFESSIONAL_CLASSES = [
        'team_a_player',      # 0
        'team_a_goalkeeper',  # 1
        'team_b_player',      # 2
        'team_b_goalkeeper',  # 3
        'referee',            # 4
        'assistant_referee',  # 5
        'ball',               # 6
        'goalpost',           # 7
        'coaching_staff',     # 8
        'medical_staff',      # 9
        'ball_boy',           # 10
        'security',           # 11
        'equipment',          # 12
        'stadium_element',    # 13
        'camera_operator',    # 14 - Broadcast crew
        'photographer',       # 15 - Press photographers
        'match_official',     # 16 - Fourth official, etc.
        'other',              # 17
    ]
    
    # Color mapping for visualization
    CLASS_COLORS = {
        # Core classes
        'team_a_player': (255, 0, 0),      # Red
        'team_a_goalkeeper': (200, 0, 0),  # Dark Red
        'team_b_player': (0, 0, 255),      # Blue
        'team_b_goalkeeper': (0, 0, 200),  # Dark Blue
        'referee': (0, 255, 0),            # Green
        'assistant_referee': (0, 200, 0),  # Dark Green
        'ball': (255, 255, 0),             # Yellow
        'goalpost': (255, 255, 255),       # White
        'other': (128, 128, 128),          # Gray
        
        # Extended classes
        'coaching_staff': (255, 165, 0),   # Orange
        'medical_staff': (255, 0, 255),    # Magenta
        'ball_boy': (0, 255, 255),         # Cyan
        'security': (139, 69, 19),         # Brown
        'equipment': (192, 192, 192),      # Silver
        'stadium_element': (64, 64, 64),   # Dark Gray
        'camera_operator': (128, 0, 128),  # Purple
        'photographer': (255, 20, 147),    # Deep Pink
        'match_official': (0, 128, 128),   # Teal
    }
    
    @classmethod
    def get_classes(cls, level='core'):
        """Get classes based on complexity level"""
        if level == 'core':
            return cls.CORE_CLASSES
        elif level == 'extended':
            return cls.EXTENDED_CLASSES
        elif level == 'professional':
            return cls.PROFESSIONAL_CLASSES
        else:
            return cls.CORE_CLASSES
    
    @classmethod
    def get_class_colors(cls, level='core'):
        """Get class colors for visualization"""
        classes = cls.get_classes(level)
        return {cls_name: cls.CLASS_COLORS.get(cls_name, (255, 255, 255)) 
                for cls_name in classes}
    
    @classmethod
    def get_class_info(cls, level='core'):
        """Get comprehensive class information"""
        classes = cls.get_classes(level)
        colors = cls.get_class_colors(level)
        
        return {
            'classes': classes,
            'num_classes': len(classes),
            'class_to_idx': {cls_name: idx for idx, cls_name in enumerate(classes)},
            'idx_to_class': {idx: cls_name for idx, cls_name in enumerate(classes)},
            'colors': colors
        }

# Usage examples
if __name__ == "__main__":
    # Test different levels
    for level in ['core', 'extended', 'professional']:
        print(f"\n=== {level.upper()} LEVEL ===")
        info = ComprehensiveFootballClasses.get_class_info(level)
        print(f"Number of classes: {info['num_classes']}")
        print("Classes:")
        for idx, class_name in info['idx_to_class'].items():
            color = info['colors'][class_name]
            print(f"  {idx}: {class_name} {color}")
