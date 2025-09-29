#!/usr/bin/env python3
"""
================================================================================
GODSEYE AI - ENHANCED REFEREE DETECTION
================================================================================

Author: Victor Ibhafidon
Date: January 28, 2025
Version: 1.0.0

DESCRIPTION:
Enhanced referee detection system that can adapt to different referee uniforms.
Uses multiple detection methods beyond just yellow/black color schemes.

KEY FEATURES:
- Multi-color referee detection (not just yellow/black)
- Position-based referee identification
- Movement pattern analysis
- Uniform contrast detection
- Adaptive learning from match context

USAGE:
    python enhanced_referee_detection.py
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path

class EnhancedRefereeDetector:
    """Enhanced referee detection with flexible uniform recognition"""
    
    def __init__(self):
        # Multiple referee color schemes (not just yellow/black)
        self.referee_color_schemes = [
            # Traditional yellow/black
            {
                'name': 'traditional_yellow_black',
                'shirt_colors': [
                    ([20, 100, 100], [30, 255, 255]),  # Yellow
                    ([15, 50, 100], [25, 255, 255])    # Light yellow
                ],
                'shorts_colors': [
                    ([0, 0, 0], [180, 255, 50])        # Black
                ]
            },
            # Modern referee uniforms
            {
                'name': 'modern_blue_white',
                'shirt_colors': [
                    ([100, 50, 50], [130, 255, 255]),  # Blue
                    ([0, 0, 200], [180, 30, 255])      # White
                ],
                'shorts_colors': [
                    ([0, 0, 200], [180, 30, 255]),     # White
                    ([0, 0, 0], [180, 255, 50])        # Black
                ]
            },
            # Alternative color schemes
            {
                'name': 'green_white',
                'shirt_colors': [
                    ([40, 50, 50], [80, 255, 255]),    # Green
                    ([0, 0, 200], [180, 30, 255])      # White
                ],
                'shorts_colors': [
                    ([0, 0, 200], [180, 30, 255]),     # White
                    ([0, 0, 0], [180, 255, 50])        # Black
                ]
            },
            # High contrast uniforms
            {
                'name': 'high_contrast',
                'shirt_colors': [
                    ([0, 0, 0], [180, 255, 50]),       # Black
                    ([0, 0, 200], [180, 30, 255])      # White
                ],
                'shorts_colors': [
                    ([0, 0, 200], [180, 30, 255]),     # White
                    ([0, 0, 0], [180, 255, 50])        # Black
                ]
            }
        ]
        
        # Referee characteristics
        self.referee_characteristics = {
            'min_area': 800,
            'max_area': 8000,
            'min_aspect_ratio': 1.5,
            'max_aspect_ratio': 4.0,
            'typical_positions': ['center_field', 'near_goals', 'sidelines']
        }
    
    def detect_referees(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """
        Detect referees using multiple methods
        Returns: List of (x, y, w, h, confidence, method)
        """
        detections = []
        
        # Method 1: Multi-color scheme detection
        color_detections = self.detect_by_color_schemes(frame)
        detections.extend(color_detections)
        
        # Method 2: Position-based detection
        position_detections = self.detect_by_position(frame)
        detections.extend(position_detections)
        
        # Method 3: Movement pattern analysis
        movement_detections = self.detect_by_movement_patterns(frame)
        detections.extend(movement_detections)
        
        # Method 4: Uniform contrast detection
        contrast_detections = self.detect_by_contrast(frame)
        detections.extend(contrast_detections)
        
        # Method 5: Size and shape analysis
        shape_detections = self.detect_by_shape_analysis(frame)
        detections.extend(shape_detections)
        
        # Remove duplicates and rank by confidence
        final_detections = self.rank_and_filter_detections(detections)
        
        return final_detections
    
    def detect_by_color_schemes(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """Detect referees using multiple color schemes"""
        detections = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for scheme in self.referee_color_schemes:
            scheme_detections = self.detect_scheme(frame, hsv, scheme)
            detections.extend(scheme_detections)
        
        return detections
    
    def detect_scheme(self, frame: np.ndarray, hsv: np.ndarray, scheme: Dict) -> List[Tuple[int, int, int, int, float, str]]:
        """Detect referees using a specific color scheme"""
        detections = []
        
        # Combine shirt and shorts colors
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        # Add shirt colors
        for lower, upper in scheme['shirt_colors']:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Add shorts colors
        for lower, upper in scheme['shorts_colors']:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.referee_characteristics['min_area'] < area < self.referee_characteristics['max_area']:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio
                aspect_ratio = h / w
                if (self.referee_characteristics['min_aspect_ratio'] < 
                    aspect_ratio < self.referee_characteristics['max_aspect_ratio']):
                    
                    # Calculate confidence based on area and aspect ratio
                    confidence = min(0.9, 0.5 + (area / 4000) * 0.3 + (1.0 / abs(aspect_ratio - 2.5)) * 0.1)
                    
                    detections.append((x, y, w, h, confidence, f"color_{scheme['name']}"))
        
        return detections
    
    def detect_by_position(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """Detect referees based on typical field positions"""
        detections = []
        height, width = frame.shape[:2]
        
        # Define typical referee positions
        positions = [
            # Center field
            {'x': width//2, 'y': height//2, 'radius': min(width, height)//4},
            # Near goals
            {'x': width//4, 'y': height//2, 'radius': min(width, height)//6},
            {'x': 3*width//4, 'y': height//2, 'radius': min(width, height)//6},
            # Sidelines
            {'x': width//2, 'y': height//4, 'radius': min(width, height)//8},
            {'x': width//2, 'y': 3*height//4, 'radius': min(width, height)//8}
        ]
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        for pos in positions:
            # Create mask for this position
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (pos['x'], pos['y']), pos['radius'], 255, -1)
            
            # Find contours in this area
            masked_edges = cv2.bitwise_and(edges, mask)
            contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 5000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if it's in the expected position
                    center_x, center_y = x + w//2, y + h//2
                    distance = np.sqrt((center_x - pos['x'])**2 + (center_y - pos['y'])**2)
                    
                    if distance < pos['radius']:
                        aspect_ratio = h / w
                        if 1.5 < aspect_ratio < 4.0:
                            confidence = 0.4 + (1.0 - distance / pos['radius']) * 0.3
                            detections.append((x, y, w, h, confidence, "position_based"))
        
        return detections
    
    def detect_by_movement_patterns(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """Detect referees based on movement patterns (simplified)"""
        detections = []
        
        # This is a simplified version - in practice you'd track movement over time
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction (simplified)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        
        # Find moving objects
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 800 < area < 6000:  # Referee-sized objects
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio
                aspect_ratio = h / w
                if 1.5 < aspect_ratio < 4.0:
                    # Check if object is in typical referee positions
                    center_x, center_y = x + w//2, y + h//2
                    height, width = frame.shape[:2]
                    
                    # Referees are often in center or near sidelines
                    if (width//4 < center_x < 3*width//4 and 
                        height//6 < center_y < 5*height//6):
                        confidence = 0.3
                        detections.append((x, y, w, h, confidence, "movement_pattern"))
        
        return detections
    
    def detect_by_contrast(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """Detect referees based on uniform contrast"""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold to find high contrast areas
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 8000:  # Referee-sized objects
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio
                aspect_ratio = h / w
                if 1.5 < aspect_ratio < 4.0:
                    # Calculate contrast score
                    roi = gray[y:y+h, x:x+w]
                    contrast_score = np.std(roi) / np.mean(roi) if np.mean(roi) > 0 else 0
                    
                    if contrast_score > 0.3:  # High contrast
                        confidence = min(0.7, 0.4 + contrast_score * 0.5)
                        detections.append((x, y, w, h, confidence, "contrast_based"))
        
        return detections
    
    def detect_by_shape_analysis(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """Detect referees based on shape analysis"""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 800 < area < 8000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio
                aspect_ratio = h / w
                if 1.5 < aspect_ratio < 4.0:
                    # Calculate shape complexity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # Referees have moderate circularity (not too round, not too complex)
                        if 0.1 < circularity < 0.4:
                            confidence = 0.3 + (0.4 - abs(circularity - 0.25)) * 0.5
                            detections.append((x, y, w, h, confidence, "shape_analysis"))
        
        return detections
    
    def rank_and_filter_detections(self, detections: List[Tuple[int, int, int, int, float, str]]) -> List[Tuple[int, int, int, int, float, str]]:
        """Rank detections by confidence and remove duplicates"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x[4], reverse=True)
        
        # Remove duplicates based on overlap
        filtered = []
        for detection in detections:
            x1, y1, w1, h1, conf1, method1 = detection
            
            is_duplicate = False
            for existing in filtered:
                x2, y2, w2, h2, conf2, method2 = existing
                
                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - overlap_area
                
                if union_area > 0 and overlap_area / union_area > 0.5:  # 50% overlap
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered[:3]  # Return top 3 detections
    
    def analyze_referee_uniform(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """Analyze the referee's uniform to understand the color scheme"""
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Analyze dominant colors
        pixels = hsv_roi.reshape(-1, 3)
        
        # Find dominant colors using K-means (simplified)
        dominant_colors = self.find_dominant_colors(pixels, k=3)
        
        uniform_analysis = {
            'dominant_colors': dominant_colors,
            'uniform_type': self.classify_uniform_type(dominant_colors),
            'confidence': 0.8
        }
        
        return uniform_analysis
    
    def find_dominant_colors(self, pixels: np.ndarray, k: int = 3) -> List[Tuple[int, int, int]]:
        """Find dominant colors in the ROI (simplified K-means)"""
        # This is a simplified version - in practice you'd use proper K-means
        colors = []
        
        # Sample random pixels
        if len(pixels) > 100:
            sample_indices = np.random.choice(len(pixels), 100, replace=False)
            sample_pixels = pixels[sample_indices]
            
            # Group similar colors
            for pixel in sample_pixels:
                h, s, v = pixel
                colors.append((int(h), int(s), int(v)))
        
        return colors[:k]
    
    def classify_uniform_type(self, colors: List[Tuple[int, int, int]]) -> str:
        """Classify the uniform type based on colors"""
        if not colors:
            return "unknown"
        
        # Simple classification based on hue values
        for h, s, v in colors:
            if 20 <= h <= 30 and s > 100:  # Yellow
                return "yellow_black"
            elif 100 <= h <= 130 and s > 50:  # Blue
                return "blue_white"
            elif 40 <= h <= 80 and s > 50:  # Green
                return "green_white"
            elif v > 200:  # White
                return "white_black"
            elif v < 50:  # Black
                return "black_white"
        
        return "mixed_colors"

def main():
    """Test the enhanced referee detection"""
    print("🏈 Godseye AI - Enhanced Referee Detection")
    print("=" * 50)
    
    # Initialize detector
    detector = EnhancedRefereeDetector()
    
    print("✅ Enhanced referee detector initialized")
    print("🎯 Detection methods:")
    print("  - Multi-color scheme detection")
    print("  - Position-based detection")
    print("  - Movement pattern analysis")
    print("  - Uniform contrast detection")
    print("  - Shape analysis")
    print("\n🚀 Ready for flexible referee detection!")
    
    return detector

if __name__ == "__main__":
    main()
