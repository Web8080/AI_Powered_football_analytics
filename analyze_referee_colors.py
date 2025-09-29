#!/usr/bin/env python3
"""
================================================================================
GODSEYE AI - REFEREE COLOR ANALYSIS
================================================================================

Author: Victor Ibhafidon
Date: January 28, 2025
Version: 1.0.0

DESCRIPTION:
Analyze the actual referee colors in the BAY_BMG video to understand
what uniform the referee is wearing.

USAGE:
    python analyze_referee_colors.py
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_referee_colors(video_path: str, sample_frames: int = 20):
    """Analyze referee colors in the video"""
    print(f"🎥 Analyzing referee colors in {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📊 Video info: {total_frames} frames, {fps} FPS")
    
    # Sample frames from different parts of the video
    frame_indices = np.linspace(0, total_frames-1, sample_frames, dtype=int)
    
    all_colors = []
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        print(f"🔍 Analyzing frame {i+1}/{sample_frames} (frame {frame_idx})")
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Focus on center area where referees are likely to be
        height, width = frame.shape[:2]
        center_roi = frame[height//4:3*height//4, width//4:3*width//4]
        hsv_roi = hsv[height//4:3*height//4, width//4:3*width//4]
        
        # Find dominant colors in the center area
        dominant_colors = find_dominant_colors(hsv_roi, k=5)
        all_colors.extend(dominant_colors)
        
        # Save a sample frame for visual inspection
        if i == 0:
            cv2.imwrite("referee_analysis_sample.jpg", center_roi)
            print("💾 Saved sample frame: referee_analysis_sample.jpg")
    
    cap.release()
    
    # Analyze the collected colors
    analyze_color_distribution(all_colors)
    
    return all_colors

def find_dominant_colors(hsv_image: np.ndarray, k: int = 5) -> list:
    """Find dominant colors in HSV image"""
    # Reshape image to be a list of pixels
    pixels = hsv_image.reshape(-1, 3)
    
    # Sample pixels for efficiency
    if len(pixels) > 1000:
        sample_indices = np.random.choice(len(pixels), 1000, replace=False)
        pixels = pixels[sample_indices]
    
    # Simple color clustering (simplified K-means)
    colors = []
    
    # Group similar colors
    for pixel in pixels:
        h, s, v = pixel
        # Only consider saturated colors (not grayscale)
        if s > 50 and v > 50:
            colors.append((int(h), int(s), int(v)))
    
    # Count color frequencies
    color_counts = {}
    for color in colors:
        # Group similar hues
        h, s, v = color
        hue_bucket = (h // 10) * 10  # Group by 10-degree hue buckets
        key = (hue_bucket, s, v)
        
        if key in color_counts:
            color_counts[key] += 1
        else:
            color_counts[key] = 1
    
    # Get top colors
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    return [color[0] for color in sorted_colors[:k]]

def analyze_color_distribution(colors: list):
    """Analyze the distribution of colors found"""
    print("\n🎨 Color Analysis Results:")
    print("=" * 40)
    
    if not colors:
        print("❌ No colors found")
        return
    
    # Count colors by hue range
    hue_ranges = {
        'Red': (0, 20),
        'Yellow': (20, 40),
        'Green': (40, 80),
        'Cyan': (80, 100),
        'Blue': (100, 140),
        'Magenta': (140, 180)
    }
    
    color_counts = {name: 0 for name in hue_ranges}
    
    for h, s, v in colors:
        for name, (min_h, max_h) in hue_ranges.items():
            if min_h <= h <= max_h:
                color_counts[name] += 1
                break
    
    print("📊 Color distribution:")
    for name, count in color_counts.items():
        if count > 0:
            percentage = (count / len(colors)) * 100
            print(f"  {name}: {count} colors ({percentage:.1f}%)")
    
    print(f"\n🔍 Most common colors:")
    # Show top 10 colors
    color_freq = {}
    for color in colors:
        color_freq[color] = color_freq.get(color, 0) + 1
    
    sorted_colors = sorted(color_freq.items(), key=lambda x: x[1], reverse=True)
    
    for i, (color, freq) in enumerate(sorted_colors[:10]):
        h, s, v = color
        print(f"  {i+1}. HSV({h}, {s}, {v}) - {freq} occurrences")
    
    # Suggest referee detection parameters
    suggest_referee_params(sorted_colors)

def suggest_referee_params(sorted_colors: list):
    """Suggest referee detection parameters based on found colors"""
    print(f"\n💡 Suggested Referee Detection Parameters:")
    print("=" * 50)
    
    if not sorted_colors:
        print("❌ No colors to analyze")
        return
    
    # Get the most common colors
    top_colors = [color[0] for color in sorted_colors[:3]]
    
    print("🎯 Recommended color ranges for referee detection:")
    
    for i, (h, s, v) in enumerate(top_colors):
        # Create a range around the detected color
        h_range = 10
        s_range = 50
        v_range = 50
        
        h_min = max(0, h - h_range)
        h_max = min(180, h + h_range)
        s_min = max(0, s - s_range)
        s_max = min(255, s + s_range)
        v_min = max(0, v - v_range)
        v_max = min(255, v + v_range)
        
        print(f"  Color {i+1}: HSV({h}, {s}, {v})")
        print(f"    Range: [{h_min}, {s_min}, {v_min}] to [{h_max}, {s_max}, {v_max}]")
        print(f"    OpenCV: cv2.inRange(hsv, np.array([{h_min}, {s_min}, {v_min}]), np.array([{h_max}, {s_max}, {v_max}]))")
        print()

def main():
    """Main function"""
    print("🏈 Godseye AI - Referee Color Analysis")
    print("=" * 45)
    
    video_path = "BAY_BMG.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    # Analyze referee colors
    colors = analyze_referee_colors(video_path, sample_frames=30)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Check 'referee_analysis_sample.jpg' for visual reference")
    print(f"🎯 Use the suggested parameters above for better referee detection")

if __name__ == "__main__":
    main()
