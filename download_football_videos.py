#!/usr/bin/env python3
"""
===============================================================================
GODSEYE AI - FOOTBALL VIDEO DOWNLOADER
===============================================================================

This script automatically downloads football videos from various public sources
without requiring NDAs. It focuses on getting real match footage for training.

Author: Victor
Date: 2025
Version: 1.0.0

SOURCES:
- YouTube (match highlights, full matches)
- Public football video repositories
- Open sports datasets

USAGE:
    python download_football_videos.py
"""

import os
import sys
import subprocess
import requests
import json
from pathlib import Path
import logging
import time
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FootballVideoDownloader:
    """Downloads football videos from public sources"""
    
    def __init__(self, output_dir: str = "data/football_videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # YouTube video IDs for football matches (publicly available)
        self.youtube_matches = [
            # Premier League matches
            "dQw4w9WgXcQ",  # Example - replace with real match IDs
            "jNQXAC9IVRw",  # Example - replace with real match IDs
            
            # La Liga matches  
            "M7lc1UVf-VE",  # Example - replace with real match IDs
            "fJ9rUzIMcZQ",  # Example - replace with real match IDs
            
            # Champions League
            "9bZkp7q19f0",  # Example - replace with real match IDs
        ]
        
        # Search terms for finding football videos
        self.search_terms = [
            "Premier League full match 2024",
            "La Liga full match 2024", 
            "Champions League full match 2024",
            "Bundesliga full match 2024",
            "Serie A full match 2024",
            "football match highlights 90 minutes",
            "soccer match full game 2024"
        ]
    
    def install_youtube_dl(self):
        """Install youtube-dl for downloading videos"""
        try:
            import yt_dlp
            logger.info("yt-dlp already installed")
            return True
        except ImportError:
            logger.info("Installing yt-dlp...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"], check=True)
                logger.info("yt-dlp installed successfully")
                return True
            except subprocess.CalledProcessError:
                logger.error("Failed to install yt-dlp")
                return False
    
    def download_from_youtube(self, video_id: str, output_path: str) -> bool:
        """Download a specific YouTube video"""
        try:
            import yt_dlp
            
            ydl_opts = {
                'format': 'best[height<=720]',  # Limit to 720p to save space
                'outtmpl': str(output_path),
                'noplaylist': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            
            return True
        except Exception as e:
            logger.error(f"Error downloading {video_id}: {e}")
            return False
    
    def search_and_download_youtube(self, search_term: str, max_results: int = 5) -> List[str]:
        """Search YouTube for football videos and download them"""
        try:
            import yt_dlp
            
            downloaded_videos = []
            
            ydl_opts = {
                'format': 'best[height<=720]',
                'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
                'noplaylist': True,
                'max_downloads': max_results,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Search for videos
                search_results = ydl.extract_info(
                    f"ytsearch{max_results}:{search_term}",
                    download=False
                )
                
                if 'entries' in search_results:
                    for entry in search_results['entries']:
                        if entry:
                            video_url = entry['webpage_url']
                            video_title = entry.get('title', 'Unknown')
                            
                            # Check if it's a football-related video
                            if self.is_football_video(video_title):
                                logger.info(f"Downloading: {video_title}")
                                try:
                                    ydl.download([video_url])
                                    downloaded_videos.append(video_title)
                                    time.sleep(2)  # Be respectful to YouTube
                                except Exception as e:
                                    logger.warning(f"Failed to download {video_title}: {e}")
            
            return downloaded_videos
            
        except Exception as e:
            logger.error(f"Error searching YouTube: {e}")
            return []
    
    def is_football_video(self, title: str) -> bool:
        """Check if video title suggests it's a football match"""
        football_keywords = [
            'football', 'soccer', 'match', 'game', 'premier league', 'la liga',
            'champions league', 'bundesliga', 'serie a', 'full match', 'highlights',
            'vs', 'v', 'versus', 'real madrid', 'barcelona', 'manchester', 'liverpool',
            'arsenal', 'chelsea', 'tottenham', 'bayern', 'juventus', 'milan'
        ]
        
        title_lower = title.lower()
        return any(keyword in title_lower for keyword in football_keywords)
    
    def download_sample_videos(self):
        """Download a curated set of football videos"""
        logger.info("Downloading sample football videos...")
        
        # Sample video URLs (these are examples - replace with real URLs)
        sample_videos = [
            {
                'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
                'title': 'Sample Match 1'
            },
            {
                'url': 'https://www.youtube.com/watch?v=jNQXAC9IVRw', 
                'title': 'Sample Match 2'
            }
        ]
        
        downloaded = []
        for video in sample_videos:
            try:
                import yt_dlp
                
                ydl_opts = {
                    'format': 'best[height<=720]',
                    'outtmpl': str(self.output_dir / f"{video['title']}.%(ext)s"),
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video['url']])
                    downloaded.append(video['title'])
                    logger.info(f"Downloaded: {video['title']}")
                    
            except Exception as e:
                logger.error(f"Failed to download {video['title']}: {e}")
        
        return downloaded
    
    def create_training_dataset(self):
        """Create a training dataset from downloaded videos"""
        logger.info("Creating training dataset...")
        
        # Create dataset structure
        dataset_dir = self.output_dir / "training_dataset"
        (dataset_dir / "videos").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "annotations").mkdir(parents=True, exist_ok=True)
        
        # Move videos to dataset directory
        video_files = list(self.output_dir.glob("*.mp4")) + list(self.output_dir.glob("*.mkv"))
        
        for i, video_file in enumerate(video_files):
            new_name = f"match_{i+1:03d}.mp4"
            new_path = dataset_dir / "videos" / new_name
            video_file.rename(new_path)
            logger.info(f"Moved {video_file.name} to {new_name}")
        
        # Create basic annotations (placeholder)
        for i in range(len(video_files)):
            annotation = {
                'video_id': f"match_{i+1:03d}",
                'duration': 90,  # Assume 90 minutes
                'events': [],
                'annotations': []
            }
            
            ann_file = dataset_dir / "annotations" / f"match_{i+1:03d}.json"
            with open(ann_file, 'w') as f:
                json.dump(annotation, f, indent=2)
        
        logger.info(f"Training dataset created with {len(video_files)} videos")
        return dataset_dir
    
    def run_download_pipeline(self):
        """Run the complete download pipeline"""
        logger.info("Starting football video download pipeline...")
        
        # Install required tools
        if not self.install_youtube_dl():
            logger.error("Cannot proceed without yt-dlp")
            return False
        
        # Download videos using search terms
        total_downloaded = 0
        
        for search_term in self.search_terms[:3]:  # Limit to first 3 terms
            logger.info(f"Searching for: {search_term}")
            downloaded = self.search_and_download_youtube(search_term, max_results=2)
            total_downloaded += len(downloaded)
            
            if total_downloaded >= 10:  # Stop when we have enough videos
                break
        
        # If we didn't get enough videos, try sample downloads
        if total_downloaded < 5:
            logger.info("Downloading sample videos...")
            sample_downloaded = self.download_sample_videos()
            total_downloaded += len(sample_downloaded)
        
        logger.info(f"Total videos downloaded: {total_downloaded}")
        
        if total_downloaded > 0:
            # Create training dataset
            dataset_dir = self.create_training_dataset()
            logger.info(f"Training dataset ready at: {dataset_dir}")
            return True
        else:
            logger.error("No videos downloaded")
            return False

def main():
    """Main function"""
    logger.info("Godseye AI - Football Video Downloader")
    logger.info("=" * 50)
    
    downloader = FootballVideoDownloader()
    success = downloader.run_download_pipeline()
    
    if success:
        logger.info("=" * 50)
        logger.info("SUCCESS! Football videos downloaded")
        logger.info("Ready for training pipeline!")
        logger.info("=" * 50)
    else:
        logger.error("Failed to download football videos")
        logger.info("Please check the error messages above")

if __name__ == "__main__":
    main()

