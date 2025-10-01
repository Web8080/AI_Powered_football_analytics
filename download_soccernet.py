#!/usr/bin/env python3
"""
===============================================================================
GODSEYE AI - SOCCERNET DATASET DOWNLOADER
===============================================================================

This script downloads the real SoccerNet dataset using the official SoccerNet package.
It will download the labels first, then prompt for NDA password to download videos.

Author: Victor
Date: 2025
Version: 1.0.0

USAGE:
    python download_soccernet.py
"""

import os
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_soccernet_dataset():
    """Download the real SoccerNet dataset"""
    logger.info("🌐 Starting SoccerNet dataset download...")
    
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
        
        # Set up local directory
        local_directory = "/Users/user/Football_analytics/data/SoccerNet"
        logger.info(f"📁 Local directory: {local_directory}")
        
        # Initialize the downloader
        downloader = SoccerNetDownloader(LocalDirectory=local_directory)
        logger.info("✅ SoccerNet downloader initialized")
        
        # Step 1: Download labels (no NDA required)
        logger.info("📥 Step 1: Downloading labels...")
        downloader.downloadGames(
            files=["Labels-v2.json"], 
            split=["train", "valid", "test"]
        )
        logger.info("✅ Labels downloaded successfully!")
        
        # Step 2: Download videos (requires NDA password)
        logger.info("📥 Step 2: Downloading videos...")
        logger.info("🔐 This requires an NDA password from SoccerNet")
        logger.info("📝 Please visit: https://www.soccer-net.org/data")
        logger.info("📝 Complete the NDA form to get the password")
        
        # Prompt for NDA password
        nda_password = input("Enter your NDA password for SoccerNet videos (or press Enter to skip videos): ").strip()
        
        if nda_password:
            downloader.password = nda_password
            logger.info("🎥 Downloading video files...")
            downloader.downloadGames(
                files=["1_224p.mkv", "2_224p.mkv"], 
                split=["train", "valid", "test"]
            )
            logger.info("✅ Videos downloaded successfully!")
        else:
            logger.info("⏭️ Skipping video download (no password provided)")
        
        # Check what was downloaded
        check_downloaded_data(local_directory)
        
        logger.info("🎉 SoccerNet dataset download completed!")
        return local_directory
        
    except ImportError as e:
        logger.error(f"❌ Error importing SoccerNet: {e}")
        logger.error("Please install SoccerNet: pip install SoccerNet")
        return None
    except Exception as e:
        logger.error(f"❌ Error downloading SoccerNet dataset: {e}")
        return None

def check_downloaded_data(directory):
    """Check what data was downloaded"""
    logger.info("🔍 Checking downloaded data...")
    
    data_path = Path(directory)
    if not data_path.exists():
        logger.error(f"❌ Directory not found: {directory}")
        return
    
    # Check for labels
    labels_found = 0
    for split in ["train", "valid", "test"]:
        split_path = data_path / split
        if split_path.exists():
            label_files = list(split_path.glob("**/Labels-v2.json"))
            labels_found += len(label_files)
            logger.info(f"📊 {split}: {len(label_files)} label files found")
    
    # Check for videos
    videos_found = 0
    for split in ["train", "valid", "test"]:
        split_path = data_path / split
        if split_path.exists():
            video_files = list(split_path.glob("**/*.mkv"))
            videos_found += len(video_files)
            logger.info(f"🎥 {split}: {len(video_files)} video files found")
    
    logger.info(f"📈 Total labels: {labels_found}")
    logger.info(f"📈 Total videos: {videos_found}")
    
    if labels_found > 0:
        logger.info("✅ SoccerNet dataset is ready for training!")
    else:
        logger.warning("⚠️ No labels found. Please check the download.")

def main():
    """Main function"""
    logger.info("🚀 Godseye AI - SoccerNet Dataset Downloader")
    logger.info("=" * 60)
    
    # Download the dataset
    dataset_path = download_soccernet_dataset()
    
    if dataset_path:
        logger.info("=" * 60)
        logger.info("🎉 SUCCESS! SoccerNet dataset downloaded")
        logger.info(f"📁 Dataset location: {dataset_path}")
        logger.info("🚀 Ready to proceed with training!")
        logger.info("=" * 60)
    else:
        logger.error("❌ Failed to download SoccerNet dataset")
        logger.info("📝 Please check the error messages above")

if __name__ == "__main__":
    main()

