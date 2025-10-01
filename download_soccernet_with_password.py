#!/usr/bin/env python3
"""
===============================================================================
GODSEYE AI - SOCCERNET DATASET DOWNLOADER WITH PASSWORD
===============================================================================

This script downloads the real SoccerNet dataset with automatic password handling.
Set your NDA password in the script and it will automatically download videos.

Author: Victor
Date: 2025
Version: 1.0.0

USAGE:
    1. Get NDA password from https://www.soccer-net.org/data
    2. Set the password in this script
    3. Run: python download_soccernet_with_password.py
"""

import os
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - SET YOUR NDA PASSWORD HERE
# =============================================================================
NDA_PASSWORD = "s0cc3rn3t"  # Set your NDA password here after getting it from SoccerNet

def download_soccernet_with_password():
    """Download SoccerNet dataset with automatic password handling"""
    logger.info("Starting SoccerNet dataset download with password...")
    
    if not NDA_PASSWORD:
        logger.error("NDA_PASSWORD not set in script!")
        logger.info("Please:")
        logger.info("1. Visit https://www.soccer-net.org/data")
        logger.info("2. Complete the NDA form")
        logger.info("3. Get your password via email")
        logger.info("4. Set NDA_PASSWORD in this script")
        return False
    
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
        
        # Set up local directory
        local_directory = "/Users/user/Football_analytics/data/SoccerNet"
        logger.info(f"Local directory: {local_directory}")
        
        # Initialize the downloader
        downloader = SoccerNetDownloader(LocalDirectory=local_directory)
        downloader.password = NDA_PASSWORD  # Set password automatically
        logger.info("SoccerNet downloader initialized with password")
        
        # Step 1: Download labels (already done, but ensure we have them)
        logger.info("Step 1: Ensuring labels are downloaded...")
        downloader.downloadGames(
            files=["Labels-v2.json"], 
            split=["train", "valid", "test"]
        )
        logger.info("Labels confirmed/downloaded successfully!")
        
        # Step 2: Download videos with automatic password
        logger.info("Step 2: Downloading videos with automatic password...")
        logger.info("This will download ~500GB of video data...")
        logger.info("Estimated time: 10+ hours depending on internet speed")
        
        # Download videos for training and validation (skip test to save space)
        downloader.downloadGames(
            files=["1_224p.mkv", "2_224p.mkv"], 
            split=["train", "valid"]  # Skip test and challenge to save space
        )
        logger.info("Videos downloaded successfully!")
        
        # Check what was downloaded
        check_downloaded_data(local_directory)
        
        logger.info("SoccerNet dataset download completed!")
        return True
        
    except ImportError as e:
        logger.error(f"Error importing SoccerNet: {e}")
        logger.error("Please install SoccerNet: pip install SoccerNet")
        return False
    except Exception as e:
        logger.error(f"Error downloading SoccerNet dataset: {e}")
        return False

def check_downloaded_data(directory):
    """Check what data was downloaded"""
    logger.info("Checking downloaded data...")
    
    data_path = Path(directory)
    if not data_path.exists():
        logger.error(f"Directory not found: {directory}")
        return
    
    # Check for labels
    labels_found = 0
    for split in ["train", "valid", "test"]:
        split_path = data_path / split
        if split_path.exists():
            label_files = list(split_path.glob("**/Labels-v2.json"))
            labels_found += len(label_files)
            logger.info(f"{split}: {len(label_files)} label files found")
    
    # Check for videos
    videos_found = 0
    for split in ["train", "valid", "test"]:
        split_path = data_path / split
        if split_path.exists():
            video_files = list(split_path.glob("**/*.mkv"))
            videos_found += len(video_files)
            logger.info(f"{split}: {len(video_files)} video files found")
    
    logger.info(f"Total labels: {labels_found}")
    logger.info(f"Total videos: {videos_found}")
    
    if labels_found > 0 and videos_found > 0:
        logger.info("SoccerNet dataset is ready for training!")
    elif labels_found > 0:
        logger.info("Labels ready, but videos still downloading...")
    else:
        logger.warning("No data found. Please check the download.")

def main():
    """Main function"""
    logger.info("Godseye AI - SoccerNet Dataset Downloader with Password")
    logger.info("=" * 60)
    
    # Download the dataset
    success = download_soccernet_with_password()
    
    if success:
        logger.info("=" * 60)
        logger.info("SUCCESS! SoccerNet dataset downloaded")
        logger.info("Ready to proceed with training!")
        logger.info("=" * 60)
    else:
        logger.error("Failed to download SoccerNet dataset")
        logger.info("Please check the error messages above")

if __name__ == "__main__":
    main()

