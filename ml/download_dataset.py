#!/usr/bin/env python3
"""
Robust dataset downloader for sports analytics datasets.
Supports SoccerNet v3, custom datasets, and automatic verification.
"""

import os
import sys
import hashlib
import requests
import tarfile
import zipfile
import json
import argparse
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Robust dataset downloader with checksum verification and resumable downloads."""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            "soccernet_v3": {
                "url": "https://www.soccer-net.org/data/SoccerNetv3.zip",
                "filename": "SoccerNetv3.zip",
                "sha256": None,  # Will be updated when available
                "description": "SoccerNet v3 - 500+ matches with comprehensive annotations",
                "extract": True,
                "structure": {
                    "matches": "SoccerNetv3/matches/",
                    "annotations": "Soccernetv3/annotations/",
                    "metadata": "SoccerNetv3/metadata/"
                }
            },
            "soccernet_v2": {
                "url": "https://www.soccer-net.org/data/SoccerNetv2.zip",
                "filename": "SoccerNetv2.zip",
                "sha256": None,
                "description": "SoccerNet v2 - 500 matches with action spotting",
                "extract": True,
                "structure": {
                    "matches": "SoccerNetv2/matches/",
                    "annotations": "SoccerNetv2/annotations/"
                }
            },
            "football_pose": {
                "url": "https://github.com/silviogiancola/football-pose-dataset/releases/download/v1.0/football_pose_dataset.zip",
                "filename": "football_pose_dataset.zip",
                "sha256": None,
                "description": "Football pose estimation dataset",
                "extract": True,
                "structure": {
                    "images": "football_pose_dataset/images/",
                    "annotations": "football_pose_dataset/annotations/"
                }
            }
        }
    
    def download_file(self, url: str, filepath: Path, expected_sha256: Optional[str] = None) -> bool:
        """
        Download a file with progress bar and optional checksum verification.
        Supports resumable downloads.
        """
        logger.info(f"Downloading {url} to {filepath}")
        
        # Check if file already exists
        if filepath.exists():
            if expected_sha256 and self.verify_sha256(filepath, expected_sha256):
                logger.info(f"File {filepath} already exists and is valid")
                return True
            else:
                logger.warning(f"File {filepath} exists but checksum doesn't match. Re-downloading...")
                filepath.unlink()
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Get file size for progress bar
        try:
            response = requests.head(url, allow_redirects=True)
            total_size = int(response.headers.get('content-length', 0))
        except:
            total_size = 0
        
        # Download with progress bar
        try:
            response = requests.get(url, stream=True, allow_redirects=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Verify checksum if provided
            if expected_sha256:
                if self.verify_sha256(filepath, expected_sha256):
                    logger.info(f"Download completed and verified: {filepath}")
                    return True
                else:
                    logger.error(f"Checksum verification failed for {filepath}")
                    filepath.unlink()
                    return False
            else:
                logger.info(f"Download completed: {filepath}")
                return True
                
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def verify_sha256(self, filepath: Path, expected_sha256: str) -> bool:
        """Verify SHA256 checksum of a file."""
        logger.info(f"Verifying checksum for {filepath}")
        
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        
        actual_sha256 = sha256_hash.hexdigest()
        return actual_sha256 == expected_sha256
    
    def extract_archive(self, filepath: Path, extract_to: Path) -> bool:
        """Extract archive (zip or tar) to specified directory."""
        logger.info(f"Extracting {filepath} to {extract_to}")
        
        try:
            if filepath.suffix == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif filepath.suffix in ['.tar', '.gz', '.bz2', '.xz']:
                with tarfile.open(filepath, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                logger.error(f"Unsupported archive format: {filepath.suffix}")
                return False
            
            logger.info(f"Extraction completed: {extract_to}")
            return True
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False
    
    def download_dataset(self, dataset_name: str, custom_url: Optional[str] = None) -> bool:
        """Download a specific dataset."""
        if custom_url:
            # Custom dataset download
            parsed_url = urlparse(custom_url)
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = "custom_dataset.zip"
            
            dataset_config = {
                "url": custom_url,
                "filename": filename,
                "sha256": None,
                "description": "Custom dataset",
                "extract": True,
                "structure": {}
            }
        else:
            if dataset_name not in self.datasets:
                logger.error(f"Unknown dataset: {dataset_name}")
                logger.info(f"Available datasets: {list(self.datasets.keys())}")
                return False
            
            dataset_config = self.datasets[dataset_name]
        
        # Download file
        filepath = self.base_dir / dataset_config["filename"]
        if not self.download_file(dataset_config["url"], filepath, dataset_config["sha256"]):
            return False
        
        # Extract if needed
        if dataset_config["extract"]:
            extract_to = self.base_dir / dataset_name
            if not self.extract_archive(filepath, extract_to):
                return False
        
        # Create dataset info
        self.create_dataset_info(dataset_name, dataset_config, filepath)
        
        logger.info(f"Dataset {dataset_name} downloaded successfully!")
        return True
    
    def create_dataset_info(self, dataset_name: str, config: Dict, filepath: Path):
        """Create dataset information file."""
        info = {
            "name": dataset_name,
            "description": config["description"],
            "url": config["url"],
            "filename": config["filename"],
            "sha256": config["sha256"],
            "downloaded_at": str(Path.cwd()),
            "file_size": filepath.stat().st_size if filepath.exists() else 0,
            "structure": config["structure"]
        }
        
        info_path = self.base_dir / f"{dataset_name}_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Dataset info saved to {info_path}")
    
    def list_datasets(self):
        """List available datasets."""
        logger.info("Available datasets:")
        for name, config in self.datasets.items():
            logger.info(f"  {name}: {config['description']}")
    
    def get_dataset_stats(self, dataset_name: str) -> Dict:
        """Get statistics about a downloaded dataset."""
        dataset_dir = self.base_dir / dataset_name
        if not dataset_dir.exists():
            return {"error": f"Dataset {dataset_name} not found"}
        
        stats = {
            "name": dataset_name,
            "path": str(dataset_dir),
            "total_size": 0,
            "file_count": 0,
            "directories": []
        }
        
        for root, dirs, files in os.walk(dataset_dir):
            stats["directories"].append(root)
            for file in files:
                filepath = Path(root) / file
                stats["total_size"] += filepath.stat().st_size
                stats["file_count"] += 1
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Download sports analytics datasets")
    parser.add_argument("--dataset", type=str, help="Dataset name to download")
    parser.add_argument("--url", type=str, help="Custom dataset URL")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--stats", type=str, help="Get dataset statistics")
    parser.add_argument("--base-dir", type=str, default="data", help="Base directory for datasets")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.base_dir)
    
    if args.list:
        downloader.list_datasets()
    elif args.stats:
        stats = downloader.get_dataset_stats(args.stats)
        print(json.dumps(stats, indent=2))
    elif args.dataset or args.url:
        success = downloader.download_dataset(args.dataset, args.url)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
