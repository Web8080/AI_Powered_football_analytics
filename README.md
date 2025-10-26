# Godseye AI: Professional Football Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://reactjs.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00d4aa.svg)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

Godseye AI is a comprehensive computer vision platform designed for professional football analytics, combining state-of-the-art object detection, multi-object tracking, and real-time video analysis capabilities.

## Quick Start

### Prerequisites

- Python 3.8

+

- Node.js 16

+ (for frontend development)

- 8GB

+ RAM (for video processing)

### Installation

```bash

# Clone the repository

git clone https://github.com/Web8080/AI_Powered_football_analytics.git
cd AI_Powered_football_analytics

# Install Python dependencies

pip install -r requirements.txt

# Install frontend dependencies

cd frontend
npm install
cd ..

```

### Running the Application

```bash

# Start the API server

python simple_inference_api.py

# Start the frontend (in a new terminal)

cd frontend
npm run dev

```

Access the application at: http://localhost:3001

## Documentation

### Core Documentation

- **[Architecture](docs/ARCHITECTURE.md)*

*
- System architecture and design

- **[Methodology & Deployment](docs/METHODOLOGY_AND_DEPLOYMENT.md)*

*
- Technical methodology and deployment guide

- **[Security Configuration](docs/SECURITY_CONFIG.md)*

*
- Security best practices and configuration

### Development & Testing

- **[Frontend Setup](docs/FRONTEND_SETUP.md)*

*
- Frontend development guide

- **[Testing Guide](docs/TESTING.md)*

*
- Testing procedures and guidelines

- **[System Status](docs/SYSTEM_STATUS.md)*

*
- Current system status and monitoring

### Features & Improvements

- **[Comprehensive Features](docs/COMPREHENSIVE_FEATURES_SUMMARY.md)*

*
- Complete feature overview

- **[Improvements Summary](docs/IMPROVEMENTS_SUMMARY.md)*

*
- Recent improvements and updates

## Key Features

- **Real-time Object Detection*

*
- Detect players, referees, and ball with high accuracy

- **Multi-object Tracking*

*
- Track players and ball throughout the match

- **Team Classification*

*
- Automatically classify players into teams

- **Event Detection*

*
- Identify goals, fouls, and other match events

- **Web Dashboard*

*
- Modern React-based interface for analysis

- **API Integration*

*
- RESTful API for third-party integrations

## Training Options

### Quick Training (Google Colab)

```bash
python google_colab_training.py

```

- Duration: < 1 hour

- Perfect for testing and prototyping

### Production Training

```bash
python robust_local_training.py

```

- Duration: 24 hours max

- Production-ready models

## API Usage

### Upload and Analyze Video

```python
import requests

# Upload video

with open('match.mp4', 'rb') as f:
response = requests.post('http://localhost:8001/upload-video', files={'file': f})
job_id = response.json()['job_id']

# Check progress

progress = requests.get(f'http://localhost:8001/progress/{job_id}').json()
print(f"Progress: {progress['progress']}%")

# Get results

results = requests.get(f'http://localhost:8001/analysis/{job_id}').json()
```

## Technology Stack

- **Backend** : FastAPI, Python 3.8

+

- ** Frontend** : React 18, TypeScript, Tailwind CSS

- ** ML/AI** : YOLOv8, OpenCV, PyTorch

- ** Database** : PostgreSQL, MongoDB, Redis

- ** Deployment*
* : Docker, Docker Compose

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License
- see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation** : [docs.godseye-ai.com](https://docs.godseye-ai.com)

- ** Issues** : [GitHub Issues](https://github.com/Web8080/AI_Powered_football_analytics/issues)

- ** Email*
* : support@godseye-ai.com

--
-

**Built with passion for the football community*

*

*Professional-grade AI analytics for the beautiful game*