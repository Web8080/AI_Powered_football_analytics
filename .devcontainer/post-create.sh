#!/bin/bash

# Godseye Development Environment Setup Script

echo "🚀 Setting up Godseye development environment..."

# Install additional Python packages for development
pip install --upgrade pip
pip install ipython jupyter notebook jupyterlab
pip install pytest pytest-django pytest-cov
pip install black isort flake8 mypy
pip install pre-commit

# Install Node.js and npm for frontend development
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt-get install -y nodejs

# Install additional system packages
apt-get update
apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    python3-dev \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libdc1394-dev

# Create shared directories
mkdir -p /app/shared/{data,models,logs}
mkdir -p /app/shared/data/{raw,processed,datasets}
mkdir -p /app/shared/models/{detection,pose,action}
mkdir -p /app/shared/logs/{training,inference,api}

# Set up pre-commit hooks
cd /app
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
fi

# Initialize git if not already done
if [ ! -d ".git" ]; then
    git init
    git config user.name "Godseye Developer"
    git config user.email "developer@godseye.com"
fi

# Create useful aliases
echo 'alias ll="ls -la"' >> ~/.bashrc
echo 'alias la="ls -A"' >> ~/.bashrc
echo 'alias l="ls -CF"' >> ~/.bashrc
echo 'alias ..="cd .."' >> ~/.bashrc
echo 'alias ...="cd ../.."' >> ~/.bashrc
echo 'alias dc="docker-compose"' >> ~/.bashrc
echo 'alias dcu="docker-compose up"' >> ~/.bashrc
echo 'alias dcd="docker-compose down"' >> ~/.bashrc
echo 'alias dcb="docker-compose build"' >> ~/.bashrc
echo 'alias dcl="docker-compose logs -f"' >> ~/.bashrc
echo 'alias pm="python manage.py"' >> ~/.bashrc
echo 'alias pms="python manage.py shell"' >> ~/.bashrc
echo 'alias pmr="python manage.py runserver"' >> ~/.bashrc
echo 'alias pmm="python manage.py migrate"' >> ~/.bashrc
echo 'alias pmc="python manage.py collectstatic"' >> ~/.bashrc

# Set up Python path
echo 'export PYTHONPATH="${PYTHONPATH}:/app"' >> ~/.bashrc
echo 'export PYTHONPATH="${PYTHONPATH}:/app/ml"' >> ~/.bashrc
echo 'export PYTHONPATH="${PYTHONPATH}:/app/inference"' >> ~/.bashrc

# Create a welcome message
cat > /app/WELCOME.md << 'EOF'
# Welcome to Godseye Development Environment! 🎉

## Quick Start Commands

### Backend Development
```bash
# Run Django development server
python manage.py runserver 0.0.0.0:8000

# Run database migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Run tests
pytest

# Run Celery worker
celery -A app worker -l info
```

### Frontend Development
```bash
cd /app/frontend
npm install
npm start
```

### ML Development
```bash
cd /app/ml
python train.py --config configs/detectron2_soccernet.yaml
```

### Docker Commands
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild services
docker-compose build --no-cache
```

## Access Points
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- MLflow UI: http://localhost:5000
- MinIO Console: http://localhost:9001 (minioadmin/minioadmin)

## Useful Aliases
- `dc` - docker-compose
- `pm` - python manage.py
- `pms` - python manage.py shell
- `pmr` - python manage.py runserver
- `pmm` - python manage.py migrate

Happy coding! 🚀
EOF

echo "✅ Godseye development environment setup complete!"
echo "📖 Check WELCOME.md for quick start commands"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend: http://localhost:8000"
echo "📊 MLflow: http://localhost:5000"
echo "💾 MinIO: http://localhost:9001"
