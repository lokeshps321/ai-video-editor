#!/usr/bin/env bash
set -e

echo "=================================================="
echo "🚀 Deploying ClipMind to Azure VM (Student Plan)"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    sudo apt update
    sudo apt install -y docker.io docker-compose-plugin git curl
    sudo usermod -aG docker $USER
fi

# Check Swap memory (Ensure at least 4GB swap to handle FFmpeg OOM)
SWAP_SIZE=$(free -m | awk '/^Swap:/{print $2}')
if [ "$SWAP_SIZE" -lt 2000 ]; then
    echo "⚡ Configuring 4GB Swap Space..."
    sudo fallocate -l 4G /swapfile 2>/dev/null || sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    if ! grep -q '/swapfile' /etc/fstab; then
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    fi
    echo "✅ 4GB Swap Space Configured."
fi

# Setup production environment file if missing
if [ ! -f backend/.env.production ]; then
    if [ -f backend/.env.production.example ]; then
        echo "📝 Creating backend/.env.production from example..."
        cp backend/.env.production.example backend/.env.production
    else
        echo "⚠️ Warning: backend/.env.production not found."
    fi
fi

# Build and start services
echo "🐳 Building & starting Docker containers..."
docker compose build
docker compose up -d

echo "=================================================="
echo "🎉 ClipMind is live!"
echo "Check container status: docker compose ps"
echo "Check logs: docker compose logs -f"
echo "=================================================="
