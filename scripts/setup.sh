#!/usr/bin/env bash
# SpatialCrew quick setup script
set -euo pipefail

echo "=== SpatialCrew Setup ==="

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $python_version"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example (edit it with your settings)"
else
    echo ".env already exists, skipping"
fi

# Create output directories
mkdir -p outputs logs

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your LLM and data settings"
echo "  2. (Optional) Start GPU server: python entrypoints/launch_gpu_server.py"
echo "  3. Run evaluation: python entrypoints/run.py --help"
