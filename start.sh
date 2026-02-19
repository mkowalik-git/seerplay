#!/bin/bash
set -e

echo "============================================================"
echo " SeerPlay Intelligence Hub — Auto Start"
echo "============================================================"

# 1. Build the parent image
echo "🏗️  [1/3] Building seerplay-omni:latest..."
podman build -f Containerfile.parent -t seerplay-omni:latest .

# 2. Cleanup existing container
echo "🧹 [2/3] Removing old container (if any)..."
podman rm -f seerplay-v3 2>/dev/null || true

# OPTIMIZATION: Clear VM caches to ensure sufficient free RAM for Ollama
echo "🧠 [Optimization] Clearing VM caches..."
podman machine ssh "sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'" || true

# 3. Run the container
echo "🚀 [3/3] Starting seerplay-v3..."
podman run --privileged \
  -p 8080:8000 \
  -p 8181:8080 \
  -p 1521:1521 \
  -v seerplay-storage:/var/lib/containers \
  -e TMPDIR=/var/lib/containers/tmp \
  -d --name seerplay-v3 seerplay-omni:latest

echo ""
echo "✅ Container started successfully!"
echo "📜 Tailing logs below (Press Ctrl+C to detach, container will keep running)..."
echo "============================================================"
podman logs -f seerplay-v3
