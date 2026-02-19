#!/bin/bash
set -e

echo "============================================================"
echo " SeerPlay Intelligence Hub — Terminate & Cleanup"
echo "============================================================"

# 1. Stop and remove the main container
echo "🛑 [1/2] Stopping and removing seerplay-v3..."
podman rm -f seerplay-v3 2>/dev/null || true

# 2. Prune system (dangling images, stopped containers, networks)
echo "🧹 [2/2] Pruning system resources (images, volumes, networks)..."
podman system prune -f

# 3. Aggressive Cleanup (User Request)
echo "🔥 [3/3] Aggressive Prune (All images & volumes)..."
podman image prune -a -f
podman volume prune -f

echo "============================================================"
echo "✅ Environment teardown complete."
echo "============================================================"
