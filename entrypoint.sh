#!/bin/bash
# set -e  <-- DISABLED FOR DEBUGGING

# Ensure temp dir exists on the persistent volume (set via TMPDIR env var)
mkdir -p "${TMPDIR:-/tmp}"

echo "============================================================"
echo " SeerPlay Intelligence Hub — Starting Pod Orchestration"
echo "============================================================"

# ---------------------------------------------------------------
# 0. Cleanup (Safe Restart)
# ---------------------------------------------------------------
echo "[0/9] Cleanup..."
podman pod rm -f seerplay-pod >/dev/null 2>&1 || true
podman rm -f oracle-db ollama-brain seerplay-api >/dev/null 2>&1 || true

# ---------------------------------------------------------------
# 1. Build the inner API image
# ---------------------------------------------------------------
echo "[1/9] Building inner seerplay-api image..."
podman build -t seerplay-api -f Containerfile.api . || echo "⚠ API Build Failed"

# ---------------------------------------------------------------
# 2. Create the Pod
# ---------------------------------------------------------------
echo "[2/9] Creating seerplay-pod..."
podman pod create --name seerplay-pod --network=host

# ---------------------------------------------------------------
# 3. Start Oracle 26ai Free
# ---------------------------------------------------------------
echo "[3/9] Starting Oracle Database 26ai Free..."
podman run -d --pod seerplay-pod --name oracle-db \
    -e ORACLE_PWD=SeerPlay123 \
    --shm-size=2g \
    container-registry.oracle.com/database/free:latest

# ---------------------------------------------------------------
# 4. Start Ollama
# ---------------------------------------------------------------
echo "[4/9] Starting Ollama AI Brain..."
mkdir -p /var/lib/containers/ollama_models
podman run -d --pod seerplay-pod --name ollama-brain \
    -v /var/lib/containers/ollama_models:/root/.ollama \
    docker.io/ollama/ollama

# ---------------------------------------------------------------
# 5. Wait for Ollama & Pull Model
# ---------------------------------------------------------------
echo "[5/9] Waiting for Ollama..."
MAX_WAIT=120
ELAPSED=0
until curl -sf http://localhost:11434 > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "  ⚠ Ollama timeout"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

echo "[5/9] Pulling llama3..."
podman exec ollama-brain ollama pull llama3 || echo "  ⚠ Model pull failed"

# ---------------------------------------------------------------
# 6. Wait for Oracle to be HEALTHY
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# 5.5 FIX: Force Listener Registration (Fixes ORA-12514 / Hostname Resolution)
# ---------------------------------------------------------------
echo "[5.5/9] Fixing Oracle Listener Registration..."
until podman exec oracle-db bash -c "printf \"ALTER SYSTEM SET LOCAL_LISTENER='(ADDRESS=(PROTOCOL=TCP)(HOST=127.0.0.1)(PORT=1521))' SCOPE=MEMORY;\\nALTER SYSTEM REGISTER;\\n\" | sqlplus -S / as sysdba" >/dev/null 2>&1; do
    echo "  Waiting for instance to be ready for config..."
    sleep 5
done
echo "  ✅ Listener registration forced."

# ---------------------------------------------------------------
# 6. Wait for Oracle to be HEALTHY
# ---------------------------------------------------------------
echo "[6/9] Waiting for Oracle DB Service (max 900s)..."
MAX_WAIT=900
ELAPSED=0
until podman exec oracle-db bash -c "echo \"WHENEVER SQLERROR EXIT FAILURE; SELECT 1 FROM DUAL;\" | sqlplus -S system/SeerPlay123@localhost/FREEPDB1" > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "  ❌ Oracle Timeout"
        # Don't exit, keep running for debug
        break
    fi
    echo "  Waiting for Oracle DB Service... (${ELAPSED}s)"
    sleep 15
    ELAPSED=$((ELAPSED + 15))
done

# ---------------------------------------------------------------
# 7. Run Schema Initialization
# ---------------------------------------------------------------
echo "[7/9] Running init.sql..."
podman cp init.sql oracle-db:/tmp/init.sql
# Retry init.sql a few times in case of ORA-12514 race
INIT_SUCCESS=false
for i in {1..3}; do
    echo "  Attempt $i..."
    # fail-fast if sql error
    if podman exec oracle-db bash -c "printf \"WHENEVER SQLERROR EXIT FAILURE;\\n@/tmp/init.sql\\n\" | sqlplus -S system/SeerPlay123@localhost/FREEPDB1"; then
        INIT_SUCCESS=true
        break
    fi
    sleep 5
done

if [ "$INIT_SUCCESS" = false ]; then
    echo "❌ init.sql Failed! Exiting."
    exit 1
fi

# ---------------------------------------------------------------
# 8. Start the API container
# ---------------------------------------------------------------
echo "[8/9] Starting SeerPlay API..."
podman run -d --pod seerplay-pod --name seerplay-api seerplay-api
sleep 5

# Check if API is running
if ! podman ps | grep -q seerplay-api; then
    echo "❌ API Container Died! Logs:"
    podman logs seerplay-api
fi

# ---------------------------------------------------------------
# 9. Run Data Seeder
# ---------------------------------------------------------------
echo "[9/9] Running Data Seeder..."
podman exec seerplay-api python3 seeder.py || echo "❌ Seeder Failed"

# OPTIMIZATION
sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

echo "✅ Boot Sequence Complete (Errors may have occurred above)"
echo "--------------------------------------------------------"
echo "Debugging: Container will stay alive."

tail -f /dev/null
