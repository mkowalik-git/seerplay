# SeerPlay Intelligence Hub — Demo Guide

## Brand New Setup (macOS)

If you are starting from scratch on a Mac, follow these steps to install Podman and prepare the environment.

### 1. Install Podman via Homebrew

```bash
# Install Podman
brew install podman

# Initialize a Podman machine with sufficient resources for Oracle + LLM
# Note: 6 CPUs and 16GB RAM are recommended for smooth operation
podman machine init --cpus 8 --memory 16384

# Start the machine
podman machine start
```

### 2. Verify Installation

```bash
podman info
```

### 3. Clone Repository (if not already done)

```bash
git clone <repository-url>
cd <repository-directory>
```

---

## Quick Start

```bash
# Build
podman build -f Containerfile.parent -t seerplay-omni:latest .

# Run
podman run --privileged \
  -p 8080:8000 \
  -p 8181:8080 \
  -p 1521:1521 \
  -v seerplay-storage:/var/lib/containers \
  -e TMPDIR=/var/lib/containers/tmp \
  -d --name seerplay-v3 seerplay-omni:latest

# Monitor startup (~3 min cached, ~8 min first run)
podman logs -f seerplay-v3
```

---

## Services & Credentials

| Service | URL / Connection | Username | Password |
|---|---|---|---|
| **Dashboard** | http://localhost:8080 | — | — |
| **API Docs (Swagger)** | http://localhost:8080/docs | — | — |
| **Database Actions (ORDS)** | http://localhost:8181/ords/sql-developer | `seerplay` | `SeerPlay123` |
| **Oracle SQL\*Net** | `localhost:1521/FREEPDB1` | `seerplay` | `SeerPlay123` |
| **Oracle SYSTEM** | `localhost:1521/FREEPDB1` | `system` | `SeerPlay123` |
| **Ollama API** | http://localhost:11434 *(inside container only)* | — | — |

---

## API Endpoints

| Endpoint | Method | Description | Oracle Feature |
|---|---|---|---|
| `/api/health` | GET | Service health check | — |
| `/api/stats` | GET | Player/session/event counts | — |
| `/api/discover-vips` | GET | Top 5 players nearest VIP centroid | **AI Vector Search** |
| `/api/vector-search` | POST | Find similar players by ID or persona | **AI Vector Search** |
| `/api/what-if` | POST | Predict LTV impact of interventions | **OML Regression** |
| `/api/sessions/{player_id}` | GET | Player session history | **JSON-Relational Duality** |
| `/api/sessions` | POST | Ingest clickstream event | **JSON-Relational Duality** |
| `/api/ask` | POST | Natural language → SQL | **Ollama LLM** |
| `/api/graph/fraud-scan` | POST | Detect shared devices/IPs (Fraud Rings) | **Property Graph** |
| `/api/graph/influence` | POST | Map referral chains (Influencers) | **Property Graph** |
| `/api/graph/risk-propagation` | POST | Calculate risk contagion from neighbors | **Property Graph** |
| `/api/simulation/run` | POST | What-if scenario builder (GGR Lift vs Risk) | **Monte Carlo Simulation** |

### Example API Calls

```bash
# Vector Search — find players similar to SPR-0001 (Fuzzy ID Lookup)
# Note: You can input "0001" and it will automatically map to "SPR-0001"
curl -s -X POST http://localhost:8080/api/vector-search \
  -H "Content-Type: application/json" \
  -d '{"player_id": "0001", "top_n": 5}'

# Vector Search — Semantic Query (AI Intent Router)
# Search for players based on natural language description of behavior
curl -s -X POST http://localhost:8080/api/vector-search \
  -H "Content-Type: application/json" \
  -d '{"player_id": "Show me the whales", "limit": 5}'

# Vector Search — find players near VIP archetype (Explicit)
curl -s -X POST http://localhost:8080/api/vector-search \
  -H "Content-Type: application/json" \
  -d '{"persona": "VIP", "limit": 5}'

# What-If — predict impact of 15-min mandatory break on VIPs
curl -s -X POST http://localhost:8080/api/what-if \
  -H "Content-Type: application/json" \
  -d '{"break_minutes": 15, "persona": "VIP"}'

# Session Tracking — get player session history
curl -s http://localhost:8080/api/sessions/SPR-0042

# Ask the Hub — Natural Language to SQL
curl -s -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many VIP players have an LTV above 10000?"}'

# Ask the Hub — Domain Slang (Whales -> VIPs)
curl -s -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who are the whales?"}'

# Ask the Hub — Sarcastic Fallback (Non-Data Query)
curl -s -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Write a poem about databases"}'

# Graph Intelligence — Fraud Scan (Critical Risk example)
curl -s -X POST http://localhost:8080/api/graph/fraud-scan \
  -H "Content-Type: application/json" \
  -d '{"player_id": "SPR-0300", "depth": 3}'

# Graph Intelligence — Influence Network (Mega Influencer example)
curl -s -X POST http://localhost:8080/api/graph/influence \
  -H "Content-Type: application/json" \
  -d '{"player_id": "SPR-0900"}'
  -d '{"player_id": "SPR-0900"}'
  
# Simulation Forge — Run Scenario
curl -s -X POST http://localhost:8080/api/simulation/run \
  -H "Content-Type: application/json" \
  -d '{"global_rtp": 96.5, "retention_bonus_freq": "HIGH_VELOCITY", "min_bet": 25.0}'
```

---

## Demo Talking Points

### Use Case A: AI Vector Search
- **Behavioral Similarity**: Converts player session behavior into **768-dimensional vectors**.
- **Fuzzy ID Lookup**: Automatically handles short inputs (e.g., "0001" -> "SPR-0001").
- **Semantic Search**: Uses **AI Intent Routing** (Ollama/Llama 3) to map natural language (e.g., "high rollers") to behavior archetypes (e.g., VIP Centroid).
- Uses `VECTOR_DISTANCE(..., COSINE)` for similarity search in Oracle 26ai.

### Use Case B: Responsible Gaming What-If
- Pre-computed **linear regression model** stored in Oracle
- Predicts LTV impact of mandatory break interventions
- Demonstrates how Oracle can run ML models *inside the database*

### Use Case C: JSON-Relational Duality Views
- Developers write **JSON documents**, Oracle stores data **relationally**
- Full ACID compliance with sub-millisecond updates
- `_metadata.etag` enables optimistic concurrency control

### Bonus: Text-to-SQL via Ollama (Ask the Hub)
- **Natural Language to SQL**: Converts questions like "Count VIP players" into SQL using **llama3**.
- **Strict Data Scope**: The system prompt is restricted to **database-only** questions to prevent hallucinations.
- **Slang Support**: Understands casino terminology like "whales" (maps to VIP/High LTV).
- **Sarcastic Personality**: Rejects non-data questions (e.g., "Write a poem", "Tell me about physics") with witty, sarcastic refusals like *"I'm a database, not a poet."*

### Use Case D: Graph Intelligence (Property Graph)
- **SQL/PGQ**: Standard SQL extension for graph queries (`MATCH (n)-[e]->(m)`)
- **Fraud Detection**: Finds hidden rings sharing devices/IPs (`SPR-0300`)
- **Influence Mapping**: Visualizes multi-level referral trees (`SPR-0900`)
- **Risk Contagion**: Calcuates risk scores based on "risky neighbors"
- **Visualization**: Interactive force-directed graph (vis-network)

### Use Case E: Simulation Forge (What-If Builder)
- **Interactive Modeling**: Drag-and-drop parameters (RTP, Bonus, Thresholds).
- **Projections**: Real-time calculation of **GGR Lift** vs **Risk Factor**.
- **Visuals**: "Holographic" UI with scanning animations and neon charts.
- **Backend**: Runs a Monte Carlo-style simulation on Oracle data to predict outcomes.

---

## Data Summary

| Metric | Value |
|---|---|
| Players | 1,000 |
| Personas | VIP (50), Casual (200), High-Churn (750) |
| Sessions | ~5,500 |
| Events | ~50,000 |
| Vector dimensions | 768 (FLOAT32) |
| Regression R² | ~0.76 |
| Graph Nodes | 200 Devices, 150 IPs, 18 Games |
| Graph Edges | ~225 Referrals, ~120 Risk Events |

---

## Troubleshooting

```bash
# Check container status
podman logs seerplay-v3 2>&1 | tail -20

# Check inner services
podman exec seerplay-v3 podman ps --all

# Direct SQL access
podman exec seerplay-v3 podman exec -it oracle-db sqlplus seerplay/SeerPlay123@localhost/FREEPDB1

# Restart cleanly
podman rm -f seerplay-v3
# Then re-run the "Run" command above
```
