# Project Rules & Architecture Documentation

This document serves as a blueprint for recreating the **SeerPlay Intelligence Hub** demo environment. It captures the architectural decisions, component interactions, and implementation patterns used to build a self-contained, database-backed AI application.

## 1. Architecture Overview

 The project implements a **"Pod-in-Container" (PinC)** architecture using **Podman**. This allows the entire complex environment (Oracle DB, AI Models, API, etc.) to be packaged into a single deliverable image (`seerplay-omni`), which then orchestrates its own internal services.

### Core Components
| Service | Technology | Role |
|---|---|---|
| **Parent Container** | Podman (v4/v5) | The outer shell. Orchestrates the inner pod and services. |
| **Database** | Oracle 26ai Free | Stores relational data, vectors, JSON, and graph data. |
| **AI Brain** | Ollama (llama3) | Local LLM for Text-to-SQL generation. |
| **API Backend** | Python (FastAPI) | Connects UI to DB/LLM. Serves the Dashboard. |
| **Orchestrator** | Bash (`entrypoint.sh`) | Manages startup order, health checks, and data seeding. |

---

## 2. Orchestration Patterns

### A. The "Parent-Child" Container Strategy
Instead of asking users to install Docker Compose, Oracle, and Python locally, we bundle everything into `seerplay-omni` (`Containerfile.parent`).
- **Outer Layer**: Contains Podman, Python, and the project source code.
- **Inner Layer**: The `entrypoint.sh` script inside the container uses `podman run` to start the actual services.
- **Networking**: Critical! We use `--network=host` for the inner pod (`seerplay-pod`). This ensures all inner containers share the parent's network namespace, avoiding complex port mapping issues in nested virtualization.

**Recreation Rule:**
> When nesting containers, prefer `podman pod create --network=host` to simplify communication between the parent orchestrator script and the running services.

### B. Robust Startup Logic (`entrypoint.sh`)
Services like Oracle Database and Large Language Models (LLMs) take time to initialize. The orchestration script must be robust:
1.  **Cleanup First**: Always run `podman rm -f ...` before starting to ensure a clean slate.
2.  **Wait Loops**: specialized "Wait for X" blocks:
    - **Wait for Ollama**: Poll `localhost:11434` before attempting to pull a model.
    - **Wait for Oracle**: Loop `sqlplus` connection attempts until successful (can take 2-5 mins).
3.  **Idempotency**: The `init.sql` and `seeder.py` should act gracefully if run multiple times (e.g., using `CREATE OR REPLACE`, `TRUNCATE` before `INSERT`).

---

## 3. Database Patterns (Oracle 26ai)

### A. dedicated Service User
Avoid using `SYSTEM` for application logic.
- **Rule**: Create a dedicated user (`seerplay`) with specific grants (`CREATE SESSION`, `CREATE VIEW`, `CREATE PROPERTY GRAPH`).
- **Tablespace**: Use `USERS` tablespace (supports ASSM) rather than `SYSTEM` to ensure advanced features like `VECTOR` indexing work correctly.

### B. Converged Data Types
The schema (`init.sql`) demonstrates Oracle's converged engine:
- **Vectors**: `VECTOR(768, FLOAT32)` column in `player_sessions` for semantic similarity.
- **JSON**: stored in `session_events` but exposed relationally via **JSON-Relational Duality Views**.
- **Graph**: `CREATE PROPERTY GRAPH` overlays graph relationships (edges) on existing relational tables (foreign keys).

### C. Duality Views
- **Pattern**: `CREATE JSON RELATIONAL DUALITY VIEW ... WITH INSERT UPDATE DELETE`.
- **Benefit**: The app interacts with simple JSON documents, but the database stores normalized, ACID-compliant relational rows.

---

## 4. AI Integration Patterns

### A. Text-to-SQL (RAG-lite)
The `/api/ask` endpoint demonstrates a safe Text-to-SQL pattern:
1.  **Context Injection**: The LLM prompt includes a concise schema definition.
2.  **Constraint**: The prompt explicitly asks for "ONLY the SQL query" and the code reinforces this by checking for `SELECT` statements only.
3.  **Execution**: The generated SQL is executed directly against the Oracle DB.

### B. In-Database Machine Learning (Simulated)
The `/api/what-if` endpoint demonstrates "Moving the Model to the Data":
- **Training**: `seeder.py` trains a Linear Regression model using `scikit-learn`.
- **Storage**: The model coefficients are saved into the `whatif_model` table.
- **Inference**: The API pulls coefficients and runs the prediction equation. In a full production version, this would use **OML4SQL** (Oracle Machine Learning for SQL) to run native inference.

---

## 5. Directory Structure for Recreation

To recreate this project, ensure the following file structure:

```
.
├── Containerfile.parent    # The outer image definition
├── Containerfile.api       # The inner API image definition
├── entrypoint.sh           # Orchestration script
├── init.sql                # Database schema & features
├── main.py                 # FastAPI application
├── seeder.py               # Data generator & model training
└── templates/
    └── index.html          # Dashboard UI
```

All files should be copied into the container at `/app`.
