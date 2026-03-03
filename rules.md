# 🏛️ Architecture Blueprint & Project Rules

This document serves as the foundational blueprint for creating **Intelligence Hub** applications. The current implementation leverages a reusable, self-contained architecture allowing you to easily spin up a database-backed AI demo environment. 

Whether you are building a **Casino Analytics** dashboard (like the original SeerPlay), a **Financial Portfolio Tracker**, or a **Healthcare Patient Monitor**, the underlying architectural patterns remain the same.

## 1. 🏗️ Architecture Overview

The project implements an innovative **"Pod-in-Container" (PinC)** architecture. This encapsulates a sprawling ecosystem (Database, LLMs, APIs) into a single deliverable image, which then orchestrates its own internal services.

### Core Components
| Component | Technology | Role |
| :--- | :--- | :--- |
| **Parent Container** | Podman (v4/v5) / Docker | The outer shell. Acts as a mini-OS to orchestrate the inner pod and microservices. |
| **Converged Database** | Oracle 26ai Free | The system of record. Stores relational data, vectors, JSON documents, and graph relationships. |
| **Local AI Brain** | Ollama | Local LLM engine (e.g., `gemma:2b`, `llama3`). Powers native Text-to-SQL and predictive features. |
| **API Backend** | Python (FastAPI) | The logic layer connecting the UI front-end to the DB/LLM back-end. |
| **Orchestrator** | Bash (`entrypoint.sh`) | The nervous system managing startup orders, health checks, and state initializations. |

---

## 2. 🚦 Orchestration Patterns

### A. The "Parent-Child" Container Strategy
We eliminate local setup friction (no need to manually configure Docker Compose, databases, and Python environs on the host machine) by shifting everything into a master image (e.g., `Containerfile.parent`).
- **Outer Layer**: Contains the container runtime, Python, and project source.
- **Inner Layer**: The runtime script (`entrypoint.sh`) executes `podman run` (or equivalent) to orchestrate the core services.
- **Networking [CRITICAL]**: Use `--network=host` for the inner pod. This allows all sub-containers to share the parent's network namespace, completely bypassing complex port forwarding in nested virtualization environments.

> **Recreation Rule:** Always prioritize host networking within nested pods to ensure seamless cross-container communication (e.g., API communicating effortlessly with the Database and AI Brain).

### B. Robust Startup Logic
Databases and LLMs are notoriously slow to initialize. The `entrypoint.sh` script must be fault-tolerant:
1.  **Aggressive Cleanup**: Execute `rm -f` sequence on previous containers to ensure a sterile startup environment.
2.  **Intelligent Wait Loops**: Implement blocking, polling loops:
    - **Wait for Brain**: Ping the Ollama API endpoint (`localhost:11434`) until a `200 OK` is returned before pulling models.
    - **Wait for DB**: Repeatedly poll the database listener until `sqlplus` or the connection string registers a successful handshake.
3.  **Idempotent Data Seeding**: Lifecycle scripts (`init.sql`, `seeder.py`) must gracefully handle multiple executions (`CREATE OR REPLACE`, `TRUNCATE` before insert, robust `IF NOT EXISTS` logic).

---

## 3. 💾 Data Engineering Patterns

### A. Dedicated Service Users
**Rule:** Never use root or `SYSTEM` accounts for application operations.
- Create domain-specific users (e.g., `hub_user`) with scoped privileges (`CREATE SESSION`, `CREATE VIEW`, `CREATE PROPERTY GRAPH`).
- Utilize a standard `USERS` tablespace (with ASSM enabled) to fully support advanced vector and large-object indexing.

### B. The Converged Data Model
Adapt the `init.sql` schema to your specific domain using these converged principles:
- **Relational Base**: Build standard, normalized tables for core entities (e.g., `users`, `transactions`).
- **Vector Search**: Use `VECTOR([Dimensions], [Format])` columns. Generate and store embeddings for semantic similarity across text or entities.
- **Property Graphs**: Overlay `CREATE PROPERTY GRAPH` on top of standard foreign keys to run powerful network/relationship queries.
- **JSON Duality Views**: 
  - **Pattern**: `CREATE JSON RELATIONAL DUALITY VIEW ... WITH INSERT UPDATE DELETE`
  - **Why?**: The application interacts with simplified JSON documents, while the database automatically maps them into ACID-compliant structured rows.

---

## 4. 🧠 AI & Logic Integration Patterns

### A. Secure Text-to-SQL Architecture (RAG-lite)
When implementing natural language interfaces (e.g., `/api/ask`), enforce these constraints:
1.  **Schema Context Injection**: Dynamically push a concise schema definition of the domain into the LLM's system prompt.
2.  **Hard Guardrails**: Instruct the LLM to output "ONLY valid SQL". 
3.  **App-Layer Validation**: Before execution, intercept the query and forcibly validate it is a `SELECT` statement to prevent accidental mutation.

### B. "Model-to-Data" Predictive Analytics
Keep the compute close to the data for what-if scenarios:
- **Training (Seeder layer)**: Use scripts (`seeder.py`) with ML libraries (`scikit-learn`) to synthesize and train simple models on the generated dataset.
- **Storage**: Save the resulting coefficients/weights as native rows in a dedicated configuration table.
- **Inference**: The API reconstructs the equation dynamically from the DB table. (Note: For massive scale, replace this with native DB machine learning routines like OML4SQL).

---

## 5. 📂 Project Template Structure

To spin up a new domain application, duplicate the repository and modify the domain-specific files within this structure:

```text
.
├── Containerfile.parent    # Architecture: Outer shell environment
├── Containerfile.api       # Architecture: Inner application specs
├── entrypoint.sh           # Architecture: Container and service orchestrator
├── init.sql                # Domain: Database schema & converged features
├── main.py                 # Domain: API routes & LLM orchestrator
├── seeder.py               # Domain: Fake data generation & ML training
└── templates/
    └── index.html          # Domain: The front-end dashboard
```

**How to Adapt:** To create a completely new application, you only need to change the **Domain** files (`init.sql`, `main.py`, `seeder.py`, `templates/index.html`). The architectural components will seamlessly handle the underlying infrastructure.
