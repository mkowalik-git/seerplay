# SeerPlay Intelligence Hub 🔮

Ever wished your database could not only store your data but also understand it, predict the future, and crack a sarcastic joke while doing it? Welcome to the **SeerPlay Intelligence Hub** — your all-in-one, AI-powered command center. 

Think of it less like a traditional application and more like a Swiss Army knife that went to grad school for data science. We've packed an Oracle database, a local LLM, and a sleek API into a single box so you don't have to spend your weekend configuring ports. (You're welcome!)

## What is this thing? 🤔

The SeerPlay Intelligence Hub is a self-contained demo environment that showcases what happens when you let modern databases off the leash. We're talking vectors, graphs, JSON, and machine learning — all living harmoniously under one roof. 

We use a "Pod-in-Container" (PinC) architecture. What does that mean? It means we've shoved an entire complex ecosystem (Oracle 26ai, Ollama, FastAPI) into a single Podman image. It's like putting a supercar engine inside a minivan — unexpectedly powerful and highly convenient. 

## The Cool Stuff Inside 🧰

Here is a quick tour of our shiny features:

*   **AI Vector Search:** Why search by exact match when you can search by *vibe*? We convert player behavior into 768-dimensional vectors. Looking for "high rollers"? We map that natural language right to the behavioral centroid. 
*   **Graph Intelligence:** Uncover hidden connections faster than a gossip columnist. Using Oracle Property Graphs, we map out referral networks (who influences whom) and scan for fraud rings sharing devices. 
*   **Ask the Hub (Text-to-SQL):** Powered by Ollama and Gemma 2B, you can literally just *ask* your database questions. Try asking "Who are the whales?" and watch it translate domain slang into flawless SQL. (Warning: It *will* get sarcastic if you ask it to write poetry).
*   **JSON-Relational Duality:** We let application developers work with simple JSON documents while the database stores them relationally. It's the best of both worlds — like eating cake and simultaneously getting a six-pack.
*   **Simulation Forge:** A drag-and-drop "what-if" builder that runs Monte Carlo simulations and linear regressions right inside the database to predict lifetime value (LTV) impacts. No data movement required.

## Getting Started 🚀

Ready to spin this up? Don't worry, it's virtually foolproof. No lab coats required!

### Step 1: Prep your Engine (macOS)
If you're on a Mac, you'll need Podman. 

```bash
# Get Podman on your machine
brew install podman

# Init a machine with enough muscle (6+ CPUs and 16GB RAM is the sweet spot)
podman machine init --cpus 8 --memory 16384

# Fire it up!
podman machine start
```

### Step 2: Build and Run
Time to bring SeerPlay to life.

```bash
# Build the master image 
podman build -f Containerfile.parent -t seerplay-omni:latest .

# Run the beast (Privileged mode gives us the network magic we need)
podman run --privileged \
  -p 8080:8000 \
  -p 8181:8080 \
  -p 1521:1521 \
  -v seerplay-storage:/var/lib/containers \
  -e TMPDIR=/var/lib/containers/tmp \
  -d --name seerplay-v3 seerplay-omni:latest

# Watch the logs to see it boot (Grab a coffee, first run takes ~8 mins)
podman logs -f seerplay-v3
```

*Note: The first boot takes a minute because it's downloading an AI model and initializing an Oracle database. Patience, grasshopper! You got this.*

## Where to go from here 🗺️

Once the logs say you're good to go, you can hit up the following URLs:

*   **The Dashboard:** `http://localhost:8080` (Your mission control)
*   **API Docs:** `http://localhost:8080/docs` (For the curious developer)
*   **Database Actions:** `http://localhost:8181/ords/sql-developer` (Login: `seerplay` / `SeerPlay123`)

Want to test out the natural language search? Try throwing this at the API:

```bash
curl -s -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many VIP players have an LTV above 10000?"}'
```

Dive in, break things, and explore. If you get lost, just remember the LLM is right there to help (or mock you lovingly).
