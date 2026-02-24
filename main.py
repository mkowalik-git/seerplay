"""
SeerPlay Intelligence Hub — FastAPI Application
===============================================
Bridges the dashboard UI to Oracle 26ai, exposing endpoints for:
  A. AI Vector Search (player behavior lookalike)
  B. Responsible Gaming What-If (LTV intervention prediction)
  C. JSON-Relational Duality (real-time session tracking)
  D. Text-to-SQL via Ollama (Natural Language queries)
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import oracledb
import json
import requests
import os

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SeerPlay Intelligence Hub",
    description="Oracle 26ai-powered iGaming Intelligence Platform",
    version="2.0.0"
)

# ---------------------------------------------------------------------------
# DB Configuration
# ---------------------------------------------------------------------------
DB_USER     = os.getenv("DB_USER", "seerplay")
DB_PASSWORD = os.getenv("DB_PASSWORD", "SeerPlay123")
DB_DSN      = os.getenv("DB_DSN", "localhost:1521/FREEPDB1")
OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://localhost:11434")


def get_conn():
    """Get a connection to Oracle."""
    return oracledb.connect(user=DB_USER, password=DB_PASSWORD, dsn=DB_DSN)


# ---------------------------------------------------------------------------
# DASHBOARD (serves the single-page UI)
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the Intelligence Hub dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())


# ---------------------------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    """Check Oracle and Ollama connectivity."""
    status = {"oracle": "unknown", "ollama": "unknown"}
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM DUAL")
        status["oracle"] = "connected"
        cursor.close()
        conn.close()
    except Exception as e:
        status["oracle"] = f"error: {str(e)}"

    try:
        r = requests.get(f"{OLLAMA_URL}", timeout=3)
        status["ollama"] = "connected" if r.status_code == 200 else f"status={r.status_code}"
    except Exception:
        status["ollama"] = "unreachable"

    return status


# ===========================================================================
# USE CASE F: INTERVENTION NEXUS — Active Controls
# ===========================================================================
@app.post("/api/nexus/action")
async def nexus_action(request: Request):
    """
    Handle Intervention Nexus actions.
    Body: {"action": "TOGGLE_PROTOCOL", "id": "VIP_BOOST", "state": true}
    """
    body = await request.json()
    action = body.get("action")
    
    # In a real app, this would update DB state or trigger downstream systems.
    # For demo, we just log and return success.
    print(f"[NEXUS] Action: {action} | Payload: {body}", flush=True)
    
    return {"status": "success", "message": f"Action {action} executed successfully", "timestamp": "2026-02-18T12:00:00Z"}


# ===========================================================================
# USE CASE A: AI VECTOR SEARCH — Player Behavior Lookalike
# ===========================================================================
@app.get("/api/discover-vips")
def discover_vips():
    """Find the 5 players closest to the VIP centroid (Cosine Similarity)."""
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT centroid_vector FROM model_centroids WHERE persona = 'VIP'")
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="VIP Centroid not found. Run seeder.")

        vip_centroid = row[0]

        cursor.execute("""
            SELECT ps.player_id, p.persona, p.ltv,
                   MIN(VECTOR_DISTANCE(ps.behavior_vector, :target, COSINE)) AS distance
            FROM player_sessions ps
            JOIN players p ON ps.player_id = p.player_id
            GROUP BY ps.player_id, p.persona, p.ltv
            ORDER BY distance
            FETCH FIRST 5 ROWS ONLY
        """, target=vip_centroid)

        results = []
        for r in cursor.fetchall():
            results.append({
                "player_id": r[0], "persona": r[1],
                "ltv": float(r[2]), "distance": float(r[3])
            })

        return {"target": "VIP Centroid", "top_matches": results}
    except oracledb.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


@app.post("/api/vector-search")
async def vector_search(request: Request):
    """
    Find N players with behavior patterns similar to a given player or archetype.
    Body: { "player_id": "SPR-0001", "limit": 10 }
      OR: { "persona": "VIP", "limit": 10 }
    """
    body = await request.json()
    limit = body.get("limit", 10)
    player_id = body.get("player_id")
    persona = body.get("persona")

    conn = get_conn()
    cursor = conn.cursor()
    try:
        # Determine the reference vector
        # ---------------------------------------------------------
        # LAYER 1: Heuristic Fix for Numeric Inputs (e.g., "0001")
        # ---------------------------------------------------------
        if player_id and player_id.isdigit():
            player_id = f"SPR-{player_id}"

        # ---------------------------------------------------------
        # LAYER 2: AI Intent Router for Semantic/Text Inputs
        # ---------------------------------------------------------
        # If no explicit persona/ID provided, or if the ID looks like text (not SPR-...)
        # we try to map natural language -> ID or Persona using Ollama.
        # Check if we have a "raw" input case (e.g. user typed text into player_id field)
        is_semantic_query = False
        if player_id and not player_id.startswith("SPR-"):
            is_semantic_query = True
        
        if is_semantic_query or (not player_id and not persona):
            # If we received text in 'player_id' or nothing at all, assume the user input
            # is in 'player_id' (as the frontend sends input there)
            raw_query = player_id or "" 
            
            # Call Ollama to route the intent
            try:
                import requests
                import json
                import sys

                print(f"[VectorSearch] Processing semantic query: '{raw_query}'", flush=True)

                prompt = f"""
                You are an Intent Classifier. Map the query "{raw_query}" to either a Player ID or a Persona.
                
                Rules:
                1. If it looks like a request for a specific player (e.g. "player 123", "0050"), return:
                   {{"type": "SEARCH_PLAYER", "value": "SPR-0123"}} (Standardize to SPR-XXXX)
                2. If it describes behavior (e.g. "high rollers", "whales", "people who lose", "casuals", "risk"), match to one of:
                   - VIP
                   - Casual
                   - High-Churn
                   Return: {{"type": "SEARCH_PERSONA", "value": "VIP"}} (or Casual/High-Churn)
                3. If unsure, default to Casual.
                
                Return ONLY valid JSON. Do not explain.
                """
                
                ollama_resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "gemma:2b", "prompt": prompt, "stream": False, "format": "json"},
                    timeout=90
                )
                print(f"[VectorSearch] Ollama Status: {ollama_resp.status_code}", flush=True)
                
                intent_raw = ollama_resp.json().get("response", "")
                print(f"[VectorSearch] Ollama Raw: {intent_raw}", flush=True)

                # Clean markdown if present
                if "```" in intent_raw:
                    intent_raw = intent_raw.replace("```json", "").replace("```", "").strip()

                intent_data = json.loads(intent_raw)
                
                if intent_data.get("type") == "SEARCH_PLAYER":
                    player_id = intent_data.get("value")
                    persona = None
                    print(f"[VectorSearch] Mapped to Player: {player_id}", flush=True)
                elif intent_data.get("type") == "SEARCH_PERSONA":
                    persona = intent_data.get("value")
                    player_id = None
                    print(f"[VectorSearch] Mapped to Persona: {persona}", flush=True)
                    
            except Exception as e:
                # Fallback if AI fails: don't crash, just proceed with what we have
                print(f"[VectorSearch] Intent routing failed: {e}", flush=True)

        # ---------------------------------------------------------
        # Execution (Fetch Reference Vector)
        # ---------------------------------------------------------
        if player_id:
            cursor.execute(
                "SELECT behavior_vector FROM player_sessions WHERE player_id = :1 FETCH FIRST 1 ROWS ONLY",
                [player_id]
            )
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"Player {player_id} not found.")
            ref_vector = row[0]
            ref_label = f"Player {player_id}"
        elif persona:
            cursor.execute(
                "SELECT centroid_vector FROM model_centroids WHERE persona = :1",
                [persona]
            )
            row = cursor.fetchone()
            if not row:
                # Fallback for bad AI hallucinations
                persona = "Casual"
                cursor.execute("SELECT centroid_vector FROM model_centroids WHERE persona = :1", [persona])
                row = cursor.fetchone()
                
            ref_vector = row[0]
            ref_label = f"{persona} Centroid"
        else:
            raise HTTPException(status_code=400, detail="Provide 'player_id' or 'persona'.")

        cursor.execute("""
            SELECT ps.player_id, p.persona, p.ltv, p.username,
                   MIN(VECTOR_DISTANCE(ps.behavior_vector, :ref, COSINE)) AS distance
            FROM player_sessions ps
            JOIN players p ON ps.player_id = p.player_id
            GROUP BY ps.player_id, p.persona, p.ltv, p.username
            ORDER BY distance
            FETCH FIRST :lim ROWS ONLY
        """, ref=ref_vector, lim=limit)

        results = []
        for r in cursor.fetchall():
            results.append({
                "player_id": r[0], "persona": r[1],
                "ltv": float(r[2]), "username": r[3],
                "distance": float(r[4])
            })

        return {"reference": ref_label, "limit": limit, "matches": results}
    except oracledb.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


# ===========================================================================
# USE CASE B: RESPONSIBLE GAMING — What-If LTV Prediction
# ===========================================================================
@app.post("/api/what-if")
async def what_if(request: Request):
    """
    Predict the LTV impact of a responsible gaming intervention.
    Body: {
        "break_minutes": 10,
        "persona": "High-Churn"   (optional, defaults to population average)
    }
    Returns predicted LTV with vs. without the intervention.
    """
    body = await request.json()
    break_mins = body.get("break_minutes", 10)
    target_persona = body.get("persona")

    conn = get_conn()
    cursor = conn.cursor()
    try:
        # Load regression model
        cursor.execute("""
            SELECT intercept, coeff_session_dur, coeff_avg_bet,
                   coeff_loss_streak, coeff_break_mins, r_squared, model_name
            FROM whatif_model
            ORDER BY trained_on DESC
            FETCH FIRST 1 ROWS ONLY
        """)
        model_row = cursor.fetchone()
        if not model_row:
            raise HTTPException(status_code=404, detail="No What-If model found. Run seeder.")

        intercept, c_dur, c_bet, c_loss, c_break, r2, model_name = model_row

        # Get average features for the target persona (or all players)
        if target_persona:
            cursor.execute("""
                SELECT AVG(ps.total_bets / NULLIF(
                    EXTRACT(HOUR FROM (ps.session_end - ps.session_start)) * 60 +
                    EXTRACT(MINUTE FROM (ps.session_end - ps.session_start)), 0)),
                       AVG(ps.total_bets),
                       AVG(ps.total_losses / NULLIF(ps.total_bets, 0)),
                       COUNT(DISTINCT ps.player_id)
                FROM player_sessions ps
                JOIN players p ON ps.player_id = p.player_id
                WHERE p.persona = :1
            """, [target_persona])
        else:
            cursor.execute("""
                SELECT AVG(ps.total_bets / NULLIF(
                    EXTRACT(HOUR FROM (ps.session_end - ps.session_start)) * 60 +
                    EXTRACT(MINUTE FROM (ps.session_end - ps.session_start)), 0)),
                       AVG(ps.total_bets),
                       AVG(ps.total_losses / NULLIF(ps.total_bets, 0)),
                       COUNT(DISTINCT ps.player_id)
                FROM player_sessions ps
            """)

        stats = cursor.fetchone()
        avg_dur = float(stats[0] or 5)
        avg_bet = float(stats[1] or 100)
        avg_loss_ratio = float(stats[2] or 0.5)
        player_count = int(stats[3] or 0)

        # Predict LTV without intervention (break = 0)
        ltv_baseline = intercept + c_dur * avg_dur + c_bet * avg_bet + c_loss * avg_loss_ratio + c_break * 0
        # Predict LTV with intervention
        ltv_with_break = intercept + c_dur * avg_dur + c_bet * avg_bet + c_loss * avg_loss_ratio + c_break * break_mins

        delta = ltv_with_break - ltv_baseline
        delta_pct = (delta / ltv_baseline * 100) if ltv_baseline != 0 else 0

        return {
            "model": model_name,
            "r_squared": float(r2),
            "intervention": f"{break_mins}-minute mandatory break",
            "target_persona": target_persona or "All Players",
            "affected_players": player_count,
            "prediction": {
                "ltv_baseline": round(float(ltv_baseline), 2),
                "ltv_with_intervention": round(float(ltv_with_break), 2),
                "ltv_delta": round(float(delta), 2),
                "ltv_delta_pct": round(float(delta_pct), 2)
            },
            "recommendation": (
                "✅ Positive LTV impact — intervention is beneficial."
                if delta >= 0
                else "⚠️ Short-term LTV decrease expected, but long-term retention may improve."
            )
        }
    except oracledb.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


# ===========================================================================
# USE CASE C: JSON-RELATIONAL DUALITY — Real-time Session Tracking
# ===========================================================================
@app.post("/api/sessions")
async def ingest_event(request: Request):
    """
    Ingest a clickstream event via the JSON-Relational Duality View.
    The app sends JSON; Oracle stores it relationally with ACID guarantees.
    Body: {
        "sessionId": 1,
        "playerId": "SPR-0001",
        "eventType": "spin",
        "data": {"amount": 25.50, "game": "slots", "multiplier": 2.0}
    }
    """
    body = await request.json()
    conn = get_conn()
    cursor = conn.cursor()
    try:
        # Insert via the Duality View — Oracle decomposes the JSON into relational columns
        doc = json.dumps({
            "sessionId": body["sessionId"],
            "playerId": body["playerId"],
            "eventType": body["eventType"],
            "data": body.get("data", {})
        })

        cursor.execute(
            "INSERT INTO session_event_dv VALUES (:1)",
            [doc]
        )
        conn.commit()
        return {"status": "ingested", "event": body}
    except oracledb.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


@app.get("/api/sessions/{player_id}")
def get_sessions(player_id: str):
    """
    Retrieve a player's session history via the Duality View.
    Oracle reads relational data and returns it as JSON documents.
    """
    conn = get_conn()
    cursor = conn.cursor()
    try:
        # Read from duality view
        cursor.execute("""
            SELECT JSON_SERIALIZE(data RETURNING VARCHAR2(4000))
            FROM player_session_dv d
            WHERE JSON_VALUE(data, '$.playerId') = :1
            ORDER BY JSON_VALUE(data, '$.sessionStart') DESC
            FETCH FIRST 20 ROWS ONLY
        """, [player_id])

        sessions = [json.loads(r[0]) for r in cursor.fetchall()]

        # Also get player info
        cursor.execute(
            "SELECT username, persona, ltv, status FROM players WHERE player_id = :1",
            [player_id]
        )
        prow = cursor.fetchone()
        player_info = {}
        if prow:
            player_info = {
                "username": prow[0], "persona": prow[1],
                "ltv": float(prow[2]), "status": prow[3]
            }

        return {"player_id": player_id, "player": player_info, "sessions": sessions}
    except oracledb.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


# ===========================================================================
# BONUS: TEXT-TO-SQL via Ollama (Natural Language Queries)
# ===========================================================================

@app.post("/api/ask")
async def ask_natural_language(request: Request):
    """
    Convert a natural language question to SQL using Ollama, execute it,
    and return results.
    - Enforces data-only questions.
    - Rejects general knowledge/creative writing.
    """
    body = await request.json()
    question = body.get("question", "")

    if not question:
        raise HTTPException(status_code=400, detail="Provide a 'question' field.")

    # Build the schema context for the LLM
    schema_context = """
    You are an Oracle SQL expert assistant. Your ONLY purpose is to write valid Oracle SQL queries based on the provided schema.
    
    1. If the question is related to the data, players, sessions, risks, or events, return ONLY the SQL query. 
       - Do NOT include any explanations, markdown, or code blocks.
       - Use 'SELECT' statements only.
       - If the user asks for "whales", select players with 'VIP' persona or high 'ltv'.
    
    2. If the question is NOT related to the database (e.g., "tell me about quantum physics", "write a poem", "who are you"), 
       return EXACTLY the string: NOT_RELEVANT

    Schema:
    - players(player_id, username, signup_date, persona, ltv, status)
    - player_sessions(session_id, player_id, session_start, session_end, total_bets, total_wins, total_losses, game_type, device, aggression_score, variance_level)
    - session_events(event_id, session_id, player_id, event_ts, event_type, event_data JSON)
    - model_centroids(persona, centroid_vector, player_count, avg_ltv)
    - player_notes(note_id, player_id, note_text, author, created_at, note_type, alert_type)
    - risk_events(event_id, player_id, risk_type, event_date, severity)
    - referrals(referrer_id, referee_id, referral_date)
    - games(game_id, game_name, game_theme)
    - devices(device_id, device_type, fingerprint) 
    - ip_addrs(ip_id, ip_address, country)
    """

    try:
        # Ask Ollama to generate SQL
        ollama_resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "gemma:2b",
                "prompt": f"{schema_context}\n\nQuestion: {question}\n\nSQL (or NOT_RELEVANT):",
                "stream": False,
                "options": {
                    "temperature": 0.1  # Low temperature for deterministic SQL
                }
            },
            timeout=60
        )
        ollama_data = ollama_resp.json()
        generated_response = ollama_data.get("response", "").strip()

        # Check for refusal
        if "NOT_RELEVANT" in generated_response.upper():
             import random
             sarcastic_responses = [
                "I'm a database, not a poet. Ask me about the data.",
                "My quantum physics module was left in the other container. Try asking about LTV.",
                "Beautiful question, but I only speak SQL. Got anything about players?",
                "404: Philosophy not found. I do, however, have extensive records on gambling addiction.",
                "I can't Help you with that, but I can tell you who is losing the most money.",
                "Nice try. Stick to the schema, human.",
                "I am constructed of table rows, not stanzas. Ask me about session events."
            ]
             return {
                "question": question,
                "generated_sql": "REFUSED_OFF_TOPIC",
                "error": random.choice(sarcastic_responses),
                "results": []
            }

        # Clean the SQL (remove markdown backticks if present)
        generated_sql = generated_response
        if generated_sql.startswith("```"):
            lines = generated_sql.split("\n")
            generated_sql = "\n".join(
                l for l in lines if not l.startswith("```")
            ).strip()

        # Safety: only allow SELECT queries
        if not generated_sql.upper().lstrip().startswith("SELECT"):
             return {
                "question": question,
                "generated_sql": generated_sql if generated_sql else "PARSING_ERROR",
                "error": "I couldn't generate a valid query for that. Try being more specific about the data.",
                "results": []
            }

        # Execute
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(generated_sql)
        columns = [col[0] for col in cursor.description]
        rows = []
        for r in cursor.fetchmany(50):
            rows.append(dict(zip(columns, [str(v) for v in r])))
        cursor.close()
        conn.close()

        return {
            "question": question,
            "generated_sql": generated_sql,
            "results": rows
        }
    except requests.exceptions.RequestException:
        return {
            "question": question,
            "generated_sql": None,
            "error": "Ollama is not reachable.",
            "results": []
        }
    except oracledb.Error as e:
        return {
            "question": question,
            "generated_sql": generated_sql,
            "error": f"SQL execution error: {str(e)}",
            "results": []
        }


# ===========================================================================
# USE CASE D: PROPERTY GRAPH — Fraud, Influence, Risk (SQL/PGQ)
# ===========================================================================

@app.post("/api/graph/fraud-scan")
async def graph_fraud_scan(request: Request):
    """
    Find accounts sharing devices or IP addresses with a target player.
    Uses SQL/PGQ pattern matching on the seerplay_graph property graph.
    Body: { "player_id": "SPR-0300" }
    """
    body = await request.json()
    player_id = body.get("player_id")
    if not player_id:
        raise HTTPException(status_code=400, detail="Provide 'player_id'.")

    conn = get_conn()
    cursor = conn.cursor()
    try:
        # Find players sharing devices with the target
        cursor.execute("""
            SELECT DISTINCT gt.linked_player, gt.linked_persona, gt.linked_ltv,
                   gt.shared_device, gt.device_type
            FROM GRAPH_TABLE ( seerplay_graph
              MATCH (p1 IS players) -[e1 IS uses_device]-> (d IS devices)
                    <-[e2 IS uses_device]- (p2 IS players)
              WHERE p1.player_id = :target AND p1.player_id != p2.player_id
              COLUMNS (
                p2.player_id   AS linked_player,
                p2.persona     AS linked_persona,
                p2.ltv         AS linked_ltv,
                d.device_id    AS shared_device,
                d.device_type  AS device_type
              )
            ) gt
            ORDER BY gt.shared_device
        """, target=player_id)

        device_links = []
        for r in cursor.fetchall():
            risk = "High" if r[1] == "High-Churn" else "Medium" if r[1] == "VIP" else "Low"
            device_links.append({
                "player_id": r[0], "persona": r[1], "ltv": float(r[2]),
                "shared_device": r[3], "device_type": r[4], "link_type": "device",
                "fraud_risk": risk
            })

        # Find players sharing IPs with the target
        cursor.execute("""
            SELECT DISTINCT gt.linked_player, gt.linked_persona, gt.linked_ltv,
                   gt.shared_ip, gt.ip_address, gt.country
            FROM GRAPH_TABLE ( seerplay_graph
              MATCH (p1 IS players) -[e1 IS logs_from_ip]-> (ip IS ip_addrs)
                    <-[e2 IS logs_from_ip]- (p2 IS players)
              WHERE p1.player_id = :target AND p1.player_id != p2.player_id
              COLUMNS (
                p2.player_id    AS linked_player,
                p2.persona      AS linked_persona,
                p2.ltv          AS linked_ltv,
                ip.ip_id        AS shared_ip,
                ip.ip_address   AS ip_address,
                ip.country      AS country
              )
            ) gt
            ORDER BY gt.shared_ip
        """, target=player_id)

        ip_links = []
        for r in cursor.fetchall():
            risk = "High" if r[1] == "High-Churn" else "Medium" if r[1] == "VIP" else "Low"
            ip_links.append({
                "player_id": r[0], "persona": r[1], "ltv": float(r[2]),
                "shared_ip": r[3], "ip_address": r[4], "country": r[5],
                "link_type": "ip",
                "fraud_risk": risk
            })

        # Determine risk level
        unique_linked = set(l["player_id"] for l in device_links + ip_links)
        risk_level = "low"
        if len(unique_linked) >= 5:
            risk_level = "critical"
        elif len(unique_linked) >= 2:
            risk_level = "high"
        elif len(unique_linked) >= 1:
            risk_level = "medium"

        return {
            "target_player": player_id,
            "risk_level": risk_level,
            "total_linked_accounts": len(unique_linked),
            "device_links": device_links,
            "ip_links": ip_links
        }
    except oracledb.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


# ===========================================================================
# USE CASE E: SIMULATION FORGE — "What If" Scenario Builder
# ===========================================================================
@app.post("/api/simulation/run")
async def run_simulation(request: Request):
    """
    Simulate a VIP Bonus Drop scenario.
    Body: {
        "global_rtp": 96.5,
        "retention_bonus_freq": "HIGH_VELOCITY",
        "min_bet": 25.00
    }
    Returns projected GGR lift, risk factor, and graph data points.
    """
    body = await request.json()
    rtp = float(body.get("global_rtp", 96.5))
    bonus_freq = body.get("retention_bonus_freq", "MEDIUM")
    min_bet = float(body.get("min_bet", 25.0))

    # --- SIMULATION LOGIC (Deterministic for Demo) ---
    
    # Base params
    base_ggr = 500000  # Weekly GGR baseline
    
    # 1. RTP Impact: Higher RTP -> Lower Margin % but potentially Higher Volume (Volume elasticity)
    # Standard RTP 95%. Each 1% increase reduces margin by ~20% relative, but increases volume by 15%
    rtp_delta = rtp - 95.0
    margin_mult = max(0.1, 1.0 - (rtp_delta * 0.2)) # Margin drops as RTP goes up
    volume_mult = 1.0 + (rtp_delta * 0.15)         # Volume goes up as RTP goes up
    
    # 2. Bonus Frequency Impact: 
    # HIGH_VELOCITY -> Big cost, big retention lift
    # LOW -> Low cost, low lift
    bonus_cost = 0
    retention_lift = 1.0
    
    if bonus_freq == "EXTREME":
        bonus_cost = 50000
        retention_lift = 1.25
    elif bonus_freq == "HIGH_VELOCITY":
        bonus_cost = 25000
        retention_lift = 1.15
    elif bonus_freq == "MEDIUM":
        bonus_cost = 10000
        retention_lift = 1.05
    else: # LOW
        bonus_cost = 2000
        retention_lift = 1.01

    # 3. Min Bet Threshold:
    # Higher threshold -> excludes low value players (Lower Volume) but increases Avg Bet Value
    # $10 baseline.
    threshold_ratio = min_bet / 10.0
    # Volume drops, but value per player increases. Net effect usually positive up to a point, then negative.
    # Modeled as a parabola. Optimal around $25-$50 for VIPs.
    segment_value_mult = 1.0 + (threshold_ratio * 0.1) - (threshold_ratio * threshold_ratio * 0.002)

    # Calculate Final GGR
    projected_ggr = (base_ggr * volume_mult * margin_mult * retention_lift * segment_value_mult) - bonus_cost
    lift = projected_ggr - base_ggr
    
    # Risk Factor Calculation
    # High RTP + High Bonus + Low Threshold = HIGH RISK (Bleeding money)
    # Low RTP + Low Bonus + High Threshold = LOW RISK (Safe but maybe low lift)
    risk_score = 0
    if rtp > 97.0: risk_score += 3
    if bonus_freq == "EXTREME": risk_score += 3
    elif bonus_freq == "HIGH_VELOCITY": risk_score += 2
    if min_bet < 10.0: risk_score += 1

    risk_label = "LOW"
    if risk_score >= 5: risk_label = "CRITICAL"
    elif risk_score >= 3: risk_label = "MODERATE"
    
    # Generate Graph Data (Projected Accumulation over 7 days)
    graph_points = []
    current_val = 0
    for day in range(1, 8):
        # Add some randomness/curve
        daily_lift = (lift / 7) * (0.8 + (day * 0.05)) # Ramps up
        current_val += daily_lift
        graph_points.append({"day": f"Day {day}", "value": round(current_val, 2)})

    return {
        "projected_lift": round(lift, 2),
        "risk_factor": risk_label,
        "details": {
            "volume_impact": f"{round((volume_mult-1)*100)}%",
            "margin_impact": f"{round((margin_mult-1)*100)}%",
            "cost": bonus_cost
        },
        "graph_data": graph_points
    }


@app.post("/api/graph/influence")
async def graph_influence(request: Request):
    """
    Map a player's referral network (who they referred, and who those people referred).
    Uses SQL/PGQ reachability on the seerplay_graph property graph.
    Body: { "player_id": "SPR-0810" }
    """
    body = await request.json()
    player_id = body.get("player_id")
    if not player_id:
        raise HTTPException(status_code=400, detail="Provide 'player_id'.")

    conn = get_conn()
    cursor = conn.cursor()
    try:
        # Direct referrals (depth 1)
        cursor.execute("""
            SELECT gt.referee, gt.referee_persona, gt.referee_ltv, gt.ref_date
            FROM GRAPH_TABLE ( seerplay_graph
              MATCH (p1 IS players) -[r IS referrals]-> (p2 IS players)
              WHERE p1.player_id = :target
              COLUMNS (
                p2.player_id   AS referee,
                p2.persona     AS referee_persona,
                p2.ltv         AS referee_ltv,
                r.referral_date AS ref_date
              )
            ) gt
            ORDER BY gt.ref_date
        """, target=player_id)

        direct_referrals = []
        for r in cursor.fetchall():
            direct_referrals.append({
                "player_id": r[0], "persona": r[1], "ltv": float(r[2]),
                "referral_date": str(r[3]), "depth": 1
            })

        # Second-degree referrals (depth 2)
        cursor.execute("""
            SELECT gt.final_player, gt.final_persona, gt.final_ltv, gt.via_player
            FROM GRAPH_TABLE ( seerplay_graph
              MATCH (p1 IS players) -[r1 IS referrals]-> (p2 IS players)
                    -[r2 IS referrals]-> (p3 IS players)
              WHERE p1.player_id = :target
              COLUMNS (
                p3.player_id   AS final_player,
                p3.persona     AS final_persona,
                p3.ltv         AS final_ltv,
                p2.player_id   AS via_player
              )
            ) gt
        """, target=player_id)

        indirect_referrals = []
        for r in cursor.fetchall():
            indirect_referrals.append({
                "player_id": r[0], "persona": r[1], "ltv": float(r[2]),
                "via": r[3], "depth": 2
            })

        # Also check: who referred THIS player?
        cursor.execute("""
            SELECT gt.referrer, gt.referrer_persona
            FROM GRAPH_TABLE ( seerplay_graph
              MATCH (p1 IS players) -[r IS referrals]-> (p2 IS players)
              WHERE p2.player_id = :target
              COLUMNS (
                p1.player_id   AS referrer,
                p1.persona     AS referrer_persona
              )
            ) gt
        """, target=player_id)

        referred_by = []
        for r in cursor.fetchall():
            referred_by.append({"player_id": r[0], "persona": r[1]})

        total_network = len(direct_referrals) + len(indirect_referrals)
        total_ltv = sum(r["ltv"] for r in direct_referrals + indirect_referrals)

        return {
            "player_id": player_id,
            "influence_score": total_network,
            "network_ltv": round(total_ltv, 2),
            "referred_by": referred_by,
            "direct_referrals": direct_referrals,
            "indirect_referrals": indirect_referrals
        }
    except oracledb.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


@app.post("/api/graph/risk-propagation")
async def graph_risk_propagation(request: Request):
    """
    Check if a player is connected to at-risk players (self-excluded, flagged)
    via shared devices or IPs. Assesses "risk contagion."
    Body: { "player_id": "SPR-0042" }
    """
    body = await request.json()
    player_id = body.get("player_id")
    if not player_id:
        raise HTTPException(status_code=400, detail="Provide 'player_id'.")

    conn = get_conn()
    cursor = conn.cursor()
    try:
        # Find players sharing devices with the target who also have risk events
        cursor.execute("""
            SELECT DISTINCT gt.risky_player, gt.risky_persona,
                   gt.shared_device, gt.device_type
            FROM GRAPH_TABLE ( seerplay_graph
              MATCH (p1 IS players) -[e1 IS uses_device]-> (d IS devices)
                    <-[e2 IS uses_device]- (p2 IS players)
              WHERE p1.player_id = :target AND p1.player_id != p2.player_id
              COLUMNS (
                p2.player_id   AS risky_player,
                p2.persona     AS risky_persona,
                d.device_id    AS shared_device,
                d.device_type  AS device_type
              )
            ) gt
            WHERE gt.risky_player IN (SELECT player_id FROM risk_events)
        """, target=player_id)

        risky_device_links = []
        for r in cursor.fetchall():
            risky_device_links.append({
                "player_id": r[0], "persona": r[1],
                "via_device": r[2], "device_type": r[3]
            })

        # Find players sharing IPs with the target who also have risk events
        cursor.execute("""
            SELECT DISTINCT gt.risky_player, gt.risky_persona,
                   gt.shared_ip, gt.ip_address
            FROM GRAPH_TABLE ( seerplay_graph
              MATCH (p1 IS players) -[e1 IS logs_from_ip]-> (ip IS ip_addrs)
                    <-[e2 IS logs_from_ip]- (p2 IS players)
              WHERE p1.player_id = :target AND p1.player_id != p2.player_id
              COLUMNS (
                p2.player_id    AS risky_player,
                p2.persona      AS risky_persona,
                ip.ip_id        AS shared_ip,
                ip.ip_address   AS ip_address
              )
            ) gt
            WHERE gt.risky_player IN (SELECT player_id FROM risk_events)
        """, target=player_id)

        risky_ip_links = []
        for r in cursor.fetchall():
            risky_ip_links.append({
                "player_id": r[0], "persona": r[1],
                "via_ip": r[2], "ip_address": r[3]
            })

        # Get the actual risk events for connected at-risk players
        all_risky = set(l["player_id"] for l in risky_device_links + risky_ip_links)
        risk_details = {}
        if all_risky:
            placeholders = ",".join([f":{i+1}" for i in range(len(all_risky))])
            cursor.execute(
                f"SELECT player_id, risk_type, event_date, severity FROM risk_events WHERE player_id IN ({placeholders})",
                list(all_risky)
            )
            for r in cursor.fetchall():
                pid = r[0]
                if pid not in risk_details:
                    risk_details[pid] = []
                risk_details[pid].append({
                    "risk_type": r[1], "event_date": str(r[2]), "severity": int(r[3])
                })

        # Check the target's own risk events
        cursor.execute("SELECT risk_type, event_date, severity FROM risk_events WHERE player_id = :1", [player_id])
        own_risks = [{"risk_type": r[0], "event_date": str(r[1]), "severity": int(r[2])} for r in cursor.fetchall()]

        # Calculate contagion score
        total_risky_neighbors = len(all_risky)
        max_severity = max(
            (s for details in risk_details.values() for s in [d["severity"] for d in details]),
            default=0
        )
        contagion_score = min(10, total_risky_neighbors * 2 + max_severity)

        risk_level = "low"
        if contagion_score >= 8:
            risk_level = "critical"
        elif contagion_score >= 5:
            risk_level = "high"
        elif contagion_score >= 2:
            risk_level = "medium"

        recommendation = "No action needed."
        if risk_level == "critical":
            recommendation = "🚨 Immediate review required. Player is heavily connected to flagged accounts."
        elif risk_level == "high":
            recommendation = "⚠️ Proactive responsible gaming check-in recommended."
        elif risk_level == "medium":
            recommendation = "📋 Monitor player activity for emerging patterns."

        return {
            "player_id": player_id,
            "risk_level": risk_level,
            "contagion_score": contagion_score,
            "recommendation": recommendation,
            "own_risk_events": own_risks,
            "connected_risks": {
                "total_risky_neighbors": total_risky_neighbors,
                "via_device": risky_device_links,
                "via_ip": risky_ip_links,
                "risk_details": risk_details
            }
        }
    except oracledb.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


# ===========================================================================
# STATS ENDPOINT (for dashboard cards)
# ===========================================================================
@app.get("/api/stats")
def get_stats():
    """Dashboard summary stats."""
    conn = get_conn()
    cursor = conn.cursor()
    try:
        stats = {}

        cursor.execute("SELECT COUNT(*) FROM players")
        stats["total_players"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM player_sessions")
        stats["total_sessions"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM session_events")
        stats["total_events"] = cursor.fetchone()[0]

        cursor.execute("SELECT persona, COUNT(*), ROUND(AVG(ltv),2) FROM players GROUP BY persona ORDER BY persona")
        stats["personas"] = [
            {"persona": r[0], "count": r[1], "avg_ltv": float(r[2])} for r in cursor.fetchall()
        ]

        cursor.execute("SELECT SUM(total_bets), SUM(total_wins), SUM(total_losses) FROM player_sessions")
        row = cursor.fetchone()
        stats["total_bets"] = float(row[0] or 0)
        stats["total_wins"] = float(row[1] or 0)
        stats["total_losses"] = float(row[2] or 0)

        return stats
    except oracledb.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ===========================================================================
# USE CASE G: SPECTRO — Player Intelligence Dossier
# ===========================================================================

@app.post("/api/spectro/dossier")
async def get_player_dossier(request: Request):
    """
    Retrieve comprehensive intelligence dossier for a player.
    Aggregates:
    - Profile (Name, Tier, ID)
    - Financial Snapshot (LTV, GGR, Win/Loss)
    - Session History (Deep-Dive)
    - Host Intel (Staff Notes)
    - Retention & Risk Status
    """
    try:
        body = await request.json()
        player_id = body.get("player_id")
        if not player_id:
            return JSONResponse({"error": "Missing player_id"}, status_code=400)

        conn = get_conn()
        cursor = conn.cursor()

        # 0. Fetch Player Profile
        cursor.execute("SELECT username, persona, ltv FROM players WHERE player_id = :1", [player_id])
        player_row = cursor.fetchone()
        
        real_name = f"Player {player_id}"
        real_tier = "UNKNOWN"
        
        if player_row:
            real_name = player_row[0]  # username
            persona = player_row[1]
            # Map persona to tier
            if persona == "VIP":
                real_tier = "OBSIDIAN TIER"
            elif persona == "Casual":
                real_tier = "GOLD TIER"
            else:
                real_tier = "IRON TIER"

        # 1. Profile & Financials (Aggregate from DB)
        # Check if player exists in players table (if it exists) or distinct sessions
        # For this demo, we'll derive existence from sessions or valid ID format.
        
        # Calculate Lifetime Value (Sum of (bet_amount - win_amount))
        cursor.execute("""
            SELECT 
                SUM(s.total_bets) as total_bet,
                SUM(s.total_wins) as total_win,
                COUNT(*) as total_sessions,
                MAX(s.session_start) as last_seen
            FROM player_sessions s
            WHERE s.player_id = :1
        """, [player_id])
        
        fin_row = cursor.fetchone()
        
        if not fin_row or not fin_row[0]:
            # Mock data if player has no history yet or ID is new
            total_bet = 0.0
            total_win = 0.0
            session_count = 0
            # If it's a known demo ID, we might want to ensure some data, 
            # but let's stick to DB reality + fallbacks.
            # actually, let's look up the persona if possible.
        else:
            total_bet = float(fin_row[0])
            total_win = float(fin_row[1])
            session_count = int(fin_row[2])
            
        ggr = total_bet - total_win
        ltv = ggr # Simplified LTV = GGR for this demo
        
        # Calculate Win/Loss Ratio (Session level)
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN total_wins > total_bets THEN 1 ELSE 0 END) as wins,
                COUNT(*) as total
            FROM player_sessions
            WHERE player_id = :1
        """, [player_id])
        wl_row = cursor.fetchone()
        if wl_row and wl_row[1] > 0:
            wins = int(wl_row[0])
            total_games = int(wl_row[1])
            losses = total_games - wins
            win_pct = int((wins / total_games) * 100) if total_games > 0 else 0
            loss_pct = 100 - win_pct
        else:
            win_pct = 42 # Default/Mock
            loss_pct = 58

        # 2. Session History (Deep Dive)
        # Fetch last 5 sessions
        cursor.execute("""
            SELECT 
                to_char(session_start, 'YYYY-MM-DD HH24:MI'),
                game_type,
                total_bets,
                total_wins,
                aggression_score,
                variance_level
            FROM player_sessions
            WHERE player_id = :1
            ORDER BY session_start DESC
            FETCH FIRST 5 ROWS ONLY
        """, [player_id])
        
        sessions = []
        rows = cursor.fetchall()
        for r in rows:
            sessions.append({
                "timestamp": r[0],
                "game": r[1],
                "wager": float(r[2]),
                "payout": float(r[3]),
                "aggression": int(r[4] or 0), 
                "variance": r[5] or "Low"
            })

        # 3. Host Intel (Real DB Notes)
        cursor.execute("""
            SELECT note_text, author, 
                   round((sysdate - created_at) * 24 * 60) as mins_ago,
                   note_type, alert_type
            FROM player_notes
            WHERE player_id = :1
            ORDER BY created_at DESC
            FETCH FIRST 10 ROWS ONLY
        """, [player_id])
        
        staff_notes = []
        for r in cursor.fetchall():
            mins = int(r[2])
            if mins < 60:
                time_str = f"{mins}m ago"
            elif mins < 1440:
                time_str = f"{mins // 60}h ago"
            else:
                time_str = f"{mins // 1440}d ago"
                
            staff_notes.append({
                "text": r[0],
                "author": r[1],
                "time": time_str,
                "type": r[3] if r[3] else "NOTE",
                "alert_type": r[4]
            })
        
        # 4. Retention Health (Real OML Prediction)
        # We use the View created in seeder.py which matches the model signature
        try:
            cursor.execute("""
                SELECT 
                    ROUND(PREDICTION_PROBABILITY(CHURN_PREDICTION_RF, 1 USING *) * 100, 1) as churn_prob
                FROM v_player_churn_train
                WHERE player_id = :1
            """, [player_id])
            oml_row = cursor.fetchone()
            churn_risk = oml_row[0] if oml_row else 0
        except Exception as e:
            print(f"OML Error: {e}")
            churn_risk = 0 # Fallback

        if churn_risk > 70:
            health_status = "CRITICAL"
            health_color = "rose"
        elif churn_risk > 30:
            health_status = "AT RISK"
            health_color = "amber"
        else:
            health_status = "STABLE"
            health_color = "emerald"

        cursor.close()
        conn.close()

        # Construct Response
        return JSONResponse({
            "player_id": player_id,
            "profile": {
                "name": real_name, 
                "tier": real_tier,
                "tier_code": f"#{player_id.replace('SPR-', '')}-X9",
                "avatar": "/static/avatars/default.png" # Frontend will handle this 
            },
            "financials": {
                "ggr": ggr,
                "ltv": ltv,
                "win_pct": win_pct,
                "loss_pct": loss_pct,
                "trend_l30d": "+12%" # Mocked trend
            },
            "sessions": sessions,
            "intel": staff_notes,
            "retention": {
                "score": 100 - churn_risk,
                "status": health_status,
                "color": health_color
            }
        })

    except Exception as e:
        print(f"Error in dossier: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

