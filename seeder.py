"""
SeerPlay Intelligence Hub — Data Seeder
======================================
Generates 1,000 player profiles with Gaussian-clustered behavior vectors,
realistic session history, clickstream events, and pre-computed regression
coefficients for the What-If scenario model.
"""

import oracledb
import json
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression
import random
import datetime
import sys
import array

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DSN      = "localhost:1521/FREEPDB1"
USER     = "seerplay"
PASSWORD = "SeerPlay123"

NUM_PLAYERS   = 1000
VECTOR_DIMS   = 768
PERSONAS      = ["VIP", "Casual", "High-Churn"]
PERSONA_COUNTS = [50, 200, 750]

GAME_TYPES = ["slots", "blackjack", "roulette", "poker", "baccarat", "craps"]
DEVICES    = ["mobile", "desktop", "tablet", "vr"]
EVENT_TYPES = ["spin", "bet", "win", "loss", "bonus_trigger", "cashout", "pause"]

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Generate Gaussian-Clustered Behavior Vectors
# ---------------------------------------------------------------------------
print(f"[Seeder] Generating {NUM_PLAYERS} players across {len(PERSONAS)} personas...")

# First 10 dims encode meaningful behavioral features:
#   [avg_bet, session_dur, loss_streak_max, win_rate, game_diversity,
#    time_of_day, bet_variance, cashout_freq, bonus_usage, device_switches]
#   [avg_bet, session_dur, loss_streak_max, win_rate, game_diversity,
#    time_of_day, bet_variance, cashout_freq, bonus_usage, device_switches]
# Remaining dims are noise padding for embedding space compatibility.

# ---------------------------------------------------------------------------
# 0. Clean Function to Wipe Tables (Reverse Dependency Order)
# ---------------------------------------------------------------------------
def clean_database(cursor):
    print("[Seeder] Cleaning existing data (Truncating tables)...")
    print("[Seeder] Cleaning existing data (Truncating tables)...")
    tables = [
        "risk_events", "referrals", "plays_game", "logs_from_ip", "uses_device",
        "player_notes", "session_events", "player_sessions", "model_centroids", "whatif_model",
        "games", "ip_addrs", "devices", "players"
    ]
    for table in tables:
        try:
            cursor.execute(f"TRUNCATE TABLE {table}")
        except oracledb.DatabaseError as e:
            # Table might not exist yet if init.sql failed or first run
            print(f"  ⚠ Could not truncate {table}: {e}")

# Call clean before generating data? No, call it inside the main block.


# Define cluster centers with meaningful feature separation
centers = np.zeros((3, VECTOR_DIMS))
# VIP center: high bets, long sessions, moderate loss streaks, high diversity
centers[0, :10] = [8.0, 9.0, 3.0, 0.55, 0.9, 14.0, 2.0, 0.3, 0.8, 0.2]
# Casual center: low bets, medium sessions, low loss streaks
centers[1, :10] = [2.0, 4.0, 1.0, 0.50, 0.4, 20.0, 0.5, 0.6, 0.3, 0.1]
# High-Churn center: mid bets, short sessions, high loss streaks, low diversity
centers[2, :10] = [5.0, 2.0, 7.0, 0.35, 0.2, 2.0, 4.0, 0.1, 0.9, 0.5]

X, y = make_blobs(
    n_samples=PERSONA_COUNTS,
    n_features=VECTOR_DIMS,
    centers=centers,
    cluster_std=1.5,
    random_state=42
)

# Compute centroids
centroids = {}
for i, persona in enumerate(PERSONAS):
    mask = (y == i)
    centroids[persona] = np.mean(X[mask], axis=0)

# ---------------------------------------------------------------------------
# 2. Generate Realistic Session + Event Data
# ---------------------------------------------------------------------------
print("[Seeder] Generating session and event data...")

players_data = []
sessions_data = []
events_data = []

session_counter = 1
base_date = datetime.datetime(2025, 6, 1)

for idx in range(NUM_PLAYERS):
    persona = PERSONAS[y[idx]]
    player_id = f"SPR-{idx:04d}"
    vector = X[idx].tolist()

    # Player-level attributes vary by persona
    if persona == "VIP":
        ltv = round(random.uniform(5000, 50000), 2)
        num_sessions = random.randint(10, 20)
        username = f"whale_{idx}"
    elif persona == "Casual":
        ltv = round(random.uniform(100, 2000), 2)
        num_sessions = random.randint(5, 12)
        username = f"player_{idx}"
    else:  # High-Churn
        ltv = round(random.uniform(10, 500), 2)
        num_sessions = random.randint(2, 6)
        username = f"churn_{idx}"

    signup_offset = random.randint(0, 180)
    signup_date = base_date + datetime.timedelta(days=signup_offset)

    players_data.append((
        player_id, username, signup_date, persona, ltv, "ACTIVE"
    ))

    # Sessions for this player
    for s in range(num_sessions):
        sid = session_counter
        session_counter += 1

        session_start = signup_date + datetime.timedelta(
            days=random.randint(0, 60),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        duration_mins = random.randint(5, 180)
        session_end = session_start + datetime.timedelta(minutes=duration_mins)
        game = random.choice(GAME_TYPES)
        device = random.choice(DEVICES)

        if persona == "VIP":
            total_bets = round(random.uniform(500, 10000), 2)
            total_wins = round(total_bets * random.uniform(0.4, 0.7), 2)
        elif persona == "Casual":
            total_bets = round(random.uniform(10, 200), 2)
            total_wins = round(total_bets * random.uniform(0.3, 0.6), 2)
        else:
            total_bets = round(random.uniform(50, 1000), 2)
            total_wins = round(total_bets * random.uniform(0.1, 0.4), 2)

        total_losses = round(total_bets - total_wins, 2)

        # Mock Aggression/Variance (now real data)
        aggression_score = random.randint(30, 95)
        variance_level = random.choice(["Low", "Med", "High", "Extreme"])

        # Convert vector list to array.array('f', ...) for oracledb VECTOR support
        vector_arr = array.array('f', vector)
        sessions_data.append((
            sid, player_id, session_start, session_end,
            total_bets, total_wins, total_losses,
            game, device, vector_arr,
            aggression_score, variance_level
        ))

        # 3-15 events per session
        num_events = random.randint(3, 15)
        for e_idx in range(num_events):
            event_ts = session_start + datetime.timedelta(
                seconds=random.randint(0, duration_mins * 60)
            )
            event_type = random.choice(EVENT_TYPES)
            event_data = json.dumps({
                "amount": round(random.uniform(1, 500), 2),
                "game": game,
                "multiplier": round(random.uniform(0.5, 10.0), 2),
                "streak": random.randint(0, 10),
                "auto_spin": random.choice([True, False])
            })
            events_data.append((sid, player_id, event_ts, event_type, event_data))

print(f"[Seeder] Generated {len(players_data)} players, {len(sessions_data)} sessions, {len(events_data)} events")

# ---------------------------------------------------------------------------
# 3. Build What-If Regression Model
# ---------------------------------------------------------------------------
print("[Seeder] Training What-If regression model...")

# Feature engineering from sessions for regression
# Features: avg session duration, avg bet size, max loss streak proxy, break_minutes (synthetic)
reg_features = []
reg_targets = []

for idx in range(NUM_PLAYERS):
    player_id = f"SPR-{idx:04d}"
    persona = PERSONAS[y[idx]]

    # Simulate features based on persona characteristics
    avg_session_dur = centers[y[idx], 1] + random.gauss(0, 0.5)
    avg_bet = centers[y[idx], 0] + random.gauss(0, 0.3)
    loss_streak = max(0, centers[y[idx], 2] + random.gauss(0, 0.5))
    break_mins = random.uniform(0, 30)  # synthetic intervention variable

    ltv = players_data[idx][4]

    reg_features.append([avg_session_dur, avg_bet, loss_streak, break_mins])
    reg_targets.append(ltv)

reg_features = np.array(reg_features)
reg_targets = np.array(reg_targets)

model = LinearRegression()
model.fit(reg_features, reg_targets)

print(f"[Seeder] Regression R²: {model.score(reg_features, reg_targets):.4f}")
print(f"[Seeder] Coefficients: {model.coef_}")
print(f"[Seeder] Intercept: {model.intercept_:.4f}")

# ---------------------------------------------------------------------------
# 4. Generate Property Graph Data
# ---------------------------------------------------------------------------
print("[Seeder] Generating Property Graph data...")

# --- Games Catalog ---
GAME_CATALOG = [
    ("GAME-001", "Viking Voyage",    "adventure"),
    ("GAME-002", "Norse Thunder",    "mythology"),
    ("GAME-003", "Golden Axe Slots", "slots"),
    ("GAME-004", "Rune Roulette",    "table"),
    ("GAME-005", "Shield Wall BJ",   "table"),
    ("GAME-006", "Odin's Fortune",   "mythology"),
    ("GAME-007", "Frost Giant Poker", "table"),
    ("GAME-008", "Valhalla Spins",   "slots"),
    ("GAME-009", "Ragnarok Rush",    "adventure"),
    ("GAME-010", "Bifrost Baccarat", "table"),
    ("GAME-011", "Dragon's Hoard",   "adventure"),
    ("GAME-012", "Lucky Leprechaun", "luck"),
    ("GAME-013", "Treasure Island",  "adventure"),
    ("GAME-014", "Mega Moolah",      "slots"),
    ("GAME-015", "Speed Roulette",   "table"),
    ("GAME-016", "Neon Nights",      "modern"),
    ("GAME-017", "Pharaoh's Gold",   "mythology"),
    ("GAME-018", "Cosmic Cash",      "modern"),
]

# --- Devices (200 unique, some shared for fraud) ---
devices_data = []
for d in range(200):
    dtype = random.choice(["mobile", "desktop", "tablet", "smartwatch"])
    fp = f"FP-{random.randint(100000, 999999)}"
    devices_data.append((f"DEV-{d:04d}", dtype, fp))

# --- IP Addresses (150 unique, some shared) ---
ips_data = []
for ip_idx in range(150):
    country = random.choice(["GB", "IE", "MT", "GI", "SE", "DK", "NO", "FI", "DE"])
    ip_str = f"{random.randint(10,200)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    ips_data.append((f"IP-{ip_idx:04d}", ip_str, country))

# --- Player ↔ Device edges ---
uses_device_data = []
# Normal: each player uses 1-3 unique devices
for idx in range(NUM_PLAYERS):
    player_id = f"SPR-{idx:04d}"
    n_devices = random.randint(1, 3)
    chosen = random.sample(range(100, 200), n_devices)  # devices 100-199 for normal
    for d in chosen:
        uses_device_data.append((player_id, f"DEV-{d:04d}", random.randint(1, 20)))

# Fraud Cluster 1: 8 accounts share DEV-0001 and DEV-0002
FRAUD_CLUSTER_1 = [f"SPR-{i:04d}" for i in range(300, 308)]
for pid in FRAUD_CLUSTER_1:
    uses_device_data.append((pid, "DEV-0001", random.randint(5, 30)))
    uses_device_data.append((pid, "DEV-0002", random.randint(1, 5)))

# Fraud Cluster 2: 6 accounts share DEV-0005
FRAUD_CLUSTER_2 = [f"SPR-{i:04d}" for i in range(500, 506)]
for pid in FRAUD_CLUSTER_2:
    uses_device_data.append((pid, "DEV-0005", random.randint(10, 40)))

# Fraud Cluster 3: 5 accounts share DEV-0010 and DEV-0011
FRAUD_CLUSTER_3 = [f"SPR-{i:04d}" for i in range(700, 705)]
for pid in FRAUD_CLUSTER_3:
    uses_device_data.append((pid, "DEV-0010", random.randint(3, 15)))
    uses_device_data.append((pid, "DEV-0011", random.randint(2, 8)))

# --- Player ↔ IP edges ---
logs_from_ip_data = []
# Normal: each player uses 1-2 IPs
for idx in range(NUM_PLAYERS):
    player_id = f"SPR-{idx:04d}"
    n_ips = random.randint(1, 2)
    chosen = random.sample(range(50, 150), n_ips)
    first = base_date - datetime.timedelta(days=random.randint(30, 180))
    last = base_date + datetime.timedelta(days=random.randint(0, 30))
    for ip in chosen:
        logs_from_ip_data.append((player_id, f"IP-{ip:04d}", first, last))

# Fraud clusters share IPs too
for pid in FRAUD_CLUSTER_1:
    logs_from_ip_data.append((pid, "IP-0001", base_date - datetime.timedelta(days=60), base_date))
for pid in FRAUD_CLUSTER_2:
    logs_from_ip_data.append((pid, "IP-0002", base_date - datetime.timedelta(days=45), base_date))
for pid in FRAUD_CLUSTER_3:
    logs_from_ip_data.append((pid, "IP-0003", base_date - datetime.timedelta(days=30), base_date))

# --- Player → Game edges ---
plays_game_data = []
for idx in range(NUM_PLAYERS):
    player_id = f"SPR-{idx:04d}"
    n_games = random.randint(2, 8)
    chosen = random.sample(GAME_CATALOG, n_games)
    for game in chosen:
        plays_game_data.append((
            player_id, game[0],
            round(random.uniform(10, 5000), 2),
            random.randint(1, 50)
        ))

# --- Referral Chains (10% of players, depth 1-4) ---
referrals_data = []
referral_sources = random.sample(range(NUM_PLAYERS), 30)  # 30 influencers
for src_idx in referral_sources:
    src_id = f"SPR-{src_idx:04d}"
    # Each influencer refers 2-8 players
    n_referees = random.randint(2, 8)
    potential = [i for i in range(NUM_PLAYERS) if i != src_idx]
    referees = random.sample(potential, min(n_referees, len(potential)))
    for ref_idx in referees:
        ref_id = f"SPR-{ref_idx:04d}"
        ref_date = base_date - datetime.timedelta(days=random.randint(10, 180))
        referrals_data.append((src_id, ref_id, ref_date))
        # Second-level referral (depth 2) — 30% chance
        if random.random() < 0.3:
            sub_idx = random.choice([i for i in range(NUM_PLAYERS) if i not in (src_idx, ref_idx)])
            referrals_data.append((ref_id, f"SPR-{sub_idx:04d}",
                                   ref_date + datetime.timedelta(days=random.randint(5, 30))))

# --- MEGA INFLUENCER SCENARIO (SPR-0900) ---
# SPR-0900 refers 5 "Lieutenants", who each refer 3 "Soldiers"
print("[Seeder] Injecting Mega Influencer (SPR-0900)...")
mega_id = "SPR-0900"
lieutenants = [f"SPR-{i:04d}" for i in range(901, 906)]
soldiers_start = 910
for lt_id in lieutenants:
    # 1. Mega refers Lieutenant
    r_date = base_date - datetime.timedelta(days=random.randint(60, 120))
    referrals_data.append((mega_id, lt_id, r_date))
    
    # 2. Lieutenant refers 3 Soldiers
    for i in range(3):
        soldier_id = f"SPR-{soldiers_start:04d}"
        soldiers_start += 1
        s_date = r_date + datetime.timedelta(days=random.randint(5, 40))
        referrals_data.append((lt_id, soldier_id, s_date))


# Deduplicate referrals (same pair shouldn't appear twice)
seen_refs = set()
unique_referrals = []
for r in referrals_data:
    key = (r[0], r[1])
    if key not in seen_refs and r[0] != r[1]:
        seen_refs.add(key)
        unique_referrals.append(r)
referrals_data = unique_referrals

# --- Player Notes (Host Intel) ---
# Generate 1-5 notes for ~20% of players + all VIPs + all High-Risks
notes_data = []
print("[Seeder] Generating host intel notes...")

NOTE_TEMPLATES = [
    ("Prefers quiet tables in the North Wing.", "NOTE", None),
    (" requested a specific dealer (Sarah J).", "NOTE", None),
    ("Celebrating a birthday/anniversary.", "NOTE", None),
    ("Reviewing comp status - potential upgrade.", "NOTE", None),
    ("Complained about drink service speed.", "NOTE", None),
    ("Likes to play multiple hands concurrently.", "NOTE", None),
    ("Asking about high-limit room access.", "NOTE", None),
]

ALERT_TEMPLATES = [
    ("Displayed aggressive betting patterns after loss.", "ALERT", "HIGH TILT SIGNS"),
    ("Rapid device switching detected.", "ALERT", "ACCOUNT SECURITY"),
    ("Chasing losses with increasing stakes.", "ALERT", "RESPONSIBLE GAMING"),
    ("Abusive language towards chat support.", "ALERT", "BEHAVIORAL FLAG"),
    ("Large withdrawal request pending approval.", "ALERT", "FINANCIAL ALERT"),
]

AUTHORS = ["Sarah J.", "Mike T.", "System AI", "Pit Boss", "VIP Host"]

for idx in range(NUM_PLAYERS):
    player_id = f"SPR-{idx:04d}"
    persona = PERSONAS[y[idx]]
    
    # VIPs always get notes
    # High churn get risk alerts
    # Casual 10% chance
    
    num_notes = 0
    if persona == "VIP":
        num_notes = random.randint(1, 4)
    elif persona == "High-Churn":
        num_notes = random.randint(0, 2)
    elif random.random() < 0.1:
        num_notes = 1
        
    for _ in range(num_notes):
        # 30% chance of alert, unless high churn (70%)
        is_alert = False
        if persona == "High-Churn" and random.random() < 0.7:
            is_alert = True
        elif random.random() < 0.3:
            is_alert = True
            
        if is_alert:
            tmpl = random.choice(ALERT_TEMPLATES)
            note_text = tmpl[0]
            note_type = tmpl[1]
            alert_type = tmpl[2]
            author = "System AI" if random.random() < 0.8 else random.choice(AUTHORS)
        else:
            tmpl = random.choice(NOTE_TEMPLATES)
            note_text = tmpl[0]
            note_type = tmpl[1]
            alert_type = None
            author = random.choice(AUTHORS)
            
        # Random time in last 30 days
        note_time = base_date - datetime.timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
        
        notes_data.append((
            player_id, note_text, author, note_time, note_type, alert_type
        ))

# --- Risk Events (5% of players) ---
RISK_TYPES = ["self_exclusion", "failed_deposit", "cooldown_request", "spending_limit_hit", "timeout"]
risk_events_data = []
at_risk_players = random.sample(range(NUM_PLAYERS), 50)  # 5% of 1000
for idx in at_risk_players:
    player_id = f"SPR-{idx:04d}"
    n_events = random.randint(1, 3)
    for _ in range(n_events):
        risk_events_data.append((
            player_id,
            random.choice(RISK_TYPES),
            base_date - datetime.timedelta(days=random.randint(0, 90)),
            random.randint(1, 5)
        ))

# Also mark fraud cluster members as at-risk
for pid in FRAUD_CLUSTER_1 + FRAUD_CLUSTER_2 + FRAUD_CLUSTER_3:
    risk_events_data.append((pid, "multi_account_flag", base_date, 5))

print(f"[Seeder] Graph data: {len(devices_data)} devices, {len(ips_data)} IPs, "
      f"{len(GAME_CATALOG)} games, {len(uses_device_data)} device-edges, "
      f"{len(logs_from_ip_data)} IP-edges, {len(plays_game_data)} game-edges, "
      f"{len(referrals_data)} referrals, {len(risk_events_data)} risk events")

# ---------------------------------------------------------------------------
# 5. Insert Everything into Oracle
# ---------------------------------------------------------------------------
print("[Seeder] Connecting to Oracle...")

import time

max_retries = 10
retry_delay = 5  # seconds

conn = None
for attempt in range(1, max_retries + 1):
    try:
        conn = oracledb.connect(user=USER, password=PASSWORD, dsn=DSN)
        print("[Seeder] Connected successfully.")
        break
    except oracledb.Error as e:
        print(f"[Seeder] ⚠ Connection failed (Attempt {attempt}/{max_retries}): {e}")
        if attempt < max_retries:
            print(f"[Seeder]   Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print("[Seeder] ❌ Max retries reached. Exiting.")
            raise

try:
    cursor = conn.cursor()
    clean_database(cursor)

    # -- Insert Players --
    print("[Seeder] Inserting players...")
    cursor.executemany(
        """INSERT INTO players (player_id, username, signup_date, persona, ltv, status)
           VALUES (:1, :2, :3, :4, :5, :6)""",
        players_data
    )

    # -- Insert Centroids --
    print("[Seeder] Inserting centroids...")
    for persona, vec in centroids.items():
        count = PERSONA_COUNTS[PERSONAS.index(persona)]
        avg_ltv_val = np.mean([p[4] for p in players_data if p[3] == persona])
        vec_arr = array.array('f', vec.tolist())
        cursor.execute(
            """INSERT INTO model_centroids (persona, centroid_vector, player_count, avg_ltv)
               VALUES (:1, :2, :3, :4)""",
            [persona, vec_arr, count, round(float(avg_ltv_val), 2)]
        )

    # -- Insert Sessions (individual inserts due to VECTOR column) --
    print("[Seeder] Inserting sessions...")
    for i, s in enumerate(sessions_data):
        cursor.execute(
            """INSERT INTO player_sessions
               (session_id, player_id, session_start, session_end,
                total_bets, total_wins, total_losses, game_type, device, behavior_vector,
                aggression_score, variance_level)
               VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12)""",
            list(s)
        )
        if (i + 1) % 500 == 0:
            print(f"  Inserted {i + 1}/{len(sessions_data)} sessions...")
            conn.commit()

    # -- Insert Events (batched — no VECTOR column, so executemany works) --
    print(f"[Seeder] Inserting {len(events_data)} events...")
    batch_size = 200
    for i in range(0, len(events_data), batch_size):
        batch = events_data[i:i+batch_size]
        cursor.executemany(
            """INSERT INTO session_events
               (session_id, player_id, event_ts, event_type, event_data)
               VALUES (:1, :2, :3, :4, :5)""",
            batch
        )

    # -- Insert Regression Model --
    print("[Seeder] Inserting What-If regression model...")
    cursor.execute(
        """INSERT INTO whatif_model
           (model_name, intercept, coeff_session_dur, coeff_avg_bet,
            coeff_loss_streak, coeff_break_mins, r_squared)
           VALUES (:1, :2, :3, :4, :5, :6, :7)""",
        [
            "LTV_Intervention_V1",
            round(float(model.intercept_), 4),
            round(float(model.coef_[0]), 6),
            round(float(model.coef_[1]), 6),
            round(float(model.coef_[2]), 6),
            round(float(model.coef_[3]), 6),
            round(float(model.score(reg_features, reg_targets)), 6)
        ]
    )

    conn.commit()

    # -- Insert Graph Node Tables --
    print("[Seeder] Inserting graph nodes (devices, IPs, games)...")
    cursor.executemany("INSERT INTO devices (device_id, device_type, fingerprint) VALUES (:1, :2, :3)", devices_data)
    cursor.executemany("INSERT INTO ip_addrs (ip_id, ip_address, country) VALUES (:1, :2, :3)", ips_data)
    cursor.executemany("INSERT INTO games (game_id, game_name, game_theme) VALUES (:1, :2, :3)", GAME_CATALOG)

    # -- Insert Graph Edge Tables --
    print("[Seeder] Inserting graph edges (device, IP, game, referrals, risk)...")
    cursor.executemany(
        "INSERT INTO uses_device (player_id, device_id, session_count) VALUES (:1, :2, :3)",
        uses_device_data
    )
    cursor.executemany(
        "INSERT INTO logs_from_ip (player_id, ip_id, first_seen, last_seen) VALUES (:1, :2, :3, :4)",
        logs_from_ip_data
    )
    cursor.executemany(
        "INSERT INTO plays_game (player_id, game_id, total_bets, session_count) VALUES (:1, :2, :3, :4)",
        plays_game_data
    )
    cursor.executemany(
        "INSERT INTO referrals (referrer_id, referee_id, referral_date) VALUES (:1, :2, :3)",
        referrals_data
    )
    cursor.executemany(
        "INSERT INTO risk_events (player_id, risk_type, event_date, severity) VALUES (:1, :2, :3, :4)",
        risk_events_data
    )

    print(f"[Seeder] Inserting {len(notes_data)} player notes...")
    cursor.executemany(
        "INSERT INTO player_notes (player_id, note_text, author, created_at, note_type, alert_type) VALUES (:1, :2, :3, :4, :5, :6)",
        notes_data
    )

    # ---------------------------------------------------------------------------
    # 6. Train OML Churn Model
    # ---------------------------------------------------------------------------
    print("[Seeder] Training OML Churn Prediction Model...")
    try:
        # 1. Create Training View (Aggregating player behavior)
        # We need to handle the case where the view already exists
        cursor.execute("BEGIN EXECUTE IMMEDIATE 'DROP VIEW v_player_churn_train'; EXCEPTION WHEN OTHERS THEN NULL; END;")
        
        # Aggregating session data for features
        cursor.execute("""
            CREATE VIEW v_player_churn_train AS
            SELECT 
                p.player_id,
                CASE WHEN p.persona = 'High-Churn' THEN 1 ELSE 0 END as is_churned,
                nvl(p.ltv, 0) as ltv,
                COUNT(s.session_id) as session_count,
                nvl(AVG(s.total_bets), 0) as avg_bet,
                nvl(SUM(s.total_losses), 0) as total_losses,
                nvl(SUM(s.total_wins), 0) as total_wins
            FROM players p
            LEFT JOIN player_sessions s ON p.player_id = s.player_id
            GROUP BY p.player_id, p.persona, p.ltv
        """)
        
        # 2. Cleanup existing model/settings
        cursor.execute("BEGIN DBMS_DATA_MINING.DROP_MODEL('CHURN_PREDICTION_RF'); EXCEPTION WHEN OTHERS THEN NULL; END;")
        cursor.execute("BEGIN EXECUTE IMMEDIATE 'DROP TABLE churn_model_settings'; EXCEPTION WHEN OTHERS THEN NULL; END;")
        
        # 3. Create Settings Table
        cursor.execute("CREATE TABLE churn_model_settings (setting_name VARCHAR2(30), setting_value VARCHAR2(4000))")
        cursor.execute("INSERT INTO churn_model_settings VALUES ('ALGO_NAME', 'ALGO_RANDOM_FOREST')")
        cursor.execute("INSERT INTO churn_model_settings VALUES ('PREP_AUTO', 'ON')")
        
        # 4. Train Model
        print("  → Starting DBMS_DATA_MINING.CREATE_MODEL (this may take a moment)...")
        cursor.execute("""
            BEGIN
                DBMS_DATA_MINING.CREATE_MODEL(
                    model_name          => 'CHURN_PREDICTION_RF',
                    mining_function     => DBMS_DATA_MINING.CLASSIFICATION,
                    data_table_name     => 'v_player_churn_train',
                    case_id_column_name => 'player_id',
                    target_column_name  => 'is_churned',
                    settings_table_name => 'churn_model_settings'
                );
            END;
        """)
        print("[Seeder] ✅ OML Model 'CHURN_PREDICTION_RF' trained successfully.")
        
    except oracledb.Error as e:
        print(f"[Seeder] ⚠ OML Training Failed: {e}")
        print("  (Ensure 'GRANT CREATE MINING MODEL TO seerplay' was executed in init.sql)")

    conn.commit()
    print("[Seeder] ✅ All data seeded successfully!")
    print(f"  → Players:    {len(players_data)}")
    print(f"  → Sessions:   {len(sessions_data)}")
    print(f"  → Events:     {len(events_data)}")
    print(f"  → Centroids:  {len(centroids)}")
    print(f"  → Devices:    {len(devices_data)}")
    print(f"  → IPs:        {len(ips_data)}")
    print(f"  → Games:      {len(GAME_CATALOG)}")
    print(f"  → Referrals:  {len(referrals_data)}")
    print(f"  → Risk Events:{len(risk_events_data)}")
    print(f"  → What-If R²: {model.score(reg_features, reg_targets):.4f}")

except Exception as e:
    print(f"[Seeder] ❌ Error: {e}", file=sys.stderr)
    sys.exit(1)
finally:
    if "conn" in locals():
        conn.close()
