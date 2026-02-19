
import oracledb
import sys

DSN      = "localhost:1521/FREEPDB1"
USER     = "seerplay"
PASSWORD = "SeerPlay123"

def verify():
    try:
        conn = oracledb.connect(user=USER, password=PASSWORD, dsn=DSN)
        cursor = conn.cursor()
        
        print("Checking for aggression_score and variance_level in player_sessions...")
        cursor.execute("SELECT count(*) FROM player_sessions WHERE aggression_score IS NOT NULL AND variance_level IS NOT NULL")
        count = cursor.fetchone()[0]
        
        print(f"Found {count} rows with populated aggression/variance data.")
        
        if count > 0:
            cursor.execute("SELECT session_id, aggression_score, variance_level FROM player_sessions FETCH FIRST 5 ROWS ONLY")
            print("\nSample data:")
            for row in cursor.fetchall():
                print(f"Session {row[0]}: Aggression={row[1]}, Variance={row[2]}")
            print("\n✅ Verification SUCCESS: Data is present.")
        else:
            print("\n❌ Verification FAILED: No data found.")
            
        conn.close()
    except Exception as e:
        print(f"\n❌ Verification FAILED: {e}")

if __name__ == "__main__":
    verify()
