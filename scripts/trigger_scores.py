import urllib.request
import json
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
db_url = os.environ.get("DATABASE_URL")

conn = psycopg2.connect(db_url)
cur = conn.cursor()

# Get 500 random transactions to score
cur.execute("""
    SELECT txn_id, user_id, amount, currency, merchant_id, device_id, 
           ip_address, country, latitude, longitude, created_at
    FROM transactions 
    ORDER BY RANDOM() LIMIT 500;
""")
txns = cur.fetchall()

print(f"Scoring {len(txns)} transactions...")
flagged = 0

for row in txns:
    txn_id, user_id, amount, currency, merchant_id, device_id, ip_address, country, latitude, longitude, created_at = row
    try:
        payload = {
            "txn_id": str(txn_id),
            "user_id": str(user_id),
            "amount": float(amount),
            "currency": currency,
            "merchant_id": merchant_id,
            "device_id": device_id,
            "ip_address": ip_address,
            "country": country,
            "latitude": float(latitude) if latitude else 0.0,
            "longitude": float(longitude) if longitude else 0.0,
            "created_at": created_at.isoformat()
        }
        req = urllib.request.Request(
            "http://localhost:8000/score",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                result = json.loads(response.read().decode('utf-8'))
                if result.get("flagged") is True:
                    flagged += 1
                    print(f"Flagged! {txn_id} (Score: {result.get('risk_score')})")
    except Exception as e:
        print(f"Error scoring {txn_id}: {e}")

print(f"Finished scoring! Flagged {flagged} transactions.")
