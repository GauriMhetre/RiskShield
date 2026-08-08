# Phase 5, Task 1 — PostgreSQL Setup & Schema Creation

This document guides you through setting up PostgreSQL for RiskShield using Docker.

## Prerequisites

- Docker Desktop installed and running (download from https://www.docker.com/products/docker-desktop/)
- Windows PowerShell
- RiskShield project at `c:\Users\GAURI\OneDrive\Desktop\Projects\RiskShield`

## Step 1: Start PostgreSQL Container

Open PowerShell and run:

```powershell
docker run --name riskshield-postgres -e POSTGRES_PASSWORD=devpassword -e POSTGRES_DB=riskshield -p 5432:5432 -d postgres:16
```

**Expected output:** A long container ID string (success).

**Possible error:** `port is already allocated` — change `-p 5432:5432` to `-p 5433:5432` if port 5432 is in use, then remember to use `5433` in connection strings.

### Verify container is running:

```powershell
docker ps
```

Should show a row with `riskshield-postgres` and status `Up ...`.

## Step 2: Verify Database Connection

```powershell
docker exec -it riskshield-postgres psql -U postgres -d riskshield
```

You should see a `riskshield=#` prompt. Inside psql, run:

```sql
SELECT version();
```

Should return the PostgreSQL version (e.g., PostgreSQL 16.x).

Exit psql:

```sql
\q
```

## Step 3: Apply the Schema

**Ensure pgcrypto extension is enabled:**

```powershell
docker exec -it riskshield-postgres psql -U postgres -d riskshield -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;"
```

**Apply the schema from db/schema.sql:**

```powershell
Get-Content db\schema.sql | docker exec -i riskshield-postgres psql -U postgres -d riskshield
```

**Expected output:** Series of `CREATE TABLE` and `CREATE INDEX` statements.

## Step 4: Verify Schema Was Applied

**List all tables:**

```powershell
docker exec -it riskshield-postgres psql -U postgres -d riskshield -c "\dt"
```

Should show three tables:
- `user_profile`
- `transactions`
- `scored_transactions`

**List all indexes:**

```powershell
docker exec -it riskshield-postgres psql -U postgres -d riskshield -c "\di"
```

Should show:
- `idx_txn_user_time` (on transactions table)
- `idx_scored_flagged` (on scored_transactions table)

## Step 5: Test with Sample Data

Insert a test user:

```powershell
docker exec -it riskshield-postgres psql -U postgres -d riskshield
```

Inside psql:

```sql
INSERT INTO user_profile (user_id, avg_txn_amount, std_txn_amount)
VALUES (gen_random_uuid(), 1000.00, 200.00);

SELECT * FROM user_profile;
```

Should return one row with a UUID, amounts, and a timestamp.

## Environment Configuration

The `.env` file contains:

```
DATABASE_URL=postgresql://postgres:devpassword@localhost:5432/riskshield
```

**IMPORTANT:** `.env` is in `.gitignore` and will never be committed. It contains sensitive credentials (the database password).

The `.env.example` file shows the format without real secrets — this IS committed so team members know what to configure.

## Connection String Format

For future Python/SQLAlchemy integration:

- **User:** `postgres`
- **Password:** `devpassword`
- **Host:** `localhost`
- **Port:** `5432`
- **Database:** `riskshield`
- **Full URL:** `postgresql://postgres:devpassword@localhost:5432/riskshield`

## Schema Overview

### `user_profile` Table

Stores aggregated user behavioral statistics. Used for computing velocity and amount deviation features during scoring.

| Column | Type | Purpose |
|--------|------|---------|
| `user_id` | UUID | Primary key, unique user identifier |
| `avg_txn_amount` | NUMERIC(12,2) | Historical mean transaction amount |
| `std_txn_amount` | NUMERIC(12,2) | Historical std dev of amounts |
| `txn_count` | INTEGER | Total transaction count |
| `last_device_id` | TEXT | Most recent device fingerprint |
| `last_country` | TEXT | Most recent transaction country |
| `last_latitude` | NUMERIC(9,6) | Most recent transaction latitude |
| `last_longitude` | NUMERIC(9,6) | Most recent transaction longitude |
| `updated_at` | TIMESTAMPTZ | When profile was last updated |

### `transactions` Table

Raw incoming transaction records. Every transaction scored by `/score` endpoint is logged here.

| Column | Type | Purpose |
|--------|------|---------|
| `txn_id` | UUID | Primary key, auto-generated |
| `user_id` | UUID | Foreign key to `user_profile` |
| `amount` | NUMERIC(12,2) | Transaction amount |
| `currency` | TEXT | Currency code (e.g., USD, INR) |
| `merchant_id` | TEXT | Merchant identifier |
| `device_id` | TEXT | Device fingerprint |
| `ip_address` | INET | IP address |
| `country` | TEXT | Country code |
| `latitude` | NUMERIC(9,6) | Geographic latitude |
| `longitude` | NUMERIC(9,6) | Geographic longitude |
| `created_at` | TIMESTAMPTZ | Transaction timestamp |

**Index:** `idx_txn_user_time` on `(user_id, created_at DESC)` — supports efficient velocity feature queries ("all txns for user X in last 24 hours").

### `scored_transactions` Table

Immutable log of all scoring decisions. Every time `/score` returns a result, one row is created here.

| Column | Type | Purpose |
|--------|------|---------|
| `id` | BIGSERIAL | Primary key, auto-incrementing |
| `txn_id` | UUID | Foreign key to `transactions` |
| `risk_score` | NUMERIC(5,4) | Model output probability (0–1) |
| `flagged` | BOOLEAN | Whether transaction was flagged as fraud |
| `model_version` | TEXT | Which model version scored this (e.g., "model_v1") |
| `feature_snapshot` | JSONB | Exact 10-element feature dict used for scoring |
| `shap_values` | JSONB | Reserved for future SHAP-based interpretability |
| `scored_at` | TIMESTAMPTZ | When transaction was scored |

**Index:** `idx_scored_flagged` on `(flagged, scored_at DESC)` — supports efficient queries for recently flagged transactions (dashboard, alerts).

## Troubleshooting

### Docker Container Won't Start

**Error:** `docker: command not found` or Docker not running.

**Solution:** Install Docker Desktop and ensure it's running. Restart PowerShell after installation.

### Port 5432 Already in Use

**Error:** `port is already allocated`

**Solution:** Either stop the service using port 5432, or use a different port by changing `-p 5432:5432` to `-p 5433:5432` in the docker run command. Update `DATABASE_URL` in `.env` accordingly.

### Connection Refused

**Error:** `could not connect to server: Connection refused`

**Solution:** Verify container is running with `docker ps`. If not running, start it with the `docker run` command above.

### Permission Denied

**Error:** `permission denied while trying to connect to the Docker daemon socket`

**Solution:** Docker Desktop needs to be running. Also ensure PowerShell is not restricted — may need to run as Administrator.

## Next Steps

Once the schema is verified:

1. **Phase 5, Task 2** will create SQLAlchemy ORM models corresponding to these tables
2. **Phase 5, Task 3** will wire the `/score` route to write to these tables
3. Later phases will add query functions to populate `user_profile` and build dashboards on `scored_transactions`

## Notes

- The schema includes `pgcrypto` extension for UUID support — this is standard in production PostgreSQL deployments
- JSONB for `feature_snapshot` allows flexible schema evolution: if `ml/features.py` adds new computed features, the table structure doesn't change
- `shap_values` column is reserved for future SHAP-based model interpretability (a stretch-goal feature)
- `feedback_labels` table is intentionally NOT included yet — that's part of a later phase after scoring is in place

## Helpful PostgreSQL Commands

**Inside psql:**

- `\dt` — list all tables
- `\di` — list all indexes
- `\d tablename` — describe table schema
- `\q` — quit psql
- `SELECT * FROM user_profile;` — view all rows in a table
- `SELECT COUNT(*) FROM transactions;` — count rows

**From PowerShell (outside container):**

```powershell
# Run a single query without entering psql interactive mode
docker exec -it riskshield-postgres psql -U postgres -d riskshield -c "SELECT * FROM user_profile;"

# Backup the database
docker exec -it riskshield-postgres pg_dump -U postgres riskshield > backup.sql

# Restore from backup (if needed)
docker exec -i riskshield-postgres psql -U postgres riskshield < backup.sql
```

