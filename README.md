# Employee Attendance & Analytics API

FastAPI + MongoDB. All application code is in `app/main.py`. Part A findings are in `REVIEW.md` and design answers in
`DECISIONS.md`.

## Run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export MONGO_URI="mongodb://localhost:27017" MONGO_DB="attendance_db"   # or put them in a local .env (see .env.example)
uvicorn app.main:app --port 8000
```

* MongoDB 6.0+ is required (uses `$setWindowFields`, `$densify` and `$lookup` with both `localField` and `pipeline`).
* Indexes are created at startup and are idempotent. If MongoDB is not reachable at boot, the app still starts,
  `/health` returns 503, and the indexes are created on the first successful `/health`.
* Real environment variables take priority over `.env`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q          # needs a MongoDB at MONGO_URI; uses and then drops the database "attendance_api_tests"
```

The suite covers R1-R10 edge cases, the validation and status codes, concurrency (parallel punch-ins, punch-outs,
creates and corrections), and the explain plans. The analytics are checked against a plain-Python reference
implementation over randomly generated "messy" data: employees with no logs, mid-month and future joiners, weekend
records, half days, legacy documents without `half_day`/`history`, orphan logs, overnight shifts and ties.

I also checked the app on about 100,000 attendance records and 1,000 employees. It is ready in about 2 s, including
index builds. List and analytics calls take 3-100 ms, and every `/admin/explain/*` plan uses `IXSCAN` with no `COLLSCAN`.

## Interpretations worth knowing

* **Trend range limit.** "Exceeds 92 days" is read as more than 92 calendar days *inclusive*: `2026-07-01..2026-09-30`
  (92 days, one quarter) is allowed, and one more day is 422.
* **`attendance_pct`** is a reported number, so it uses 2 decimals (R8). Only the trend's `attendance_rate` and
  `moving_avg_7d` are rates (4 decimals).
* **`moving_avg_7d`** averages the reported (already rounded) `attendance_rate` values of the window.
* **Rounding** is half-up everywhere: `Decimal` in Python, and `floor(x * 10^n + 0.5) / 10^n` on Decimal128 inside
  pipelines, because MongoDB's `$round` rounds half to even.
* **PATCH "changes nothing"** means none of `status`, `punch_in` or `punch_out` would change (after truncation to
  seconds). `punch_in` is checked against the record's attendance date (R1) when it is supplied. A final presence
  status must have a `punch_in`, and a final `ABSENT`/`LEAVE` status must not be sent with punch times.
* **Concurrency.** Punch-in and employee creation rely on unique indexes. Punch-out and PATCH are compare-and-set
  updates (the filter includes the values that were read, and the history length for PATCH), so the loser gets 409
  rather than overwriting.
* Optional instant fields (`punched_at`, PATCH `punch_in`/`punch_out`) may be omitted, but an explicit `null` is 422
  because the contract types them as non-nullable integers.

## Not included

Nothing from the contract is knowingly missing.
