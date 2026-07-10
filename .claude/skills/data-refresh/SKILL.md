---
name: data-refresh
description: Keep the whole-market data asset fresh — the nightly full-market ingest that maintains EdgeFinder's standing "fresh set" (top-N by dollar volume) plus a storage-headroom check. Use when invoked by the end-of-day data Routine, or when the user says "refresh the market data", "run the nightly ingest", or "top up the universe".
---

# EdgeFinder Data-Refresh — one nightly ingest

This is the **standing-fresh-set maintainer**. The hourly trading cycle only
tops up ~15 names (held + streamed + watchlist) via `agent.refresh --source
alpaca`. Every OTHER name (the long tail you chart or research) stays current
only because this routine runs. Miss it and coverage decays — names silently go
stale (that is exactly how CPB/SUNE ended up weeks behind).

Run this **once daily, after the close** (the Routine that also runs
`app-evolver`). Alpaca is the sole data source. Everything is idempotent and
safe to re-run.

## The cycle — do these in order

### 1. Ingest the top-N by dollar volume
```
python -m agent.refresh --source alpaca-market --top 1000
```
This enumerates Alpaca's whole tradable catalog (~13k), ranks by dollar volume,
and keeps fresh daily-bar history for the top-N **plus** benchmarks / held /
watchlist (they never fall off the edge). It also merge-syncs the grow-only R2
parquet archive when the R2_* creds are present. `--top` is the breadth dial:
raise it to keep more of the long tail current, lower it to conserve storage.
Anything outside top-N is still quote- and fill-able live and can be topped up
on demand with `agent.refresh --source alpaca --symbols X,Y` (or by putting a
name on the watchlist).

### 2. Build tonight's research brief
```
python -m agent.market brief-build --top 40
```
The whole-market picture is freshest right now, so pack it once: regime,
ranked universe, movers across the last two well-covered sessions, a trend
roster with indicators, headlines for the leaders, and the data-coverage
verdict. Tomorrow's hourly trading cycles read this ONE payload
(`agent.market brief`) instead of re-deriving it every hour — the trader
spends its context deciding, not gathering. The build is section-tolerant:
transient failures land in the output's `errors` list instead of aborting.
If `errors` is non-empty, re-run the build ONCE (it upserts in place); if
errors persist or `coverage_status` is not `green`, say so loudly in your
report.

### 3. Report coverage
Count names with a bar on the latest trading day so a decaying ingest is
visible, not silent:
```
python - <<'PY'
from agent.store import get_store
from datetime import date, timedelta
store = get_store()
for d in [date.today()-timedelta(days=i) for i in range(1,5)]:
    n = len(store.select("daily_bars", columns="symbol",
                         filters={"date": d.isoformat()}, limit=100000))
    if n:
        print(f"{d}: {n} symbols fresh"); break
PY
```
Healthy is coverage at/near `--top`. A sharp drop means the ingest is failing
partway (usually a network hang) — investigate before it decays further.

### 4. Storage-headroom check — flag before a limit bites
Free tiers: **R2 = 10 GB**, **Supabase DB = 500 MB**. `daily_bars` is the
dominant table and (post-rebuild) only GROWS — there is no prune — so watch it.
Report R2 bytes used and the `daily_bars` row scale; if R2 is over ~8 GB or the
DB looks to be approaching ~450 MB, say so loudly (the owner asked to be pinged
on limits) and recommend lowering `--top` or adding a prune step before the next
run. R2 size:
```
python - <<'PY'
import os, boto3
s3 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"],
     aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
     aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"])
tot=n=0
for p in s3.get_paginator("list_objects_v2").paginate(Bucket=os.environ["R2_BUCKET"]):
    for o in p.get("Contents", []): tot+=o["Size"]; n+=1
print(f"R2: {n} objects, {tot/1e9:.2f} GB / 10 GB")
PY
```

## Guardrails
- **Data only.** This routine never touches the book, the strategy, or UI
  files — it maintains the market-data asset the whole system reads. Its ONE
  sanctioned `desk_*` write is the research brief (`desk_briefs`, via
  `agent.market brief-build`) — packaged market observation, not book state.
- **Idempotent + bounded.** `agent.refresh` sets a socket timeout so a hung
  Alpaca call fails fast instead of stalling the whole ingest; re-running is a
  cheap near-noop when already current.
- **Honest reporting.** If coverage dropped or storage is tight, that IS the
  headline of the run — don't bury it.

## Owner setup (one-time — cannot be done from a sandbox)
Create the Routine at **claude.ai/code/routines** on this repo:
- **Cron:** `~7 6 * * *` (a few minutes past 06:00 local — an off-:00 minute so
  it doesn't collide with every other job on the hour), i.e. after the U.S.
  close and before the next session.
- **Prompt:** `Run the data-refresh skill.`
- **Env:** the session needs the Supabase (DB) + Alpaca + R2_* credentials
  available (same env as the trading Routine).
This can share the end-of-day Routine that already runs `app-evolver`: refresh
the data first, then evolve the desk.
