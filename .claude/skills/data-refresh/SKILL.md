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

The same run also ingests **SEC EDGAR fundamentals** for the universe (the
`edgar` block in the summary JSON — validation-gated source, see
`docs/fundamentals-validation.md`). READ that block and report
`fetched / rows_inserted / errors`. New filings trickle in daily — tens of
rows on a quiet night, hundreds during earnings season — so
`rows_inserted: 0` for several consecutive nights DURING earnings season is
a symptom to flag, not calm. ETFs and foreign listings legitimately have no
CIK (`no_cik`); a handful of 404s on ETF trusts (DIA, MDY, QQQ) are normal.

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

Also check the fundamentals side of the asset:
```
python -m agent.edgar coverage
```
Report `symbols / rows / stale_symbols`. Healthy is ~740+ symbols with a
median of ~60 filings each; `stale_symbols` (no filing in 120 days) growing
run-over-run means CIK resolution or the EDGAR pass is quietly failing —
flag it loudly, same rule as bar coverage.

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
- **`fundamentals_pit` is written only by `agent.edgar`** (which rides this
  Routine's market refresh). EDGAR is a refetchable public-domain source, so
  `python -m agent.edgar ingest --rebuild` exists to recompute history when
  the derivation logic improves — it is a maintenance tool, never a nightly
  step. The frozen `fundamentals_snapshots` table is the validation
  reference: read-only, forever.
- **Idempotent + bounded.** `agent.refresh` sets a socket timeout so a hung
  Alpaca call fails fast instead of stalling the whole ingest; re-running is a
  cheap near-noop when already current.
- **Honest reporting.** If coverage dropped or storage is tight, that IS the
  headline of the run — don't bury it.

## Owner setup (one-time — cannot be done from a sandbox)
Create the Routine at **claude.ai/code/routines** on this repo:
- **Cron:** `45 0 * * 2-6` — **UTC**. A fixed UTC cron drifts with DST:
  this lands at **8:45 PM ET in summer (EDT) but 7:45 PM ET in winter
  (EST)** — either way after the U.S. close, Mon–Fri. That is fine: the
  archive's include-today gate keys off the trading **calendar** (has
  today's ET session ended?), not the wall clock, so tonight's final bar
  joins R2 in both seasons. The Strategy Lab Routine follows at
  `0 2 * * 2-6` UTC so it sweeps on tonight's fresh data.
- **Prompt:** `Run the data-refresh skill.`
- **Env:** the session needs the Supabase (DB) + Alpaca + R2_* credentials
  available (same env as the trading Routine). SEC EDGAR needs no secret —
  its authentication is the declared User-Agent (`settings.edgar_user_agent`).
