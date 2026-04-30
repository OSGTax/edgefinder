# Coach + Weekly Summary — your setup checklist

The code is shipped. **None of these jobs will actually run until you do
the steps below.** Each step is ~1-2 minutes; total ~10 minutes.

Until you run through this, the workflows live in `.github/workflows/`
but the kill-switch variables (`COACH_ENABLED`, `WEEKLY_SUMMARY_ENABLED`)
are unset, so the cron does nothing.

## What you're enabling

- **Coach** (`edgefinder/agents/coach.py`) — daily Mon-Fri at 5:30 PM ET.
  Reviews one strategy on a fixed weekday rotation:
  Mon=alpha, Tue=bravo, Wed=charlie, Thu=degenerate, Fri=echo. Always
  writes a markdown review to `reviews/YYYY-MM-DD-<strategy>.md`.
  Sometimes opens a PR with a single `config/settings.py` tweak that
  auto-merges after the test suite passes.

- **Weekly summary** (`edgefinder/agents/weekly_summary.py`) — Saturday
  morning. Reads the past 7 days of reviews + trades and writes a
  portfolio-level synthesis to `reviews/WEEK-YYYY-WW.md`. No code or
  parameter changes — just observation.

- **Tests workflow** (`.github/workflows/tests.yml`) — gates the
  coach's auto-merge. If a coach-proposed change breaks any test,
  the PR stays open instead of merging.

## The setup checklist

### Step 1 — Generate your Claude OAuth token (one-time, ~1 minute)

You need this if you don't already have it from the watchdog setup.
On a machine with Claude Code installed (your laptop, or any
Codespace), run:

```bash
claude setup-token
```

This walks you through a browser OAuth flow and prints a long-lived
token (good ~1 year). Copy it — you'll paste it in step 2. The token
authenticates `claude -p` against your Claude subscription quota in CI.

### Step 2 — Add secrets to GitHub Actions

GitHub → repo Settings → Secrets and variables → Actions → **Secrets** tab.

Make sure both of these exist (add if missing):

- `DATABASE_URL` — your Supabase pooler URL.
  - Find it: Render dashboard → your service → Environment, copy the
    `DATABASE_URL` value.
  - Or: Supabase → Project Settings → Database → Connection Pooling →
    **Session** mode (port 6543).

- `CLAUDE_CODE_OAUTH_TOKEN` — the token from step 1.

### Step 3 — Add the kill-switch variables

Same page → **Variables** tab → New repository variable.

Add **both**:

- `COACH_ENABLED` = `true`
- `WEEKLY_SUMMARY_ENABLED` = `true`

If you want to pause either job later, flip the variable to `false` —
no deploy needed; the workflow's `if:` gate skips the entire job.

### Step 4 — Allow auto-merge in repo settings

GitHub → repo Settings → General → scroll down to **Pull Requests**.

Check ☑ **Allow auto-merge**.

This is required for the coach's `gh pr merge --auto` to work. If
this isn't enabled, the workflow opens the PR but can't enable
auto-merge — the PR sits open for you to merge manually.

### Step 5 — Smoke test the coach

GitHub → Actions → **Coach (daily review + tune)** → Run workflow.

In the inputs:
- **Strategy:** leave blank (use today's rotation) or set `alpha` for a quick test
- **Dry run:** set to `true` for the smoke test (prints prompt, no commits)

Click **Run workflow**. The run should:

1. Install Python + Node + the `claude` CLI
2. Print `[coach] today is <weekday>, strategy=<name>` (or `weekend, exiting`)
3. Print the trade count it pulled from Supabase
4. Print Claude's review JSON
5. Exit green

If step 4 fails with `CLAUDE_CODE_OAUTH_TOKEN not set`, your secret in
step 2 is missing or named wrong.

If step 4 succeeds but you see `claude -p exited 1`, your OAuth token
is invalid — re-run `claude setup-token` and update the secret.

### Step 6 — Real first run

Re-run the workflow with **Dry run = false**. The run will:

1. Write a `reviews/YYYY-MM-DD-<strategy>.md` file.
2. Either commit just that to `main` (no tune proposed), or open a PR
   with both the review AND a `config/settings.py` tweak (tune
   proposed). Auto-merge will be enabled on the PR.

Pull `main` locally, look in `reviews/` for the new file, read it. If
it reads sensibly, **you're done forever** — the cron handles itself
from here.

### Step 7 (optional) — Smoke test the weekly summary

GitHub → Actions → **Weekly portfolio summary** → Run workflow → Dry
run = `true`. Same shape as step 5. Then run with Dry run = `false` —
should commit `reviews/WEEK-YYYY-WW.md` to main.

### Step 8 (optional) — Codespaces secrets for ad-hoc runs

If you want to run the coach manually from a Codespace while
iterating: GitHub → personal Settings → Codespaces → Secrets. Add
`DATABASE_URL` and `CLAUDE_CODE_OAUTH_TOKEN`, scope to
`osgtax/edgefinder`.

Then in any Codespace:

```bash
python -m edgefinder.agents.coach --strategy alpha --dry-run
python -m edgefinder.agents.coach --strategy alpha           # real run
python -m edgefinder.agents.weekly_summary --dry-run
```

## Pausing or rolling back

| Want to... | Do this |
|---|---|
| Pause the coach | Set `COACH_ENABLED` = `false` (variable). Instant, no deploy. |
| Pause the weekly summary | Set `WEEKLY_SUMMARY_ENABLED` = `false`. |
| Revert a bad coach tune | `git revert <sha>` on `main`, push. Render redeploys with the old params. |
| Stop forever | Delete the workflow files in `.github/workflows/`, or the secrets. |

## Files added in this PR

- `edgefinder/agents/coach.py` — rotation + review + tune + PR
- `edgefinder/agents/weekly_summary.py` — Saturday digest
- `.github/workflows/coach.yml` — daily Mon-Fri cron
- `.github/workflows/weekly-summary.yml` — Saturday cron
- `.github/workflows/tests.yml` — gates the auto-merge
- `tests/test_coach.py`, `tests/test_weekly_summary.py` — 25 new tests
- `reviews/.gitkeep` — empty directory for the markdown reviews

That's everything. When you have ~10 minutes, walk steps 1-6 and
you're done.
