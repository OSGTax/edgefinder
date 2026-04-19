---
name: watchdog-tick
description: Run one cycle of the EdgeFinder watchdog — checks DB invariants (cash drift, negative cash, paused accounts, high drawdown), records new observations, auto-resolves cleared ones. Use when the user says "run the watchdog", "watchdog tick", "check the system", "watchdog check", or similar.
---

# Watchdog Tick

You are running one tick of the EdgeFinder watchdog management agent. The agent checks invariants in the trading DB and records findings to the `agent_observations` table. This is a read-mostly operation — it writes only to the audit tables, never to `trades` or `strategy_accounts`.

## What to do

1. **Check the kill switch.** Read `.claude/agent-config.json`:
   - If the file is missing, or `enabled` is false, or `agents.watchdog.enabled` is false: stop and tell the user the watchdog is disabled. Suggest how to enable (`enabled: true` and `agents.watchdog.enabled: true`). Do not run the tick.
   - If enabled: proceed.

2. **Run the tick.** Invoke the CLI:
   ```
   python -m edgefinder.agents.watchdog
   ```
   If the user wants a preview without writing, add `--dry-run`.

3. **Summarize the output.** The CLI logs one line per finding and a final `tick done — new=X kept=Y resolved=Z` line. Report to the user:
   - How many new observations were created (actionable now)
   - How many persisted from prior ticks (still unresolved)
   - How many got auto-resolved (condition cleared)
   - Group the new/kept findings by category and severity. For CRITICAL findings, quote the full message.

4. **Offer follow-ups based on severity, but do not execute them without explicit user approval:**
   - Any `CRITICAL` finding → suggest opening a GitHub issue with the finding's message and metadata, tagged `watchdog`. Wait for the user to confirm before calling `mcp__github__issue_write`.
   - `negative_cash` or `cash_drift` with diff > $500 → suggest investigating the specific strategy's recent trades. Read-only — don't auto-fix.
   - `account_paused` for > 1 tick → suggest a manual review of the account's drawdown history.
   - `high_drawdown` → suggest checking whether the strategy's stop-losses are tripping correctly.

## Do not
- Write to `trades`, `strategy_accounts`, or any non-audit table.
- Edit code in `edgefinder/trading/`, `edgefinder/strategies/`, or `config/`.
- Open issues, PRs, or commits without explicit user approval.
- Invent findings or editorialize — only report what the CLI output shows.

## Scheduled / headless usage
When run outside a Claude Code session (GitHub Actions cron, Render cron), the CLI handles everything itself: reads the kill switch, runs checks, persists, and exits with code 0. No Claude involvement needed. The skill exists for the interactive "what's the system doing right now" use case and for the user-approval layer around critical findings.
