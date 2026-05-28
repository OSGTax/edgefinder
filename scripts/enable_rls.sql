-- Enable Row Level Security on all public tables (Supabase advisory 0013).
--
-- WHY: Supabase exposes the `public` schema through PostgREST. Every table
-- here currently has RLS disabled, so the `anon` and `authenticated` API
-- roles can read/write the entire DB over the public REST endpoint. The
-- security linter flags this as ERROR-level (rls_disabled_in_public).
-- See: https://supabase.com/docs/guides/database/database-linter?lint=0013_rls_disabled_in_public
--
-- WHAT THIS DOES: enables RLS on each table WITHOUT adding any policy.
-- With RLS on and zero policies, the `anon`/`authenticated` roles are
-- denied all access — which is what we want, since EdgeFinder is an
-- internal app and nothing should reach these tables via the public API.
--
-- WHY IT DOES NOT BREAK THE APP: the EdgeFinder backend (Render) and the
-- agent crons connect through the Supabase pooler as the `postgres` role,
-- which OWNS these tables. Table owners BYPASS RLS by default (we do NOT
-- use FORCE ROW LEVEL SECURITY), so application queries are unaffected.
--
-- BEFORE APPLYING — verify the connecting role bypasses RLS:
--   1. Confirm DATABASE_URL on Render starts with `postgres.<ref>:` (the
--      `postgres` owner role), not a restricted custom role.
--   2. Sanity check ownership:
--        SELECT tablename, tableowner FROM pg_tables WHERE schemaname='public';
--      All should be owned by `postgres`. If any table is owned by a
--      different non-superuser role, that role would be subject to RLS and
--      you must add a policy (or ALTER TABLE ... OWNER TO postgres) first.
--   3. This script is idempotent — ENABLE ROW LEVEL SECURITY is a no-op if
--      already enabled. Safe to re-run.
--
-- APPLY (do NOT auto-apply via alembic on deploy — run manually after the
-- checks above, e.g. through the Supabase SQL editor or MCP):
--   \i scripts/enable_rls.sql

ALTER TABLE public.agent_actions                  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_memory                   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_observations             ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.alembic_version                ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.daily_bars                     ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fundamentals                   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.index_daily                    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.manual_injections             ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.market_snapshots               ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.strategy_accounts              ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.strategy_parameters            ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.strategy_snapshots             ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ticker_dividends               ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ticker_news                    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ticker_splits                  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ticker_strategy_qualifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tickers                        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.trade_context                  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.trades                         ENABLE ROW LEVEL SECURITY;

-- VERIFY (every row should show rowsecurity = true):
--   SELECT tablename, rowsecurity FROM pg_tables
--   WHERE schemaname='public' ORDER BY tablename;
