/* Formatters + time normalization. Mirrors common.js semantics (which it
   replaces at cleanup) plus toEpochSec(), the single boundary where any
   date representation becomes lightweight-charts time: UTC-midnight
   epoch seconds. All chart data passes through it — no more
   date-string-vs-epoch axis mismatches. */

const DASH = '—';

export function toEpochSec(t) {
  if (t == null) return null;
  if (typeof t === 'number') return t > 1e10 ? Math.floor(t / 1000) : Math.floor(t);
  if (typeof t === 'string') {
    // "YYYY-MM-DD" or full ISO — pin date-only strings to UTC midnight
    const iso = t.length === 10 ? `${t}T00:00:00Z` : t;
    const ms = Date.parse(iso);
    return Number.isNaN(ms) ? null : Math.floor(ms / 1000);
  }
  if (t instanceof Date) return Math.floor(t.getTime() / 1000);
  return null;
}

export function fmtDollar(n) {
  if (n == null) return DASH;
  const abs = Math.abs(n);
  const sign = n < 0 ? '-' : '';
  if (abs >= 1e12) return `${sign}$${(abs / 1e12).toFixed(1)}T`;
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(1)}M`;
  return sign + '$' + abs.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function fmtPrice(n) {
  return n == null ? DASH : '$' + Number(n).toFixed(2);
}

export function fmtPct(n, { signed = true, decimals = 1 } = {}) {
  if (n == null) return DASH;
  const sign = signed && n >= 0 ? '+' : '';
  return sign + Number(n).toFixed(decimals) + '%';
}

export function fmtPnl(n) {
  if (n == null) return DASH;
  return (n >= 0 ? '+' : '') + fmtDollar(n);
}

export function fmtNum(n, decimals = 2) {
  return n == null ? DASH : Number(n).toFixed(decimals);
}

export function fmtInt(n) {
  return n == null ? DASH : Number(n).toLocaleString('en-US', { maximumFractionDigits: 0 });
}

export function fmtCompact(n) {
  if (n == null) return DASH;
  const abs = Math.abs(n);
  if (abs >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (abs >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (abs >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

export function upDownClass(n) {
  if (n == null) return '';
  return n >= 0 ? 't-up' : 't-down';
}

/* ── time: everything the desk shows is on the MARKET's clock ──
   Fills, decisions and sessions are all stamped in US/Eastern, never in the
   viewer's zone. A reader checking a fill against the tape should not have
   to do zone math, and a laptop left on the wrong timezone should not be
   able to make an accurate fill look late. The API serializes every
   timestamp with an explicit offset, so these conversions are exact. */

const ET = 'America/New_York';

function parse(iso) {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
}

/* A date-ONLY value's calendar day (ex-dates, split dates, bar dates).
   Rendered in UTC because that is the zone toEpochSec() pins date-only
   inputs to: reading a UTC-midnight epoch in the viewer's zone shows the
   previous day for everyone west of UTC — in New York every dividend and
   split would be dated a day early. Instants carrying a real time of day
   belong in fmtDateET/fmtDateTimeET instead. */
export function fmtDate(t) {
  if (!t) return DASH;
  const d = parse(typeof t === 'number' ? (t > 1e10 ? t : t * 1000) : t);
  if (!d) return DASH;
  return d.toLocaleDateString('en-US', {
    timeZone: 'UTC', month: 'short', day: 'numeric', year: 'numeric' });
}

/* The ET calendar date of an instant — the session a fill belongs to. A
   19:25 UTC fill is 3:25 PM the same day in New York, but a 01:30 UTC fill
   is the previous session's evening; only the ET date says which. */
export function fmtDateET(iso) {
  if (!iso) return DASH;
  const d = parse(iso);
  if (!d) return DASH;
  return d.toLocaleDateString('en-US', {
    timeZone: ET, month: 'short', day: 'numeric', year: 'numeric' });
}

/* ET clock time alone, for tables that already carry the date in their own
   column. Seconds are deliberate: a basket submitted in one cycle fills
   inside the same minute, and the second is what tells those fills apart
   when checking them against the tape. */
export function fmtTimeET(iso) {
  if (!iso) return DASH;
  const d = parse(iso);
  if (!d) return DASH;
  return d.toLocaleTimeString('en-US', {
    timeZone: ET, hour: 'numeric', minute: '2-digit', second: '2-digit',
    hour12: true,
  }) + ' ET';
}

/* Full ET stamp — date and time to the second, for independent verification. */
export function fmtDateTimeET(iso) {
  if (!iso) return DASH;
  const d = parse(iso);
  if (!d) return DASH;
  return d.toLocaleString('en-US', {
    timeZone: ET, month: 'numeric', day: 'numeric',
    hour: 'numeric', minute: '2-digit', second: '2-digit', hour12: true,
  }) + ' ET';
}

export function fmtTime(iso) {
  if (!iso) return DASH;
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) + ' ' +
         d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

export function timeAgo(iso) {
  if (!iso) return '';
  const mins = Math.floor((Date.now() - new Date(iso).getTime()) / 60000);
  if (mins < 1) return 'now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

export function fmtDuration(hours) {
  if (hours == null) return DASH;
  if (hours < 24) return `${Math.round(hours)}h`;
  return `${(hours / 24).toFixed(1)}d`;
}
