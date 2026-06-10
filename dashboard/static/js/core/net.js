/* Network layer — every fetch in the redesigned dashboard goes through
   here. Guarantees: timeouts (no infinite spinners), bounded retries on
   transient failures only, in-flight deduplication (poller + user action
   never double-fetch), and typed errors the UI renders as error cards.
   NO silent failures, NO fabricated fallback data — ever. */

export class ApiError extends Error {
  constructor(path, status, body) {
    super(`API ${path} -> ${status}`);
    this.path = path;
    this.status = status; // 0 = network/timeout
    this.body = body;
  }
}

const inflight = new Map(); // path -> Promise (GET dedup)

async function fetchOnce(path, timeout, signal) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new DOMException('timeout', 'AbortError')), timeout);
  if (signal) {
    if (signal.aborted) ctrl.abort(signal.reason);
    else signal.addEventListener('abort', () => ctrl.abort(signal.reason), { once: true });
  }
  try {
    const res = await fetch(path, { signal: ctrl.signal });
    if (!res.ok) {
      let body = null;
      try { body = await res.json(); } catch { /* non-JSON error body */ }
      throw new ApiError(path, res.status, body);
    }
    return await res.json();
  } catch (err) {
    if (err instanceof ApiError) throw err;
    // network failure, timeout, or caller abort
    throw new ApiError(path, 0, { reason: String(err && err.message || err) });
  } finally {
    clearTimeout(timer);
  }
}

export async function apiGet(path, { timeout = 12000, retries = 2, signal } = {}) {
  if (inflight.has(path)) return inflight.get(path);

  const run = (async () => {
    let lastErr;
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        return await fetchOnce(path, timeout, signal);
      } catch (err) {
        lastErr = err;
        const callerAborted = signal && signal.aborted;
        const retriable = !callerAborted && (err.status === 0 || err.status >= 500);
        if (!retriable || attempt === retries) throw err;
        await new Promise(r => setTimeout(r, attempt === 0 ? 500 : 1500));
      }
    }
    throw lastErr;
  })();

  inflight.set(path, run);
  try {
    return await run;
  } finally {
    inflight.delete(path);
  }
}

export async function apiPost(path, payload, { timeout = 20000 } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeout);
  try {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload ?? {}),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      let body = null;
      try { body = await res.json(); } catch { /* ignore */ }
      throw new ApiError(path, res.status, body);
    }
    return await res.json();
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(path, 0, { reason: String(err && err.message || err) });
  } finally {
    clearTimeout(timer);
  }
}
