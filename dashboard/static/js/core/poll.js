/* Poller — replaces every ad-hoc setInterval/setTimeout polling loop.
   Guarantees: pauses when the tab is hidden, exponential backoff on
   failures, HARD STOP after maxFailures (no infinite spinners — the
   backtest-job bug class), clean teardown. */

export function poller(fn, {
  intervalMs = 60000,
  maxFailures = 5,
  backoffFactor = 2,
  onGiveUp = null,
} = {}) {
  let timer = null;
  let failures = 0;
  let stopped = false;

  const schedule = (ms) => {
    if (stopped) return;
    timer = setTimeout(tick, ms);
  };

  const tick = async () => {
    if (stopped) return;
    if (document.hidden) { schedule(intervalMs); return; }
    try {
      const keepGoing = await fn();
      failures = 0;
      if (keepGoing === false) { stop(); return; } // fn signals completion
      schedule(intervalMs);
    } catch (err) {
      failures += 1;
      if (failures >= maxFailures) {
        stop();
        if (onGiveUp) onGiveUp(err);
        return;
      }
      schedule(intervalMs * Math.pow(backoffFactor, failures));
    }
  };

  const onVisible = () => {
    if (!document.hidden && !stopped) {
      clearTimeout(timer);
      tick();
    }
  };
  document.addEventListener('visibilitychange', onVisible);

  function stop() {
    stopped = true;
    clearTimeout(timer);
    document.removeEventListener('visibilitychange', onVisible);
  }

  tick(); // immediate first run
  return { stop };
}
