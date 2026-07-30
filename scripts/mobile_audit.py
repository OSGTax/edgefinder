#!/usr/bin/env python3
"""Measure the desk's mobile layout — geometry, not opinion.

`/desk` is read mostly on phones. "Looks better on my phone" is not a check
that survives the nightly app-evolver, so this asserts the two properties that
actually break mobile reading:

  1. **No horizontal overflow.** A page wider than the viewport scrolls
     sideways, which is the single most broken-feeling mobile symptom.
  2. **Bounded page height.** The desk once rendered 57,102px at 390px wide —
     68 phone screens — because every long-form corpus (wiki pages, the claims
     registry, the thinking feed) rendered in full.

It prints the tallest blocks and every overflowing element, so a failure says
WHICH element to fix rather than just "too wide".

Why it proxies the API: several `/api/desk/*` endpoints read through the `pg`
lane (direct Postgres), and sandboxes/CI commonly block that port — the page
then renders empty and every measurement is meaningless (an empty desk measures
0px of overflow and passes vacuously). So requests are intercepted via CDP and
fulfilled from a live origin. Measure a populated page or don't bother.

Usage:
    # against a locally running dashboard, data proxied from production
    python scripts/mobile_audit.py --url http://127.0.0.1:8811/desk

    python scripts/mobile_audit.py --width 320 --height 568   # smallest iPhone
    python scripts/mobile_audit.py --width 1440 --desktop     # no-regression
    python scripts/mobile_audit.py --screenshot /tmp/desk.png
    python scripts/mobile_audit.py --max-height 12000         # tighten the gate

Exit code is non-zero when a gate fails, so CI can consume it.

Requires: a Chromium binary (PLAYWRIGHT_BROWSERS_PATH, or --chrome), and the
`websockets` package (already a runtime dep).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import glob
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

# Gates. The height ceiling is generous on purpose — it exists to catch a
# regression back toward "unbounded corpus rendered inline", not to police
# layout taste. Tighten with --max-height as the desk gets leaner.
DEFAULT_MAX_HEIGHT = 14000
DEFAULT_MAX_OVERFLOW = 0
# Touch-target floor: 44px is the iOS HIG minimum (Android asks 48dp).
TOUCH_MIN = 44

DATA_ORIGIN = "https://edgefinder-pm8h.onrender.com"

# The measurement runs in the page. Kept as one expression so CDP can return
# it by value in a single round-trip.
MEASURE_JS = r"""
(() => {
  const de = document.documentElement, b = document.body;
  const vw = de.clientWidth;
  const docW = Math.max(de.scrollWidth, b.scrollWidth);
  const out = {
    viewport: vw,
    doc_width: docW,
    overflow: docW - vw,
    height: b.scrollHeight,
    blocks: [], offenders: [], small_targets: [], tiny_text: [],
  };
  const label = el => {
    const h = el.querySelector('h1,h2,h3,.c-card-title,[class*=title]');
    return ((h && h.textContent) || el.id || '.' + (el.className || ''))
      .toString().trim().slice(0, 48);
  };
  document.querySelectorAll(
    'section,.c-card,.desk-zone,[class*=zone],[class*=panel]'
  ).forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.height >= 40) out.blocks.push({label: label(el), h: Math.round(r.height)});
  });
  out.blocks.sort((a, b2) => b2.h - a.h);
  out.blocks = out.blocks.slice(0, 12);

  const seen = new Set();
  document.querySelectorAll('*').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return;
    const cls = (el.className || '').toString().slice(0, 44);
    const tag = el.tagName.toLowerCase();
    const txt = (el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 46);
    // Wide content inside a horizontal scroll container is REACHABLE — a
    // 7-column holdings table living in `.c-card-body { overflow-x: auto }` is
    // working as designed, not broken. Reporting it buries the genuine
    // offenders under a dozen <td>s, so walk up and suppress it.
    let inScroller = false;
    for (let p = el.parentElement; p && p !== b; p = p.parentElement) {
      const ox = getComputedStyle(p).overflowX;
      if (ox === 'auto' || ox === 'scroll') { inScroller = true; break; }
    }
    // sticks out past the right edge
    if (r.right > vw + 1 && !inScroller) {
      const k = 'o|' + tag + '|' + cls;
      if (!seen.has(k)) { seen.add(k);
        out.offenders.push({kind: 'past-viewport', tag, cls,
          over: Math.round(r.right - vw), w: Math.round(r.width), txt}); }
    }
    // own content wider than the box, and NOT in a scroll container
    if (el.scrollWidth > el.clientWidth + 2 && r.width > 60) {
      const cs0 = getComputedStyle(el);
      const scrolls = cs0.overflowX === 'auto' || cs0.overflowX === 'scroll';
      if (!scrolls) {
        const k = 'i|' + tag + '|' + cls;
        if (!seen.has(k)) { seen.add(k);
          out.offenders.push({kind: 'content-overflows', tag, cls,
            over: el.scrollWidth - el.clientWidth, w: Math.round(r.width), txt}); }
      }
    }
    const cs = getComputedStyle(el);
    // Only CONTROLS are held to the touch minimum. An inline link inside a
    // sentence ("…as AAPL reclaimed…") is legitimately text-sized — padding it
    // to 44px would wreck the paragraph, so `display: inline` is the signal
    // that this is prose, not a tap target. Without this the report drowns in
    // inline symbol links and the real offenders get ignored.
    const interactive = tag === 'button' || tag === 'select'
      || (tag === 'a' && cs.display !== 'inline')
      || (cs.cursor === 'pointer' && cs.display !== 'inline');
    // An icon or badge INSIDE a button is not its own tap target — the button
    // is. Flagging the 10px glyph inside a 44px button is a false positive
    // that buries the real ones.
    let coveredByAncestor = false;
    if (interactive) {
      for (let p = el.parentElement; p; p = p.parentElement) {
        const pt = p.tagName.toLowerCase();
        if (pt === 'button' || pt === 'a' || getComputedStyle(p).cursor === 'pointer') {
          const pr = p.getBoundingClientRect();
          if (pr.height >= TOUCH_MIN_PX && pr.width >= TOUCH_MIN_PX) {
            coveredByAncestor = true;
          }
          break;
        }
      }
    }
    if (interactive && !coveredByAncestor && (el.textContent || '').trim()
        && (r.height < TOUCH_MIN_PX || r.width < TOUCH_MIN_PX)) {
      const k = 't|' + tag + '|' + cls;
      if (!seen.has(k)) { seen.add(k);
        out.small_targets.push({tag, cls, w: Math.round(r.width),
          h: Math.round(r.height), txt: txt.slice(0, 22)}); }
    }
    const fs = parseFloat(cs.fontSize);
    if (fs && fs < 11 && !el.children.length
        && (el.textContent || '').trim().length > 2) {
      const k = 'f|' + cls + '|' + fs;
      if (!seen.has(k)) { seen.add(k); out.tiny_text.push({cls, fs}); }
    }
  });
  return out;
})()
"""


def find_chrome(explicit: str | None) -> str:
    if explicit:
        return explicit
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or "/opt/pw-browsers"]
    pats = ["chromium-*/chrome-linux/chrome", "chromium/chrome-linux/chrome",
            "chromium-*/chrome-linux/headless_shell"]
    for root in roots:
        for pat in pats:
            hits = sorted(glob.glob(os.path.join(root, pat)))
            if hits:
                return hits[-1]
    for cand in ("/usr/bin/chromium", "/usr/bin/chromium-browser",
                 "/usr/bin/google-chrome"):
        if os.path.exists(cand):
            return cand
    raise SystemExit("no Chromium found — pass --chrome /path/to/chrome")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# The local CDP endpoint must NOT go through an outbound proxy (an agent proxy
# answers 405 for localhost); the data origin must.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class DataProxy:
    """Fetches (and caches) API responses from a live origin."""

    def __init__(self, origin: str):
        self.origin = origin.rstrip("/")
        self.cache: dict[str, tuple[int, bytes]] = {}

    def get(self, path: str) -> tuple[int, bytes]:
        if path not in self.cache:
            try:
                with urllib.request.urlopen(self.origin + path, timeout=30) as r:
                    self.cache[path] = (r.status, r.read())
            except Exception as exc:  # noqa: BLE001 — report, don't crash the run
                self.cache[path] = (
                    502, json.dumps({"error": str(exc)}).encode())
        return self.cache[path]


async def audit(url: str, *, width: int, height: int, mobile: bool,
                chrome: str, origin: str, settle: float,
                screenshot: str | None) -> dict:
    import websockets

    port = free_port()
    profile = tempfile.mkdtemp(prefix="mobile-audit-")
    proc = subprocess.Popen(
        [chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
         "--hide-scrollbars", "--no-proxy-server",
         f"--user-data-dir={profile}", f"--remote-debugging-port={port}",
         "--remote-allow-origins=*", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws_url = None
        for _ in range(60):
            try:
                targets = json.load(
                    _DIRECT.open(f"http://127.0.0.1:{port}/json/list"))
                pages = [t for t in targets if t.get("type") == "page"]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:  # noqa: BLE001 — browser still starting
                pass
            time.sleep(0.5)
        if not ws_url:
            raise SystemExit("Chromium did not expose a CDP target")

        proxy = DataProxy(origin)
        async with websockets.connect(ws_url, max_size=80 * 1024 * 1024) as ws:
            counter = [0]

            async def send(method, params=None):
                counter[0] += 1
                await ws.send(json.dumps({"id": counter[0], "method": method,
                                          "params": params or {}}))
                return counter[0]

            async def on_event(msg):
                if msg.get("method") != "Fetch.requestPaused":
                    return
                p = msg["params"]
                rid, req_url = p["requestId"], p["request"]["url"]
                # An SSE stream never completes, so virtual time never settles.
                if "/api/desk/stream" in req_url:
                    await send("Fetch.failRequest",
                               {"requestId": rid, "errorReason": "Aborted"})
                    return
                path = None
                if "/api/" in req_url:
                    path = "/api/" + req_url.split("/api/", 1)[1]
                if path:
                    status, body = proxy.get(path)
                    await send("Fetch.fulfillRequest", {
                        "requestId": rid, "responseCode": status,
                        "responseHeaders": [{"name": "content-type",
                                             "value": "application/json"}],
                        "body": base64.b64encode(body).decode()})
                else:
                    await send("Fetch.continueRequest", {"requestId": rid})

            async def cmd(method, params=None):
                mid = await send(method, params)
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == mid:
                        return msg
                    await on_event(msg)

            await cmd("Emulation.setDeviceMetricsOverride",
                      {"width": width, "height": height,
                       "deviceScaleFactor": 2 if mobile else 1,
                       "mobile": mobile})
            await cmd("Page.enable")
            await cmd("Fetch.enable", {"patterns": [{"urlPattern": "*"}]})
            await send("Page.navigate", {"url": url})

            deadline = asyncio.get_event_loop().time() + settle
            while asyncio.get_event_loop().time() < deadline:
                try:
                    await on_event(json.loads(
                        await asyncio.wait_for(ws.recv(), timeout=1.0)))
                except asyncio.TimeoutError:
                    pass

            expr = MEASURE_JS.replace("TOUCH_MIN_PX", str(TOUCH_MIN))
            res = await cmd("Runtime.evaluate",
                            {"expression": expr, "returnByValue": True})
            val = res["result"]["result"].get("value")
            if val is None:
                raise SystemExit("measurement failed: "
                                 + json.dumps(res)[:600])
            val["endpoints_proxied"] = len(proxy.cache)
            val["api_errors"] = sorted(
                p for p, (s, _) in proxy.cache.items() if s != 200)
            if screenshot:
                shot = await cmd("Page.captureScreenshot",
                                 {"format": "png", "captureBeyondViewport": True})
                with open(screenshot, "wb") as fh:
                    fh.write(base64.b64decode(shot["result"]["data"]))
                val["screenshot"] = screenshot
            return val
    finally:
        proc.kill()
        proc.wait(timeout=10)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000/desk")
    ap.add_argument("--width", type=int, default=390)
    ap.add_argument("--height", type=int, default=844)
    ap.add_argument("--desktop", action="store_true",
                    help="measure as desktop (no mobile emulation) — the "
                         "no-regression check")
    ap.add_argument("--chrome", default=None)
    ap.add_argument("--data-origin", default=DATA_ORIGIN,
                    help="origin the /api/* responses are proxied from")
    ap.add_argument("--settle", type=float, default=25.0,
                    help="seconds to let the page fetch and paint")
    ap.add_argument("--max-height", type=int, default=DEFAULT_MAX_HEIGHT)
    ap.add_argument("--max-overflow", type=int, default=DEFAULT_MAX_OVERFLOW)
    ap.add_argument("--screenshot", default=None)
    ap.add_argument("--json", action="store_true", help="raw JSON only")
    a = ap.parse_args(argv)

    r = asyncio.run(audit(
        a.url, width=a.width, height=a.height, mobile=not a.desktop,
        chrome=find_chrome(a.chrome), origin=a.data_origin,
        settle=a.settle, screenshot=a.screenshot))

    if a.json:
        print(json.dumps(r, indent=2))
    else:
        screens = r["height"] / max(a.height, 1)
        print(f"\n{a.url}  @ {r['viewport']}x{a.height}"
              f"{'' if a.desktop else ' (mobile)'}")
        print(f"  page height      {r['height']:>8,}px   ({screens:.1f} screens)")
        print(f"  horiz overflow   {r['overflow']:>8,}px   "
              f"(doc {r['doc_width']}px vs viewport {r['viewport']}px)")
        print(f"  endpoints proxied{r['endpoints_proxied']:>8}")
        if r["api_errors"]:
            print("  ! non-200 API paths (measurement may be unpopulated):")
            for p in r["api_errors"]:
                print(f"      {p}")
        print("\n  tallest blocks")
        for b in r["blocks"]:
            print(f"    {b['h']:>7,}px  {b['h']/max(a.height,1):>5.1f} scr  "
                  f"{b['label']}")
        if r["offenders"]:
            print("\n  overflowing elements")
            for o in sorted(r["offenders"], key=lambda x: -x["over"]):
                print(f"    +{o['over']:>5}px  {o['kind']:<17} "
                      f"{o['tag']}.{o['cls'][:30]:<30} {o['txt'][:34]!r}")
        if r["small_targets"]:
            print(f"\n  tap targets under {TOUCH_MIN}px")
            for t in r["small_targets"]:
                print(f"    {t['w']:>4}x{t['h']:<4} {t['tag']}.{t['cls'][:34]:<34}"
                      f" {t['txt']!r}")
        if r["tiny_text"]:
            print("\n  text under 11px")
            for t in r["tiny_text"]:
                print(f"    {t['fs']}px  .{t['cls']}")
        if r.get("screenshot"):
            print(f"\n  screenshot -> {r['screenshot']}")

    fails = []
    if r["overflow"] > a.max_overflow:
        fails.append(f"horizontal overflow {r['overflow']}px "
                     f"> {a.max_overflow}px")
    if not a.desktop and r["height"] > a.max_height:
        fails.append(f"page height {r['height']:,}px > {a.max_height:,}px")
    # An empty page trivially passes the geometry gates — refuse to call that a
    # pass, since the whole point is measuring a POPULATED desk. Zero proxied
    # endpoints means the page never ran its fetches at all (dead server, wrong
    # URL, a browser error page), which is the most misleading pass of all: it
    # reports ~350px and 0px overflow and looks like a win.
    if r["endpoints_proxied"] == 0:
        fails.append("no /api/ requests were intercepted — the page did not "
                     "load (is the server up at --url?); geometry below is "
                     "meaningless")
    if r["api_errors"]:
        fails.append(f"{len(r['api_errors'])} API path(s) did not return 200 — "
                     "measured page may be unpopulated")
    # With --json, stdout must stay a single parseable document — the verdict
    # goes to stderr so `... --json | jq` works.
    out = sys.stderr if a.json else sys.stdout
    if fails:
        print("\nFAIL", file=out)
        for f in fails:
            print("  -", f, file=out)
        return 1
    print("\nPASS", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
