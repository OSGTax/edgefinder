import { load, save } from '../engine/storage';
import { settings } from '../ui/settings';

// The phone-app shell around the game: offline play (service worker), the
// home-screen install hints, full screen + landscape where the platform allows,
// and stopping browser gestures
// (pinch, double-tap zoom, long-press menus) from fighting the game. Keeping
// the screen awake and auto-pause during play live in the game screen.
// Everything here is best-effort: any API can be missing or refuse, and the
// game must carry on regardless.

type InstallPrompt = Event & { prompt(): Promise<void>; userChoice: Promise<{ outcome: string }> };

const ua = typeof navigator !== 'undefined' ? navigator.userAgent : '';
const isIOS = /iPad|iPhone|iPod/.test(ua) || (typeof navigator !== 'undefined' && navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
const isTouchPhone = () => matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) < 820;
const standalone = () => matchMedia('(display-mode: standalone), (display-mode: fullscreen)').matches || (navigator as Navigator & { standalone?: boolean }).standalone === true;

class Shell {
  /** a newer version has been downloaded; a reload will use it */
  updateWaiting = false;
  private updateCbs: (() => void)[] = [];
  private prompt: InstallPrompt | null = null;

  init() {
    this.guardGestures();
    window.addEventListener('beforeinstallprompt', (e) => { e.preventDefault(); this.prompt = e as InstallPrompt; this.fire(); });
    window.addEventListener('appinstalled', () => { this.prompt = null; this.fire(); });
    // full screen on the first tap (Android and friends; iOS Safari has no full screen API for pages)
    const first = () => { window.removeEventListener('pointerup', first); this.goFullscreen(); };
    window.addEventListener('pointerup', first);
    if (import.meta.env.PROD) this.registerWorker();
  }

  onUpdate(cb: () => void) { this.updateCbs.push(cb); }
  private fire() { for (const cb of this.updateCbs) try { cb(); } catch { /* a screen went away */ } }

  // ── offline ────────────────────────────────────────────────────────────

  private registerWorker() {
    if (!('serviceWorker' in navigator)) return;
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('./sw.js', { scope: './' }).then((reg) => {
        reg.addEventListener('updatefound', () => {
          const w = reg.installing;
          w?.addEventListener('statechange', () => {
            // a new worker took over: if it doesn't carry this page's own script, this page is an old build
            if (w.state === 'activated' && navigator.serviceWorker.controller) this.checkStale();
          });
        });
        // look for a new build on every launch (browsers may skip their own check) ...
        if (reg.active) reg.update().catch(() => {});
        // ... and when the app comes back to the front: phones keep a PWA open for days
        document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') reg.update().catch(() => {}); });
      }).catch(() => { /* no offline play; everything else works */ });
    });
  }

  private async checkStale() {
    const src = document.querySelector<HTMLScriptElement>('script[type="module"][src*="assets/"]')?.src;
    if (!src || !('caches' in window)) return;
    try {
      if (!(await caches.match(src))) { this.updateWaiting = true; this.fire(); }
    } catch { /* no cache access: say nothing */ }
  }

  // ── install hints ──────────────────────────────────────────────────────

  /** What to suggest on the title screen: iOS "Add to Home Screen", the browser's own prompt, or nothing. */
  installHint(): 'ios' | 'prompt' | null {
    if (standalone()) return null;
    if (this.prompt) return 'prompt';
    if (isIOS && isTouchPhone() && !load('installHintDismissed', false)) return 'ios';
    return null;
  }

  dismissInstall() { save('installHintDismissed', true); }

  async install() {
    const p = this.prompt;
    if (!p) return;
    this.prompt = null;
    try { await p.prompt(); await p.userChoice; } catch { /* dismissed */ }
  }

  // ── full screen, orientation ────────────────────────────

  /** Must run inside a tap. */
  goFullscreen() {
    if (!settings.fullscreen || isIOS || !isTouchPhone() || standalone() || document.fullscreenElement) return;
    const el = document.documentElement;
    if (!el.requestFullscreen) return;
    el.requestFullscreen({ navigationUI: 'hide' }).then(() => {
      const o = screen.orientation as ScreenOrientation & { lock?: (o: string) => Promise<void> };
      return o?.lock?.('landscape');
    }).catch(() => { /* refused: fine */ });
  }

  /** A game is starting (called from the tap that starts it): go full screen if we can. */
  enterGame() {
    this.goFullscreen();
  }

  // ── gestures ───────────────────────────────────────────────────────────

  private guardGestures() {
    const stop = (e: Event) => e.preventDefault();
    // iOS pinch zoom (ignores user-scalable=no)
    document.addEventListener('gesturestart', stop, { passive: false } as AddEventListenerOptions);
    // two-finger pinch on Android
    document.addEventListener('touchmove', (e) => { if (e.touches.length > 1) e.preventDefault(); }, { passive: false });
    // double-tap zoom is off through CSS touch-action (none on the game, pan-y on menus)
    document.addEventListener('dblclick', stop, { passive: false });
    // long-press menus (keep them for text fields)
    document.addEventListener('contextmenu', (e) => { if (!(e.target as HTMLElement).closest('input, textarea')) e.preventDefault(); });
  }
}

export const shell = new Shell();
