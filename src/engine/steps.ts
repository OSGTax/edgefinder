// Long builds written as generators so the loading screen can breathe.
// Each yield says what's about to happen and how far along we are (0..1).

export interface Step { done: number; msg: string }
export type Steps = Generator<Step, void, void>;

/** Run every step straight through (dev views, tests). */
export function runNow(steps: Steps) {
  while (!steps.next().done) { /* keep going */ }
}

/** Resolves once the browser has had a chance to paint. */
export const nextPaint = () => new Promise<void>((ok) => requestAnimationFrame(() => setTimeout(ok, 0)));

/** Run the steps, letting the browser paint between them. */
export async function runPaced(steps: Steps, onStep: (s: Step) => void) {
  for (let r = steps.next(); !r.done; r = steps.next()) {
    onStep(r.value);
    await nextPaint();
  }
}
