# Grass Stain League — developer map

A backyard-baseball game (working title) for web, iOS and Android. The owner
is the product owner, not a developer — see `GAME-PLAN.md` for the vision,
status and roadmap in plain language, and keep it current when phases change.

**Direction change (v2, owner decision):** the game is moving to real 3D with
much more realistic graphics. See `docs/3D-FRAMEWORK.md` (Three.js renderer on
top of the existing `src/sim`; glTF characters) and `docs/LEAGUE-FRAMEWORK.md`
(10 teams × 9 kids, 4 traits: Hitting/Speed/Fielding/Pitching, everyone pitches,
no benches/injuries/trades, no pick-up draft, one home field per team). The
owner asked for frameworks/plans first; don't build large pieces of the 3D
version until the art-source decision in `docs/3D-FRAMEWORK.md` is made.

**v0.1 asset rule (still true for the current prototype):** every asset is
generated in code. The 3D version will add real 3D models and textures for
characters; yards, sound and music should stay code-generated where possible.

**Originality rule:** inspired by 90s backyard baseball games, but never use
their names, characters, art, music or trademarks ("Backyard Baseball",
"Sandlot"), and no real MLB teams/players.

## Stack

TypeScript (strict) + Vite, no framework. Canvas 2D for the game scene
(pseudo-3D via our own perspective camera), plain DOM + CSS for HUD and
menus, Web Audio for sound. Vitest for tests. Mobile builds will wrap the web
build with Capacitor (not set up yet — Phase 5 in the plan).

## Commands

```bash
npm install
npm run dev            # http://localhost:5173  (dev-only: #gallery, #scene=<yardId>&cam=bat|field|overview)
npm test               # vitest: sim balance, season, human-hitting, audio
npm run typecheck
npm run build          # dist/ (normal hashed assets)
npm run build:web      # dist-single/index.html (one file) + dist-single/page.html (body-only, for hosts that add their own <head>)
```

The published preview lives at https://claude.ai/artifact/KGG4KBQdjTg3MHKAyHEP1U
— republish `dist-single/page.html` there after `npm run build:web`.

## Layout

```
src/
  engine/   math (Vec, segIntersect...), seeded Rng, safe localStorage
  data/     types.ts (Kid/Team/Yard...), kids.ts (72 mini-adult kids),
            teams.ts (8 teams + announcers), yards.ts (8 yards), palette.ts
  sim/      the game engine — pure, no DOM, deterministic from a seed
    field.ts     yard geometry, surfaces, obstacles, fair/foul, zone
    physics.ts   ball flight/bounce/roll, fences, trees, boxes; predict()
    pitching.ts  pitch types, trajectories (break, knuckle wobble, Brain Freeze time-warp)
    batting.ts   swing timing+aim → contact, exit velo, launch, spray
    play.ts      LivePlay: fielder AI (intercepts, covers, throws), runner AI, outs, end of play
    ai.ts        CPU pitch selection and swing decisions
    match.ts     Match: counts/innings/score/box score/hype/specials; simulateMatch() headless
    lineup.ts    auto positions + batting order
  league/   season.ts (schedule, standings, leaders, playoffs), draft.ts (pick-up game)
  art/      kid.ts (procedural chibi kids + costume pieces + portraits), yard.ts
            (ground, fences, props), grownup.ts, logo.ts, draw3d.ts (projected primitives)
  render/   camera.ts (perspective camera, plate-plane raycast), cameras.ts (presets), scene.ts
  audio/    Web Audio synth: sfx, music sequencer + songs, ambience (no-op without AudioContext)
  ui/       app.ts (menus, season hub, draft UI), game.ts (game screen: input, camera director,
            HUD, events → sounds/popups/commentary), commentary.ts (Chet & Dottie), style.css
  dev/      gallery/scenes — dev-only art previews
tests/      sim balance + determinism, season, human-hitting, audio
scripts/    web-page.mjs (single-file build → body-only page)
```

World units are feet. Home plate is the origin, +y toward second base/center
field, +x toward first base, +z up. Bases are 60 ft apart (Little League).

## How the engine fits together

- `Match.update(dt)` drives phases: `prePitch → windup → pitch → (live) → result → halfOver → over`.
- Human input enters via `Match.selectPitch / swing / throwTo / runners`; anything
  the human doesn't decide, the CPU does (throws after `autoThrowDelay`).
- `LivePlay` owns fielders, runners and the ball during a ball in play and emits
  `MatchEvent`s; `GameScreen.handleEvents` turns events into sounds, popups and commentary.
- League games are simulated with the same engine (`simulateMatch`, ~0.3 s per 6-inning game).

## Balancing

`tests/sim.test.ts` prints league-wide numbers (AVG, runs, HR, K%, BB%, errors)
from many CPU games and asserts sane ranges; `tests/human.test.ts` checks a
pretend human can hit on Rookie and that All-Star is harder. When changing
physics/AI constants, run them and look at the printed summary. Current
targets: AVG ~.33–.37, 4–5 runs per team per 6 innings, ~1 HR/team-game,
2–3 errors/game, K% ~10–14%.

## Verifying visually

Chromium + Playwright are preinstalled in the cloud container. Run `npm run dev`
and screenshot with Playwright (import from `$(npm root -g)/playwright/index.mjs`);
`#gallery` shows every kid and pose, `#scene=<yard>&cam=…` shows a yard.

## Conventions

- Keep `src/sim` free of DOM/canvas so it runs headless in tests.
- Determinism: all randomness goes through the match/season `Rng`.
- Storage must stay optional (wrapped in try/catch) — the game must run without it.
- No `alert/confirm/prompt` (blocked in embedded hosts) — use in-page UI.
- Test gate before committing: `npm run typecheck && npm test && npm run build`.
