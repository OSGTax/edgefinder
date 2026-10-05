# Grass Stain League — developer map

A backyard-baseball game (working title) for web, iOS and Android. The owner
is the product owner, not a developer — see `GAME-PLAN.md` for the vision,
status and roadmap in plain language, and keep it current when phases change.

**Current state: the 3D demo** — two teams (Maple Street Mudcats at Cedar Lane
Comets), one heavily detailed yard (Pool Party Paradise), real 3D kids and
gameplay. This is the quality bar for the final game; the owner wants the
demo polished before more teams/yards are built. League build-out plans:
`plans/LEAGUE-FRAMEWORK.md` (10 teams × 9 kids, 7 traits: Contact/Power/
Speed/Fielding/Arm/Pitching/Control, everyone pitches, no benches/injuries/trades, no pick-up
draft, one home field per team) and `plans/3D-FRAMEWORK.md`. The v0.1 2D kids
and yards live in git history (commit `a00d3e8`).

**Asset rule:** every asset is generated in code — 3D geometry, textures,
faces, uniforms, sound, music. No model, image or audio files.

**Originality rule:** inspired by 90s backyard baseball games, but never use
their names, characters, art, music or trademarks ("Backyard Baseball",
"Sandlot"), and no real MLB teams/players.

## Stack

TypeScript (strict) + Vite, Three.js (WebGL2) for the game scene, plain DOM +
CSS for HUD and menus, Web Audio for sound. Vitest for tests. Mobile builds
will wrap the web build with Capacitor (not set up yet).

## Commands

```bash
npm install
npm run dev            # http://localhost:5173
npm test               # vitest: sim balance + determinism, human-hitting, kid traits, audio
npm run typecheck
npm run build          # dist/ (hashed assets; deploy this folder)
```

Dev-only URL hashes (combine with `&`): `#gallery` (all kids; `&faces`,
`&poses=run,swing,...&t=0.3`, `&kid=bo`, `&expr=yell`), `#dev=<cam>` (empty
yard from a named camera: bat, field, house, patio, pool, cf, street, high,
dugout...), `#play=mudcats|comets` (skip menus; `&cpu` = CPU vs CPU),
`q=low|medium|high` (force graphics tier), `ff=N` (simulate N fixed steps per
rendered frame — for the slow software renderer in headless Chromium).

## Layout

```
src/
  engine/   math, seeded Rng, safe localStorage, steps (generator builds paced for the loading bar)
  data/     types.ts (Kid/Traits/Team/Yard...), kids.ts (18 kids), teams.ts (2 teams +
            announcers), yards.ts (Pool Party Paradise), palette.ts (skin/hair)
  sim/      the game engine — pure, no DOM, deterministic from a seed (unchanged from v0.1
            apart from traits + pitching changes): field, physics, pitching, batting,
            play (LivePlay fielder/runner AI), ai, match (phases, specials, simulateMatch), lineup
  gfx/      Three.js basics: renderer, quality tiers, sky + sun + PMREM environment,
            ground (splat-map lawn shader + instanced grass blades), procedural canvas
            textures + normal maps, material library, geometry batching (merge per material)
  world/    the ballpark: stadium.ts (assembles everything, LAYOUT of patio/dugouts),
            house.ts, pool.ts (water + caustics shaders), fences.ts (pickets, hedge, leaf
            cards), trees.ts (procedural trees + far-tree blobs), props.ts (patio set, grill,
            bases, flamingos, dugouts...), neighborhood.ts (houses, street, poles, water tower)
  kid3d/    3D kids: rig.ts (skeleton + proportions from KidLook), geom.ts (lofts, limbs,
            skin weights), model.ts (KidModel: one skeleton, ~7 skinned meshes), face.ts
            (painted expression atlas), uniform.ts (jersey texture, numbers), costume.ts
            (hair, hats, persona pieces), items.ts (bat, glove, ball, props), anim.ts
            (Animator: poses, cycles, IK, look-at, blinks), ik.ts
  game/     world.ts (World: stadium + 18 Actors + ball + overlays; sync(match) maps sim
            state to kids each frame), director.ts (camera shots), screen.ts (GameScreen:
            HUD, input, events → sounds/popups/fx/commentary), fx.ts (particles, ball trail),
            portraits.ts (3D portraits for HUD/menus)
  audio/    Web Audio synth: sfx, music sequencer + songs, ambience
  ui/       app.ts (loading, title over an attract-mode CPU game, team pick, roster, how-to,
            settings), commentary.ts (Chet & Dottie), settings.ts, dom.ts, style.css
  dev/      view3d.ts (#dev), gallery3d.ts (#gallery) — dev-only
tests/      sim balance + determinism, human-hitting, kid traits, audio
```

World units are feet. **Sim coordinates:** home plate is the origin, +y toward
second base/center field, +x toward first base, +z up. **Three.js coordinates:**
`three = (sim.x, sim.z, -sim.y)` — use `W(x, y, z)` from `gfx/units.ts`; a sim
facing angle `f` (0 = +y, clockwise) becomes yaw `π - f` (`yawOf`). Kid models
face local +z; their left is +x.

## How the engine fits together

- `Match.update(dt)` drives phases: `prePitch → windup → pitch → (live) → result → halfOver → over`.
- Human input enters via `Match.selectPitch / swing / throwTo / runners`; anything
  the human doesn't decide, the CPU does.
- `LivePlay` owns fielders, runners and the ball during a ball in play and emits
  `MatchEvent`s; `GameScreen.handleEvents` turns them into sounds, popups, effects and commentary.
- `World.sync(match)` decides where every kid should be and in which animation `Mode`
  (`kid3d/anim.ts`); kids not in the play jog to their team's dugout.
- The app builds one `World` at startup and reuses it for the title attract game and
  every match. `World.build()` / `Stadium.build()` are generators of loading steps
  (`engine/steps.ts`): `runPaced` lets the loading bar paint between them, `runNow`
  builds straight through.

## Balancing

`tests/sim.test.ts` prints league-wide numbers from many CPU games and asserts sane
ranges; `tests/human.test.ts` checks a pretend human can hit on Rookie and that
All-Star is harder. Current (300-game sample): AVG ~.335, ~4 runs and ~1.3 HR per
team per 6 innings, ~2.8 errors/game, K% ~11%, BB% ~2.5%.

Where each trait (`data/types.ts` `Traits`, 1–10) feeds the sim: **contact** → swing
window, sweet-spot size, quality of near-misses, launch-angle consistency, CPU pitch
reading (`batting.ts`, `ai.ts`); **power** → exit velocity (contact adds a little),
CPU power swings; **speed** → runner and fielder speed; **fielding** → reaction,
reach, gather, catch odds, throw accuracy; **arm** → throw speed (plus a little
pitch velocity and throw accuracy); **pitching** → fastball mph; **control** → pitch
aim scatter; pitching + control → stamina and who relieves (`match.ts`).
`lineup.ts` picks positions and batting order from them.

## Verifying visually

Chromium + Playwright are preinstalled in the cloud container (software WebGL via
SwiftShader: ~1 frame/s, so use `q=low` and `ff=`). Launch with
`--enable-unsafe-swiftshader --ignore-gpu-blocklist`, `goto(..., {waitUntil:'commit'})`,
and wait for `window.__ready` (dev views) or `window.__game.ready` (games).

## Conventions

- Keep `src/sim` free of DOM/Three so it runs headless in tests.
- Determinism: all game randomness goes through the match `Rng` (cosmetic randomness in
  the renderer is fine).
- Storage must stay optional (wrapped in try/catch) — the game must run without it.
- No `alert/confirm/prompt` — use in-page UI.
- Static scenery: add primitives to a `Batch` (one mesh per material). Textures are
  painted neutral and tinted per material (`gfx/materials.ts`) — painting a new texture
  per colour costs seconds at load.
- Test gate before committing: `npm run typecheck && npm test && npm run build`.
