# Grass Stain League — developer map

A backyard-baseball game (working title) for web, iOS and Android. The owner
is the product owner, not a developer — see `GAME-PLAN.md` for the vision,
status and roadmap in plain language, and keep it current when phases change.

**Current state: the 3D demo** — two teams (Maple Street Mudcats at Cedar Lane
Comets), one heavily detailed yard (Pool Party Paradise), real 3D kids and
gameplay. This is the quality bar for the final game; the owner wants the
demo polished before more teams/yards are built. **The mobile revamp (Oct 2026)** made it
phone-first and cartoon: thumb controls, toon-shaded kids with friendly faces and per-kid
personality, comic-book pop-ups on a clean screen, a cartoon UI kit, kid-band sound, a PWA
shell, adaptive graphics tiers. `plans/MOBILE-REVAMP.md` holds the owner's art direction
(fun, cartoon, comical, uncluttered; the "not AI-generated" checklist) — follow it. League build-out plans:
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
npm run deploy         # build docs/ and publish it to the gh-pages branch → https://osgtax.github.io/edgefinder/
```

`npm run dev` and plain `vite build` drop the dev views; for headless checks build a dev-mode
copy and serve it: `NODE_ENV=development npx vite build --mode development --outDir <dir>`,
then `npx vite preview --outDir <dir> --port 4180`.

Dev-only URL hashes (combine with `&`): `#gallery` (all kids; `&grid` contact sheet,
`&faces`, `&exprs`, `&atlas`, `&lite`, `&portraits`, `&poses=run,swing,...&t=0.3`,
`&kid=bo`, `&expr=yell`, `&zoom=2`, `&yaw=`), `#dev=<cam>` (empty yard from a named
camera: bat, field, house, patio, pool, cf, street, high, dugout...), `#play=mudcats|comets`
(skip menus; `&cpu` = CPU vs CPU), `#look` (UI kit specimen), `#menu=pick|kids|kid&kid=bo|how|settings`,
`#loadhold` (freeze the loading screen), `q=low|medium|high` (force graphics tier; also
disables the tier governor), `ff=N` (simulate N fixed steps per rendered frame — for the
slow software renderer). `window.__holdPops = true` freezes the last comic pop-up.
`#sounds` (the sound board) also works in production. Changing only the `#hash` doesn't
reload the page: in Playwright `goto('about:blank')` first.

## Layout

```
src/
  engine/   math, seeded Rng, safe localStorage, steps (generator builds paced for the loading bar)
  shell/    phone-app shell: service worker registration + update check, fullscreen/landscape
            on Android, install hints, gesture guards (wake lock + auto-pause live in GameScreen)
  data/     types.ts (Kid/Traits/Team/Yard...), kids.ts (18 kids), teams.ts (2 teams +
            announcers), yards.ts (Pool Party Paradise), palette.ts (skin/hair)
  sim/      the game engine — pure, no DOM, deterministic from a seed (unchanged from v0.1
            apart from traits + pitching changes): field, physics, pitching, batting,
            play (LivePlay fielder/runner AI), ai, match (phases, specials, simulateMatch), lineup
  gfx/      Three.js basics: renderer, quality.ts (tiers, device/GPU probe, learned tier per
            chip, prefs), governor.ts (pure frame-time tier governor), perf.ts (speed readout),
            contact.ts (soft contact shadows), sky + sun + PMREM environment (Neutral tone
            mapping), ground (splat-map lawn + instanced grass blades), procedural textures,
            material library, geometry batching (merge per material; plain paints folded)
  world/    the ballpark: stadium.ts (assembles everything, LAYOUT of patio/dugouts),
            house.ts, pool.ts (water + caustics shaders), fences.ts (pickets, hedge, leaf
            cards), trees.ts (procedural trees + far-tree blobs), props.ts (patio set, grill,
            bases, flamingos, dugouts...), neighborhood.ts (houses, street, poles, water tower)
  kid3d/    3D kids: rig.ts (skeleton + proportions from KidLook), geom.ts (lofts, limbs,
            skin weights), model.ts (KidModel: one skeleton, skinned meshes; setDetail('lite'
            |'full'), setLids, setOutline), toon.ts (toon ramp + ink outline), face.ts (painted
            expression atlas, 4×3 incl. laugh), face-recipes.ts (a face per kid), uniform.ts,
            costume.ts (hair, hats, persona pieces), items.ts, anim.ts (Animator: modes,
            cycles, IK, look-at, blinks, squash & stretch), pose.ts (allocation-free keyframe
            tracks), personality.ts (per-kid stance, gait, idles, fidgets, celebrations), ik.ts
  game/     world.ts (World: stadium + 18 Actors + ball + overlays; sync(match) maps sim
            state to kids; detects moments (World.mo: HR, splash, out, K, run, game over) for
            bench/fielder reactions; syncMendoza (grill, watching, the pool skimmer trip);
            budgetKids: shadows, culling, lite models), director.ts (camera shots, phone framing), screen.ts (GameScreen: thumb
            controls, pitch meter, swing reads, coach, HUD from the look kit, events →
            pop-ups/fx/commentary), comic.ts (pop-up moments, words, no-repeat), emotes.ts
            (stars, sweat drops, "!" over heads), fx.ts (cartoon particles, ball trail),
            portraits.ts (3D portraits for HUD/menus)
  audio/    Web Audio: context (bus, phone lifecycle, limiter), dsp, bake (pre-rendered
            textures), instruments (kid band), notation, tracks, music (sequencer), sfx
            (+ comic stingers), ambience, voice + voices (formant gibberish per kid),
            cues (GameSound: match events → crowd, barks, voices, surfaces)
  ui/       app.ts (loading, title over an attract-mode CPU game, team pick, Meet the Kids
            trading cards, how-to, settings), commentary.ts (Chet & Dottie; never repeats a
            line or template in a game), settings.ts, dom.ts, base.css (tokens), menus.css,
            hud.css (in-game screen), look/ (the cartoon UI kit: comic lettering, comicPop
            bursts, code-drawn icons, components, look.css; guide in look/README.md)
  dev/      view3d.ts (#dev), gallery3d.ts (#gallery), soundboard.ts (#sounds)
build/      pwa.ts (Vite plugin: manifest, code-painted icons, favicon, sw.js per build)
tests/      sim balance + determinism, human-hitting, kid traits, faces, controls, quality
            governor, commentary, audio (+ sample-accurate render checks via fake-audio.ts)
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
- Graphics: `World.q` is the tier the yard was built at, `World.tier` the live tier;
  `TierGovernor` steps tiers by measured frame times and `World.setTier()` applies
  resolution, shadow map, grass and kid budgets live (textures/leaves next visit).
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
- UI: display text uses `lettering(text, { style: 'comic' })`, icons come from `icon()`, big
  moments use `comicPop()`; **no emoji** anywhere. Keep the screen clean (see the brief).
- Every path stays relative (the site is served from a subpath). The service worker only
  runs in production builds; never strand players on an old build (index is network-first).
- Test gate before committing: `npm run typecheck && npm test && npm run build`.
