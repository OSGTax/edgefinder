# Mobile revamp — shared brief for every helper session

The owner's words (October 2026):

- "Completely revamp and upscale this game quality and playability on mobile phones."
- "Gameplay needs to present as not AI generated."
- "We need to spend some more time on faces as they look kind of scary right now."
- On a computer it was "deathly slow"; on the owner's phone it ran fine.
- "It should be a deep, deep quality overview revamp. We're looking for fun and cartoon
  features. It needs to be comical but not too much going on in the screen, more
  cartoonish pop-ups like boom, slap, bam — but not like that, just in that style."

The phone in landscape is now the primary target: touch only, two thumbs, a screen
roughly 844×390 CSS px (iPhone 13/14) to 915×412 (common Android). Tablets and computers
must keep working with touch, mouse and keyboard, but every decision is judged on a phone first.

Read `CLAUDE.md` (developer map) and `GAME-PLAN.md` (owner's plan) before starting.

---

## 1. Art direction: a Saturday-morning cartoon

**Fun, cartoon, comical — with a clean screen.** This is the owner's direction and it
outranks anything else in this brief.

- **Cartoon, not realistic.** Bold, clean shapes; bright color with simple shading; clean
  ink outlines where they help (characters and all UI). Exaggerated, readable characters
  with big expressive features. Squash and stretch on whatever moves: kids, the ball on
  impact, the UI pop-ups.
- **Not too much on the screen.** At any moment the player sees the game plus only what
  they need right now. The persistent HUD is at most a small score/inning/count bug, the
  controls for the current action, and pause. Everything else is transient: the batter's
  name card slides in when they step up and gets out of the way; a caption is one short
  line and leaves; no stat bars or paragraphs during play. When in doubt, leave it out.
- **Comic-book pop-ups for the big moments** — the game's signature. Starburst or jagged
  balloon shapes, bold slanted letters with a thick ink outline and an offset shadow,
  halftone dots and speed lines, a punchy scale-in with overshoot and a little shake, gone
  in about a second. The words are **our own, specific to the moment** (e.g. a crushed ball
  "THWACK!", a whiff "WHIFF!", a diving grab "SNAG!", into the pool "SPLOOSH!", off the
  fence "BONK!", strike three "SIT DOWN!", a home run "SEE YA!"), not the old-TV
  "BAM / POW / ZAP" set. Used sparingly so they stay special: real moments only, never on
  routine pitches, and never the same word twice in a row.
- **Comical in the world.** Cartoon touches on the field: dust puffs and speed lines,
  stars circling a kid who stumbles, a sweat drop on a pitcher in a jam, a "!" over a
  fielder about to make the play, exaggerated takes and celebrations. Funny, never mean,
  never noisy.
- **The humor stays authored.** Kids playing at being grown-ups; Chet & Dottie's homemade
  broadcast (*Channel 4½, Maple Hollow Public Access*) lives in the copy, not as clutter.

**Type.** Display lettering is *generated in code* in a bold comic style (slanted, thick
outline, a little hand-drawn wobble): titles, team names, score numbers, pop-ups. No web
fonts or font files. Body text uses a calm rounded system stack, sentence case, at least
13 CSS px on phones.

**Icons.** Drawn in code (SVG paths) in the same cartoon style: rounded, thick outline.
**No emoji anywhere** in the UI (today: ⚾ ⚡ 📱 ↻ ▶ ◀ are used — replace them all).

**Palette.** Bright, sunny and cheerful but harmonious: grass greens, sky blue, sunshine
yellow, tomato red, each team's colors, and a warm dark ink (not pure black) for outlines.
No neon gradients, no glossy pills.

**3D look.** Characters get toon-style shading (a soft two- or three-tone ramp), a clean
ink outline and simple saturated materials (the Faces helper owns kid materials; check the
outline's cost with the Speed helper; the lite model may drop it). The world goes
cartoon-friendly: simplified, saturated, soft shadows, no noisy realistic textures. The
Speed & graphics helper decides how far toward toon to push, with side-by-side
screenshots, so kids and yard sit together as one style.

### The "not AI-generated" checklist

| Reads as generated / templated | Reads as made by people |
| --- | --- |
| Emoji as icons | Code-drawn icons in one cartoon style |
| Gradient pill buttons, glossy circles | Bold cartoon shapes, ink outlines, flat color, a little hand-drawn wobble |
| A busy screen: panels, bars and labels all at once | Only what's needed now; cards come and go; big moments get a pop-up, routine ones don't |
| Clip-art "POW!" bursts | Our own words, drawn in our style, tied to the moment, never repeated back-to-back |
| One heavy system font for everything | Comic display lettering plus a calm body face, with clear hierarchy |
| Seven stat bars on every card | Two or three numbers that matter right now; the rest on demand |
| Placeholder/dev copy ("Watching the kids play", "Demo build…") | In-world copy written for this game |
| Repetition: same face, same line twice, same celebration | Per-kid variation; no-repeat logic; specific details |
| Uniform timing, symmetric poses, everything at once | Hand-tuned timing, anticipation and follow-through, squash and stretch |
| Same-looking faces with a color swap | Each kid designed as a cartoon character |

The originality rule still applies: inspired by 90s backyard baseball games, never
their names, characters, art, music or trademarks, and no real MLB teams or players.

---

## 2. Faces: friendly, never scary (top priority)

What's wrong today (see `#gallery&kid=ruby&q=high`): realistic eyeballs with small
irises and lots of white, heavy half-lowered upper lids and a dark lid line, so every
kid has the same flat suspicious stare; a long egg-shaped head with a big empty lower
face; a thin little mouth sitting low; identical features on every kid.

The target is appealing **cartoon** kids with big, simple, expressive features that read
at every size (the owner asked for "fun and cartoon features"):

1. **Eyes** — big dark irises and pupils filling most of the eye opening, clear
   catchlights, upper lids high at rest (lids come down only for blinks and
   expressions), little or no white showing at rest, set a bit lower and closer
   together. Painted/decal eyes are allowed if they read friendlier than 3D eyeballs;
   choose whatever looks best. The look-at and blink behavior must keep working.
2. **Head** — rounder, fuller cheeks, softer smaller chin; features centered lower,
   bigger forehead.
3. **Mouth** — larger and readable: a soft smile with defined corners at rest, a hint
   of lower lip, closer to the nose; a real expression range (open laugh, determined
   grit, surprised O, pout, yell). Expressions are cartoon "takes": eyes pop wide in
   surprise, squeeze shut in a laugh, brows do half the acting.
4. **Brows** — thicker, expressive, a shape per kid.
5. **Every kid a character** — a hand-picked face recipe per kid (eye shape, size and
   spacing, brow style, nose, mouth width, ears, freckles, gap tooth, cheeks) that fits
   their persona. Put recipes in a new `src/kid3d/face-recipes.ts` keyed by kid id,
   not in `data/kids.ts`.
6. **Skin and light** — warm shading, rosy cheeks, a soft rim so faces don't go
   dead-dark under cap brims; the cap shadow must never black out the eyes.

Check at four sizes: menu portrait (~150 px), HUD portrait (~50 px), batter at the
plate in the batting view, and an outfielder in the field view (a few pixels: eyes
should still read as two friendly dark dots, not white glare). The test: would a
parent call these kids cute?

---

## 3. Playability on a phone

- **Thumbs.** Batting must be playable with one thumb on Rookie; aiming is optional
  help, not a requirement. Pitching: pick a pitch, place it, throw, with big targets.
  Fielding: a thumb-reachable "throw to" diamond. Running: clear send/hold. Every
  touch target at least 48 CSS px, nothing important under a thumb, the notch or the
  home indicator (use safe-area insets).
- **Readability.** A compact HUD that works at 390 px tall; the ball always readable
  (size, trail, shadow, an off-screen marker for high flies); timing feedback on every
  swing (early / late / on time; contact quality).
- **Feel.** Weighty contact (sound, a frame of hit-stop, camera kick, slow-mo on a crush)
  and the comic pop-ups from §1 on the big moments;
  haptics where the platform allows; snappy pacing (skippable flyovers, short
  transitions); a 3-inning default on phones.
- **Onboarding.** A first-game coach that teaches swing, pitch, throw and run in
  context, once, and can be replayed from How to Play.
- **Interruptions.** Auto-pause when the app is hidden (calls, app switch), a clear
  resume, the screen kept awake while playing, the game never stuck after audio
  interruption.
- **App feel.** Installable to the home screen (generated icons, manifest),
  fullscreen and landscape where the platform allows, works offline after the first
  load, no browser gestures fighting the game (pull-to-refresh, double-tap zoom, long-press
  menus, text selection).

## 4. Speed

- **The desktop bug.** `gfx/quality.ts` `autoTier()` gives every non-phone the top
  tier ("Beautiful"), which is why a computer was "deathly slow". Start every device
  on Balanced, measure real frame times over the first seconds, and step the whole tier
  (not just resolution) down or up. Detect software rendering (no GPU: SwiftShader,
  llvmpipe, "Basic Render Driver") and go straight to Fast with a one-line tip about
  turning on graphics acceleration.
- **Targets.** 60 fps on recent phones on Balanced; a steady 30+ on older phones on
  Fast; never a thermal spiral (offer a 30 fps "battery saver" cap); survive WebGL
  context loss with a friendly recovery.
- **Budget the kids.** 18 kids at ~20k triangles each, drawn twice for shadows, is the
  biggest single cost. Far kids get a lite model; shadows are budgeted by distance and tier.
- **A speed readout** (fps and the graphics chip the browser reports), switchable in
  Settings, so the owner can send numbers from any device.

---

## 5. Who owns what (to keep parallel work mergeable)

Six helper sessions work in parallel, each on its own branch cut from
`ccr-92524618-3jffwa`. Stay inside your files. If you truly need a change elsewhere,
keep it tiny, additive and clearly commented, and mention it in your report.

| Helper | Branch | Owns |
| --- | --- | --- |
| **Faces & character appeal** | `ccr-92524618-3jffwa-m-faces` | `src/kid3d/{rig,geom,model,face,uniform,costume,items,outfits}.ts`, new `src/kid3d/face-recipes.ts`, `src/game/portraits.ts`, `src/world/grownups.ts`, `src/dev/gallery3d.ts`; in `kid3d/anim.ts` only the eye, lid, blink and expression code |
| **Animation & personality** | `ccr-92524618-3jffwa-m-anim` | `src/kid3d/{anim,ik}.ts` (not the eye/lid code), new `src/kid3d/personality.ts`, new `src/game/emotes.ts` (stars, sweat drops, "!" over heads), the `Actor` class and `World.sync` animation logic in `src/game/world.ts` |
| **Phone controls & game feel** | `ccr-92524618-3jffwa-m-controls` | `src/game/{screen,director,fx}.ts` (including the comic pop-up system and cartoon particles), `src/ui/hud.css`, `src/sim/**` (feel, timing, difficulty; keep balance tests green), the aim overlays in `src/game/world.ts` |
| **Look, menus & phone app** | `ccr-92524618-3jffwa-m-look` | `src/ui/{app,dom,settings}.ts`, `src/ui/{base,menus}.css`, new `src/ui/look/**` (comic lettering, pop-up bursts, icons, cartoon components), `index.html`, `vite.config.ts`, `public/`, `src/main.ts`, PWA files |
| **Speed & graphics** | `ccr-92524618-3jffwa-m-gfx` | `src/gfx/**`, `src/world/**` (not `grownups.ts`), `src/dev/view3d.ts`, the `World` constructor, renderer, `adapt()` and quality code in `src/game/world.ts` |
| **Sound & voices** | `ccr-92524618-3jffwa-m-sound` | `src/audio/**`, `src/ui/commentary.ts`, audio and commentary tests |

Shared contact points, agreed up front:

- **Look → Controls.** The Look helper builds the visual kit (tokens in `base.css`,
  comic lettering, pop-up burst shapes and icons in `src/ui/look/`, components) **first** and reports it
  as an early milestone. The coordinator merges it into the base branch and tells the
  Controls helper, who then dresses the in-game HUD with it. Until then, Controls works on
  mechanics, layout and feel.
- **Faces → Speed.** Faces adds a lite detail level to `KidModel` (about a third of the
  triangles, same look from 60+ ft, e.g. `model.setDetail('lite' | 'full')`); Speed
  decides when to use it.
- **Settings rows.** Speed and Sound may each add their own rows to the settings
  screen (`app.ts`) and keys to `settings.ts`, kept small; Look restyles the screen later.
- **Sounds in the game screen.** Sound may add small, isolated event→sound hooks in
  `GameScreen.handleEvents`; Controls owns the rest of `screen.ts`.
- **Kid data.** Nobody edits `src/data/kids.ts` or `src/data/types.ts` except
  additively and only when unavoidable; put per-kid recipes in your own new files.
- **Coordinator only.** `CLAUDE.md`, `GAME-PLAN.md`, `README.md`, `plans/**`, `docs/`
  (the built site), the `gh-pages` branch, `scripts/publish-pages.sh`. Put notes for
  those files in your report instead.

## 6. Rules that still apply

- Every asset is generated in code: geometry, textures, faces, lettering, icons,
  sound, music. No model, image, audio or font files. No new runtime npm dependencies.
- `src/sim` stays free of DOM/Three and deterministic from the match seed.
- Storage stays optional (wrapped in try/catch); no `alert/confirm/prompt`.
- The test gate before every push: `npm run typecheck && npm test && npm run build`.

## 7. Verifying on a "phone" in the container

Headless Chromium uses software WebGL (~1 frame/s), so judge looks and layout, not
smoothness. Launch with `--enable-unsafe-swiftshader --ignore-gpu-blocklist`; emulate a
phone with a context like `{ viewport: { width: 844, height: 390 }, deviceScaleFactor: 2,
isMobile: true, hasTouch: true }` and also check 915×412, a portrait phone and a 1280×720
desktop. Use `q=low` / `ff=N` URL flags (see `CLAUDE.md`), `goto(url, { waitUntil:
'commit' })`, and wait for `window.__ready` or `window.__game.ready`. **Changing only
the `#hash` doesn't reload the page** — `goto('about:blank')` first. Serve a dev-mode
build with `vite preview` rather than `npm run dev` (HMR reloads break long runs).
Never `pkill -f`/`pgrep -f` with a pattern that matches your own command line.

## 8. Working with the coordinator

- Commit early and often; push to your own branch only. Don't deploy anything.
- **Milestones.** When you have something the owner should see early (Faces: the new
  faces; Speed: the auto-quality fix; Look: the visual kit), send the coordinator a short
  milestone message with the commit. The coordinator merges and refreshes the live link.
- **Before every report:** `git fetch origin ccr-92524618-3jffwa && git merge
  origin/ccr-92524618-3jffwa`, resolve conflicts keeping everyone's work, and re-run the gate.
- **Report** with `send_message` to the coordinator session `session_01EynejVPjuvR7vuS1Xegdw6`:
  branch and commit; what changed, in plain language; before/after evidence (what you
  looked at and what you saw); anything left undone; notes for `CLAUDE.md` / `GAME-PLAN.md`.
- **Then end your turn and wait.** The coordinator reviews and replies either with
  changes to make or with "APPROVED". On APPROVED, archive your own session
  (`get_session` with no id gives yours, then `archive_session`).
