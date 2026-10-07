# Mobile revamp — shared brief for every helper session

The owner's words (October 2026):

- "Completely revamp and upscale this game quality and playability on mobile phones."
- "Gameplay needs to present as not AI generated."
- "We need to spend some more time on faces as they look kind of scary right now."
- On a computer it was "deathly slow"; on the owner's phone it ran fine.

The phone in landscape is now the primary target: touch only, two thumbs, a screen
roughly 844×390 CSS px (iPhone 13/14) to 915×412 (common Android). Tablets and computers
must keep working with touch, mouse and keyboard, but every decision is judged on a phone first.

Read `CLAUDE.md` (developer map) and `GAME-PLAN.md` (owner's plan) before starting.

---

## 1. Art direction: "made by the kids of Maple Hollow"

The joke of the game is kids playing at being grown-ups. The presentation leans into
it: **everything around the game looks like the kids made it themselves** out of
what's in the garage — corrugated cardboard, poster board, masking tape, permanent
marker, sidewalk chalk, felt pennants, stickers, bottle caps, index cards, a
public-access TV station run out of somebody's basement. The 3D world is a warm,
real-feeling late-summer afternoon. Together that gives the game a specific,
authored identity instead of a generic app look.

Concretely:

- **Scoreboard** — a hand-painted cardboard sign with masking-tape labels and marker
  numbers; the count as chalk tallies or bottle caps; bases as a little drawn diamond.
- **Announcer captions** — Chet & Dottie's "broadcast" on *Channel 4½, Maple Hollow
  Public Access*: a homemade lower-third (construction paper, a hand-cut logo,
  slightly crooked), with their names, not bare text pills.
- **Kid cards** — trading cards: photo (3D portrait), name in marker, persona on a
  typed label, two or three stats that matter right now (a batter's Contact/Power, a
  pitcher's pitches and Control). The full stat sheet lives on the card's back, not on
  the HUD.
- **Menus** — the kids' clubhouse: team pennants, a shoebox of trading cards for Meet
  the Kids, a clipboard for settings, a chalkboard for How to Play.
- **Big moments** ("HOME RUN!", "SPLASH DOUBLE!") — hand-lettered, as if painted on a
  bedsheet banner or stamped, with movement that has weight.

**Type.** Display text (titles, team names, scoreboard numbers, big popups) uses
hand-made lettering *generated in code* (e.g. single-stroke glyph skeletons rendered
with a marker/brush stroke and slight wobble). No web fonts or font files. Body
text uses a calm rounded system stack, sentence case, at least 13 CSS px on phones.

**Icons.** Drawn in code (SVG paths) in one consistent hand style. **No emoji
anywhere** in the UI (today: ⚾ ⚡ 📱 ↻ ▶ ◀ are used — replace them all).

**Palette.** Warm, slightly sun-faded summer: grass greens, cardboard tan, poster-board
off-white, marker black, marker red and blue, highlighter yellow, masking-tape cream,
plus each team's colors as accents. No neon gradients, no glossy pills.

### The "not AI-generated" checklist

| Reads as generated / templated | Reads as made by people |
| --- | --- |
| Emoji as icons | Code-drawn icons in one hand style |
| Gradient pill buttons, glossy circles | Surfaces from the kid-made world: cardboard, tape, felt, stickers; a little imperfect (1–2° tilt, torn edge, tape corner) |
| One heavy system font for everything | Hand-lettered display type plus a calm body face, with clear hierarchy |
| The same rounded panel everywhere | A few deliberate surfaces, each with a reason to exist |
| Seven stat bars on every card | Two or three numbers that matter right now; the rest on demand |
| Placeholder/dev copy ("Watching the kids play", "Demo build…") | In-world copy written for this game |
| Repetition: same face, same line twice, same celebration | Per-kid variation; no-repeat logic; specific details |
| Uniform timing, symmetric poses, everything at once | Hand-tuned timing, anticipation and follow-through, small asymmetries |
| Everything explained by text | Shown in the world (the scoreboard flips, kids react); text short |
| Same-looking faces with a color swap | Each kid designed as a character |

The originality rule still applies: inspired by 90s backyard baseball games, never
their names, characters, art, music or trademarks, and no real MLB teams or players.

---

## 2. Faces: friendly, never scary (top priority)

What's wrong today (see `#gallery&kid=ruby&q=high`): realistic eyeballs with small
irises and lots of white, heavy half-lowered upper lids and a dark lid line, so every
kid has the same flat suspicious stare; a long egg-shaped head with a big empty lower
face; a thin little mouth sitting low; identical features on every kid.

The target is appealing stylized cartoon kids that read at every size:

1. **Eyes** — big dark irises and pupils filling most of the eye opening, clear
   catchlights, upper lids high at rest (lids come down only for blinks and
   expressions), little or no white showing at rest, set a bit lower and closer
   together. Painted/decal eyes are allowed if they read friendlier than 3D eyeballs;
   choose whatever looks best. The look-at and blink behavior must keep working.
2. **Head** — rounder, fuller cheeks, softer smaller chin; features centered lower,
   bigger forehead.
3. **Mouth** — larger and readable: a soft smile with defined corners at rest, a hint
   of lower lip, closer to the nose; a real expression range (open laugh, determined
   grit, surprised O, pout, yell).
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
- **Feel.** Weighty contact (sound, a frame of hit-stop, camera kick, slow-mo on a crush);
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
| **Animation & personality** | `ccr-92524618-3jffwa-m-anim` | `src/kid3d/{anim,ik}.ts` (not the eye/lid code), new `src/kid3d/personality.ts`, the `Actor` class and `World.sync` animation logic in `src/game/world.ts` |
| **Phone controls & game feel** | `ccr-92524618-3jffwa-m-controls` | `src/game/{screen,director,fx}.ts`, `src/ui/hud.css`, `src/sim/**` (feel, timing, difficulty; keep balance tests green), the aim overlays in `src/game/world.ts` |
| **Look, menus & phone app** | `ccr-92524618-3jffwa-m-look` | `src/ui/{app,dom,settings}.ts`, `src/ui/{base,menus}.css`, new `src/ui/look/**` (lettering, icons, paper/tape/cardboard drawing), `index.html`, `vite.config.ts`, `public/`, `src/main.ts`, PWA files |
| **Speed & graphics** | `ccr-92524618-3jffwa-m-gfx` | `src/gfx/**`, `src/world/**` (not `grownups.ts`), `src/dev/view3d.ts`, the `World` constructor, renderer, `adapt()` and quality code in `src/game/world.ts` |
| **Sound & voices** | `ccr-92524618-3jffwa-m-sound` | `src/audio/**`, `src/ui/commentary.ts`, audio and commentary tests |

Shared contact points, agreed up front:

- **Look → Controls.** The Look helper builds the visual kit (tokens in `base.css`,
  lettering and icons in `src/ui/look/`, surface components) **first** and reports it
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
