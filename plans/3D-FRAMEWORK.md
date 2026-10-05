# 3D Framework (v2 direction)

Goal: move from the v0.1 drawn-in-code 2.5D look to **real 3D with much better, more realistic graphics**: real lighting and shadows, 3D characters with proper animation, and cameras that move through the yard.

## The big picture in plain terms

- **Keep the brain, replace the body.** The simulation (rules, ball physics, fielding and running AI, seasons, stats, commentary, sound) is already true 3D underneath, with the ball tracked in feet in all three directions. That's roughly 70% of today's code, and it stays. What gets replaced is everything you *see*: the renderer, the character art, the yards.
- **Same platforms.** It's still one codebase for web, iPhone and Android, using a 3D engine that runs in browsers and inside the app-store wrappers.
- **The trade-off.** "Everything made in code" works for cartoons. Realistic characters need real 3D models made in modeling tools. Fields, fences, houses and props can still be built largely in code, with realistic materials. Characters need an art source: see *Decisions* at the bottom.
- **App size grows.** Today it's 85 KB. With 3D models and textures, expect roughly 15–40 MB, which is normal for a mobile game.

## Recommended technology

| Piece | Choice | Why |
|---|---|---|
| 3D engine | **Three.js** (WebGL 2, WebGPU later) | Keeps our TypeScript simulation as is. Runs on web and in the iPhone/Android wrappers. Huge community. Free. I can build and screenshot-test it in this environment. |
| Lighting | Physically based materials, a sun with soft shadows, sky lighting per time of day | This is most of what makes 3D look "real" |
| Characters | glTF 3D models with skeletons; animations from a motion library (e.g. Mixamo, free) | Industry-standard format; animations blend smoothly |
| Ground | Generated grass and dirt textures, mow stripes, real blades of grass near the camera (high quality setting) | Grass is what sells a backyard |
| Yards | Built in code from parts (fences, houses, trees, pools, sheds), using real materials (wood, siding, brick) | Keeps the 10 fields cheap to make and easy to tweak |
| App wrapper | Capacitor (unchanged) | Same plan as before |

**Alternative: Unity** (the most common engine for 3D mobile games). It has better artist tooling and top mobile performance. Cons: it means rewriting the whole game in a different language, I can't run its editor in this environment, and builds would need a computer with Unity installed. I'd only switch if we later want console-level visuals.

## How the pieces fit

```
 Simulation (unchanged, 120 ticks/sec)          3D presentation (new)
 ┌──────────────────────────────┐         ┌─────────────────────────────────┐
 │ Match / LivePlay / physics   │ state → │ Scene: yard, lights, sky        │
 │ fielders, runners, ball (ft) │         │ Characters: model + animation    │
 │ events (hit, catch, out...)  │ events →│ state machine per kid            │
 └──────────────────────────────┘         │ Camera director (bat/pitch/field │
         ▲                                │ /home-run/replay cams)           │
         │ input (swing, pitch, throw)    │ Effects: dust, grass, splash     │
 ┌──────────────────────────────┐         └─────────────────────────────────┘
 │ Controls (touch/mouse/keys)  │◄──────── HUD (existing DOM HUD, restyled)
 └──────────────────────────────┘
```

- The renderer smooths motion between simulation ticks, so movement looks fluid at any frame rate.
- **Replays:** the simulation is deterministic, so a play can be re-run from a saved snapshot and replayed from any camera. This makes home-run replays cheap.

## Characters

- **One shared kid skeleton** for all 90 kids, so every animation works on everyone.
- **Variety from parts:** body sizes (height, build), heads and faces, hair, skin tones, uniforms tinted per team, and costume pieces for each persona: ties, reading glasses, hard hats, capes, curlers, props. That's how 90 kids come from a manageable set of models.
- **Animation set (about 25 clips):** idle, ready crouch; batting stance, swing (normal/power/bunt), follow-through, whiff; pitch windup and release; throws (overhand, sidearm); catches (high, low, backhand); dive; run, slide; celebrations; frustration (bat slam, glove kick); persona extras (the Judge bangs his gavel, the Mime pantomimes).
- **Live touches on top of the clips:** heads track the ball, gloves reach for it, feet stay planted.
- **Faces:** expressions (happy, determined, crying-laughing after an error) via the face rig or swappable face textures.

## Cameras

- **Batting:** behind and slightly above the batter, like modern pro baseball games. The pitch comes at you.
- **Pitching:** behind the pitcher, looking at the batter.
- **Fielding:** a "broadcast" camera that frames both the ball and the fielder going after it.
- **Big moments:** home-run trot camera, diving-catch slow motion, replay angles.

## Fields in 3D

- Real elevation: The Big Hill's sloped outfield, the mound, the pond bank.
- Time of day per field: morning, noon, golden-hour sunset, dusk under string lights at Raccoon Hollow.
- Small life in each yard: the dog in the doghouse, sprinklers, swaying sunflowers, steam from the grill, grown-ups reacting to home runs.

## Performance targets

- 60 fps on mid-range phones from about 2020 on (iPhone 11, Pixel 6). Graceful fallback to 30 fps with lighter settings on older phones.
- Three quality settings (Low/Medium/High), picked automatically, changeable in Settings.
- Budgets: about 5–10k triangles per kid with a simpler version at distance, about 150 draw calls per frame, one shadow-casting sun.

## Build milestones (each ends with something playable)

| # | Milestone | What you'll see |
|---|---|---|
| 0 | **League switch** (can happen in today's version) | 10 teams × 9, seven traits, everyone can pitch, Pick-Up mode removed, two new teams and fields |
| 1 | **3D proof** | One yard in real 3D with lighting and shadows; simple stand-in players driven by the real simulation; working cameras; frame rate checked on phones |
| 2 | **First real kid** | One fully animated 3D kid batting, pitching, running and fielding |
| 3 | **All 90 kids** | The parts system plus persona costumes |
| 4 | **All 10 fields** | Each home field built, lit and dressed |
| 5 | **3D gameplay feel** | Controls tuned to the 3D cameras; replays; celebrations |
| 6 | **Polish and performance** | Quality settings; testing on real phones; then app-store prep (unchanged plan) |

Milestones 1 and 2 are the riskiest, so they go first. If the look doesn't land there, we adjust before building 90 kids.

## Decisions needed before Milestone 2

1. **Look:** "stylized realistic" (soft, real lighting and materials, Pixar-like proportions; recommended, and it suits the comedy) or photo-realistic (very expensive; real-looking kids can feel creepy).
2. **Where character art comes from** (the main cost decision):
   - **A. Buy a ready-made 3D kid character pack** and add free animations. About $50–$300 one-time; fastest; good quality; customized per kid in code. *Recommended.*
   - **B. AI-generated 3D models** (text-to-3D services). About $20–$60/month while building. Unique per kid, but quality is uneven and many need cleanup.
   - **C. Hire a 3D artist.** About $3,000–$15,000+. Best and fully original, but slow.
   - **D. Code-only 3D.** Free and fully original, but it looks like toys or claymation, not realistic.
3. **Engine:** Three.js (recommended) or Unity (see above).
