# The look: a Saturday-morning cartoon

Fun, comical and clean. Bold flat colour, a warm ink outline on everything
(`--line`, 3 px), hard offset shadows, a little hand-drawn wobble, squash and
stretch on whatever moves. The screen stays uncluttered: the game plus only what
the player needs right now. The kids' world (pennants, trading cards, a
chalkboard, a clipboard, Channel 4½) is drawn as cartoon objects. All of it is
drawn in code: no fonts, images or emoji.

```ts
import { lettering, icon, comicPop, panel, label, tradingCard, lowerThird, teamPatch } from '../ui/look';
```

`main.ts` calls `installTextures()` and `ensureDefs()` once at startup; every
helper also works without them.

## Tokens (`src/ui/base.css`)

| Token | Use |
| --- | --- |
| `--grass`, `--grass-deep`, `--grass-light`, `--sky` | lawn, page background |
| `--sunshine` (also `--highlighter`, `--sun`) | the default button, highlights, bursts |
| `--tomato` (also `--marker-red`, `--red`) | "go" buttons, pop-up words, the circle around a choice |
| `--marker-blue`, `--marker-green` | accents (Channel 4½ logo, links) |
| `--marker` (also `--ink`) | the warm ink: outlines, text, hard shadows (never pure black) |
| `--poster`, `--poster-shade` | cream: cards, panels, plain buttons |
| `--cardboard`, `--cardboard-dark` | the logo sign and the card box |
| `--tape`, `--chalk`, `--board` | craft-era names kept as aliases (cream, off-white, deep green); don't use in new code |
| `--line` | outline width (3 px) |
| `--s1`…`--s6` | spacing 4/8/12/16/24/32 px |
| `--t-xs`…`--t-xl` | body type 13/14/16/19/24 px (never under 13 on a phone) |
| `--font`, `--font-typed` | calm rounded system stack; typewriter labels |
| `--shadow`, `--shadow-soft` | hard offset shadows (never blurred glows) |
| `--cut`, `--cut2` | rounded, slightly uneven corners |
| `--safe-t/r/b/l` | safe-area insets |
| `--scribble`, `--scribble-blue`, `--underline` | a marker loop around the chosen option; a wavy underline (from `textures.ts`) |

Team colours come in as `--team`, `--team2`, `--team3` (primary, secondary, accent).

## Hand lettering (`letters.ts`)

Display text only: titles, team names, scoreboard numbers, big popups. Body copy
stays in `--font`.

```ts
lettering('SEE YA!', { style: 'comic', size: 40, color: 'var(--sunshine)' })  // <svg>
letteringSVG(text, opts)   // the same as a string (cached), e.g. for innerHTML
letteringParts(text, opts) // { vb, body } to nest inside a bigger SVG
```

- `style`: `comic` (fat slanted letters, thick ink outline, offset shadow: titles, team
  names, score numbers, pop-ups) · `marker` (default, one even stroke: small labels) ·
  `poster` (comic, upright) · `chalk` (grainy) · `brush` (two loose passes).
- `highlight` (comic: on by default) adds a light streak on each stroke.
- `size` is the cap height in CSS px; the `<svg>` gets matching `width`/`height`.
  To fit a box instead, CSS `width: 100%; height: auto` (it has a viewBox).
- `color` or `colors` (cycled per word), `ink` (outline), `wobble` (0–2), `weight`,
  `spacing`, `slant` and `tilt` (deg), `align`, `seed`. `\n` makes lines.
- Capitals, digits, `! ? . , ' " - – — : / & # + ½ ( ) * %`, and accents (Calderón,
  Peña). Lowercase is drawn as capitals.
- The wobble is seeded by the text, so redrawing "3" on the scoreboard every frame
  gives the same "3". Pass `seed` to get a different hand for the same word.

## Icons (`icons.ts`)

`icon(name, { size?, title?, weight?, class? })` returns an inline `<svg>` sized
`1em` (set `font-size` to size it). Strokes are `currentColor`; parts marked
"accent" use `--ico-accent` (e.g. the ball's red stitches). Give a `title` when the
icon stands alone as a button label; otherwise it's decorative.

Names: `ball bolt forward back pause play up down close check phone rotate replay
whistle clipboard cards pennant chalkboard sound mute music mic home plate base
diamond fullscreen share add trophy star bat glove tap flip info gauge brush wave`.
See them all at `#look` in dev.

### Every emoji/symbol in the project and its icon

| Where | Today | Use |
| --- | --- | --- |
| `screen.ts` pause button | `❚❚` | `icon('pause', { title: 'Pause' })` |
| `screen.ts` rotate hint | `📱↻ Turn your phone sideways…` | `rotateHint()` (a paper card with the `rotate` icon tipping over); show it with the existing portrait media query |
| `screen.ts` scoreboard inning | `▲` / `▼` | `icon('up')` / `icon('down')` |
| `screen.ts` special button | `⚡ Moonshot` | `icon('bolt')` + label |
| `screen.ts` run controls | `RUN! ▶` / `◀ BACK!` | `icon('forward')` / `icon('back')` + label |
| `app.ts` Play ball, back button, special | `⚾`, `◀`, `⚡` | `ball`, `back`, `bolt` (done in the menus) |
| `world/props.ts` dugout sign (canvas texture) | `★` | not UI; left for the Speed & graphics owner (a drawn star would match) |

Also worth swapping in the HUD: `teamBadge()` (Georgia letter in a circle) →
`teamPatch(team, size)` (round patch, stitched ring, comic initial); score digits →
`lettering(String(runs), { style: 'comic', size: 18 })`.

## Comic pop-ups (`comic.ts`)

The signature for big moments: a burst with a thick ink outline, an offset shadow,
halftone dots and speed lines, the word in comic lettering. It scales in with an
overshoot, shakes, and is gone in about a second.

```ts
layer.appendChild(comicPop('SPLOOSH!', { shape: 'cloud', color: '#7cc8f0', color2: '#d8f0ff', textColor: '#fff6e0' }));
```

- `shape`: `star` (default) · `jagged` (a blast) · `cloud` (a puffy balloon).
- `burst` (same as `shape`), `colors: [outer, inner, text]` as shorthand, `animate: false` for a
  still element you animate yourself.
- `color` (outer), `color2` (inner burst, `'none'` for one layer), `textColor`, `ink`,
  `width` (px, default 280), `tilt`, `lines`, `dots`, `sub` (a small caption under it),
  `ms` (auto-remove; default 1150, 0 keeps it), `seed`.
- The shape is seeded by the word, so each word has its own burst.
- `comicBurstSVG(text, opts)` gives the bare SVG string (no animation).
- Which word and when (our own words, real moments only, never the same twice in a
  row) is the game screen's job.

Suggested pairings: SEE YA! (home run) star, sunshine/tomato · SPLOOSH! (pool) cloud,
sky blue · WHIFF! (strike out swinging) jagged, tomato/sunshine · SNAG! (diving catch)
star · BONK! (off the fence) jagged · THWACK! (crushed) star.

## Surfaces (`surfaces.ts` + `look.css`)

Each helper returns a plain element; the CSS class works on hand-written markup too.

| Helper | Class | What it is | Use it for |
| --- | --- | --- | --- |
| `sign(children, {seed, tilt})` | `.cardboard` | a flat cardboard sign, ink outline, slight tilt | the logo, big notices |
| `panel(children, {title, tone})` | `.cpanel` (`-sun`, `-sky`, `-grass`) | a flat panel, ink outline, hard shadow, optional comic title tab | How to Play, Settings, pause and end-of-game panels |
| `label(text, {tone})` | `.label` | a small flat tag with an outline | captions, notes, tags |
| `paper(children, {ruled})` | `.paper` (`.ruled`) | a cream card with an ink outline | notes, option cards |
| `button(label, onClick, {icon, kind, size, lettered})` | `.btn` (`.go`, `.ghost`, `.small`, `.big`, `.on`) | chunky flat button, ink outline, hard shadow, squashes when pressed; `.on` is circled in marker | every button |
| `.choices` / `.choice` | | plain words, the chosen one circled in red marker | segmented options |
| `pennant(team)` | `.pennant` | a pennant: team colour, sleeve, stitched edge, comic name | team pick |
| `teamPatch(team, size)` | `.patch` | round patch with a stitched ring and the initial | scoreboard, lists |
| `tradingCard({photo, name, persona, number, team, stats, back})` | `.tcard` | a trading card; `.flipped` (or `flipCard`) shows the back | kid cards: HUD at-bat card, Meet the Kids |
| `lowerThird({who, role, text, tone})` | `.lower-third` | Channel 4½ caption strip with the "4½" blob logo; one short line | Chet & Dottie (`tone: 'chet' \| 'dottie'`) |
| `channelBug()` | `.ch-bug` | the station logo alone | corner bug, replays |
| `bigMoment(text, {color, sub, ms})` | `.comic-pop` | old name for `comicPop` | |
| `rotateHint(text?)` | `.rotate-card` | paper card, phone icon tipping sideways | portrait warning |

## Rules of thumb

- Keep the screen clean: the game plus what's needed right now. Things come and go.
- Everything gets the ink outline and a hard offset shadow; colour is flat. No gradients
  on buttons, no glossy pills, no blurred glows, no realistic textures.
- Tilt a little (±1–2°), never everything the same way. Use the seeded helpers so
  nothing jitters on redraw. Things that move squash and stretch.
- Comic lettering for display words only; body text in sentence case in `--font`, 13 px+.
- Pop-ups are for real moments, with our own words.
- Write in-world copy: the kids, the yard, the Mendozas, Channel 4½. No "demo",
  "build", "loading assets".
- Touch targets at least 48 px (44 for small secondary ones).
