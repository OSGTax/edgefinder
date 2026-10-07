# The look: "made by the kids of Maple Hollow"

Everything around the 3D game looks like the kids made it from what was in the
garage: cardboard, poster board, masking tape, permanent marker, sidewalk chalk,
felt pennants, trading cards and a public-access TV station (Channel 4½). All of
it is drawn in code. No fonts, images or emoji.

```ts
import { lettering, icon, sign, tape, tradingCard, lowerThird, bigMoment, teamPatch } from '../ui/look';
```

`main.ts` calls `installTextures()` and `ensureDefs()` once at startup; every
helper also works without them (flat colours instead of texture).

## Tokens (`src/ui/base.css`)

| Token | Use |
| --- | --- |
| `--grass`, `--grass-deep`, `--grass-light` | lawn, page background |
| `--cardboard`, `--cardboard-dark` | signs, the scoreboard, the shoebox |
| `--poster`, `--poster-shade` | poster board, index cards, plain buttons |
| `--marker` (also `--ink`) | text, outlines, hard shadows |
| `--marker-red`, `--marker-blue`, `--marker-green` | emphasis, "go", links, the scribbled circle |
| `--highlighter` (also `--sun`) | the default button, highlights |
| `--tape`, `--chalk`, `--board`, `--wood` | tape labels, chalk text, chalkboard |
| `--s1`…`--s6` | spacing 4/8/12/16/24/32 px |
| `--t-xs`…`--t-xl` | body type 13/14/16/19/24 px (never under 13 on a phone) |
| `--font`, `--font-typed` | calm rounded system stack; typewriter labels |
| `--shadow`, `--shadow-soft` | hard offset shadows (cut out and stuck down, never glows) |
| `--cut`, `--cut2` | uneven hand-cut corner radii |
| `--safe-t/r/b/l` | safe-area insets |
| `--tex-paper`, `--tex-cardboard`, `--tex-felt`, `--tex-chalkdust`, `--tex-lawn` | textures from `textures.ts`; use with `background-blend-mode: multiply` over a colour |
| `--scribble`, `--scribble-blue`, `--underline` | a marker loop around the chosen option; a wavy underline |

Team colours come in as `--team`, `--team2`, `--team3` (primary, secondary, accent).

## Hand lettering (`letters.ts`)

Display text only: titles, team names, scoreboard numbers, big popups. Body copy
stays in `--font`.

```ts
lettering('HOME RUN!', { style: 'poster', size: 40, color: 'var(--highlighter)' })  // <svg>
letteringSVG(text, opts)   // the same as a string (cached), e.g. for innerHTML
letteringParts(text, opts) // { vb, body } to nest inside a bigger SVG
```

- `style`: `marker` (default, one even stroke) · `poster` (fat paint, outline, drop
  shadow: titles, popups) · `chalk` (grainy, on the chalkboard) · `brush` (two loose
  passes: banners).
- `size` is the cap height in CSS px; the `<svg>` gets matching `width`/`height`.
  To fit a box instead, CSS `width: 100%; height: auto` (it has a viewBox).
- `color` or `colors` (cycled per word), `ink` (outline), `wobble` (0–2), `weight`,
  `spacing`, `tilt` (deg), `align`, `seed`. `\n` makes lines.
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
`teamPatch(team, size)` (felt patch, stitched ring, lettered initial); score digits →
`lettering(String(runs), { size: 18 })`.

## Surfaces (`surfaces.ts` + `look.css`)

Each helper returns a plain element; the CSS class works on hand-written markup too.

| Helper | Class | What it is | Use it for |
| --- | --- | --- | --- |
| `sign(children, {seed, tilt})` | `.cardboard` | corrugated cardboard, torn edges, slight tilt | the scoreboard, the logo, big notices |
| `paper(children, {ruled})` | `.paper` (`.ruled`) | poster board / an index card | cards, notes, menus |
| `tape(text, {tone})` | `.tape` | masking tape with marker on it | labels, small tags, tabs |
| `tapeCorners(el)` | `.tape-bit` | two bits of tape holding a thing up | anything "stuck to the wall" |
| `button(label, onClick, {icon, kind, size, lettered})` | `.btn` (`.go`, `.ghost`, `.small`, `.big`, `.on`) | cut-out poster board, marker border, hard shadow; `.on` is circled in marker | every button |
| `.choices` / `.choice` | | plain words, the chosen one circled in red marker | segmented options |
| `pennant(team)` | `.pennant` | felt pennant: team colour, sleeve, lettered name | team pick |
| `teamPatch(team, size)` | `.patch` | round felt patch with the initial | scoreboard, lists |
| `tradingCard({photo, name, persona, number, team, stats, back})` | `.tcard` | a trading card; `.flipped` (or `flipCard`) shows the back | kid cards: HUD at-bat card, Meet the Kids |
| `chalkboard(children)` | `.chalkboard` | slate in a wooden frame, chalk tray | How to Play, the coach |
| `clipboard(children, {title})` | `.clipboard` | board, metal clip, ruled sheet | Settings, pause menu, box score |
| `lowerThird({who, role, text, tone})` | `.lower-third` | Channel 4½ construction-paper caption with the hand-cut "4½" logo | Chet & Dottie (`tone: 'chet' \| 'dottie'`) |
| `channelBug()` | `.ch-bug` | the station logo alone | corner bug, replays |
| `bigMoment(text, {color, sub, ms})` | `.big-moment` | a bedsheet banner with brush lettering that drops in, bounces and settles | HOME RUN!, SPLASH DOUBLE!, STRUCK HIM OUT! |
| `rotateHint(text?)` | `.rotate-card` | paper card, phone icon tipping sideways | portrait warning |
| `tear(el, seed)` / `tornClip(...)` | | a torn edge as a `clip-path` | any surface |

## Rules of thumb

- One or two surfaces per screen, each for a reason. Don't box everything.
- Tilt a little (±1–2°), never everything the same way. Use the seeded helpers so
  nothing jitters on redraw.
- Shadows are hard and offset (stuck-down paper), never blurred glows; no gradients
  on buttons, no glossy pills.
- Lettering for display words only; body text in sentence case in `--font`, 13 px+.
- Write in-world copy: the kids, the yard, the Mendozas, Channel 4½. No "demo",
  "build", "loading assets".
- Touch targets at least 48 px (44 for small secondary ones).
