# /desk revamp — implementation spec

Owner-requested, 2026-08-12. Baseline measured against a local mirror of the
live desk with production data, Chromium 1400×1100, default landing state
(no cards expanded by hand), version v10.2.2.

**The measurement that drives everything:** the page is **40,971px** tall on
landing — 37 screens. The claims registry is 44.9% of that and predictions vs
outcomes another 28.4%; two archival tables are **73% of the page**. The latest
decision — the only thing on the desk no other site has — is **1.6%** and
begins 5.1 screens down. The thinking feed is 0.4%.

| Card | Height | Share |
|---|---:|---:|
| Claims registry (179 entries) | 18,389px | 44.9% |
| Predictions vs outcomes | 11,622px | 28.4% |
| The AI's notebook | 4,288px | 10.5% |
| What the AI owns (50 rows + 48-item donut legend) | 2,888px | 7.1% |
| Open orders (44 stop sentences) | 1,249px | 3.0% |
| The latest decision | 635px | 1.6% |
| Account value chart | 617px | 1.5% |
| Hero + honesty strip | 592px | 1.4% |
| Strategy Lab | 191px | 0.5% |
| Thinking feed | 145px | 0.4% |
| Trade receipts (collapsed) | 41px | 0.1% |

Other measured facts: the section nav first appears at y=3941px (3.6 screens
down); the allocation block alone is 1,023px; `--t-accent` and `--t-up` are the
same hex (`#26d49c`).

**Aesthetic direction:** keep the dark trading-terminal identity and fix its
execution. The owner was offered the alternative (something friendlier) and
did not take it, so this spec does not restyle the product's character.

---

## Phase 1 — make the page navigable

### 1.1 Zone nav becomes real tabs

`desk.html`, `desk.js`, `desk.css`.

The markup already has two `<section class="desk-zone">` blocks and a sticky
`.desk-anchornav`. Today the nav scroll-jumps within one 37-screen page. Change
it to a tab bar that renders one panel at a time.

Three tabs, replacing the current Overview / Reasoning / Learning:

| Tab | Cards |
|---|---|
| **Now** | latest decision · thinking feed · account value chart · holdings · open orders |
| **Track record** | predictions vs outcomes · trade receipts |
| **What it's learned** | Strategy Lab · notebook · claims registry |

- Hero and honesty strip stay above the tab bar, always visible on every tab.
- The tab bar moves to directly under the honesty strip (from y≈3941 to y≈600).
- **The decision leads the Now tab** — this is the "promote above the fold"
  move. Equity chart and holdings follow it.
- Inactive panels get `hidden`; they are still populated (the loaders already
  run on an interval), so switching tabs is instant and no new endpoints are
  needed.
- Active tab persists in `localStorage` under `ef-desk-tab-v1`, and `#zone-*`
  hash links still select the right tab so existing deep links keep working.
- Keyboard: the bar is `role="tablist"`, buttons are `role="tab"` with
  `aria-selected`, panels are `role="tabpanel"`.

Delete `wireAnchorNav`'s IntersectionObserver scrollspy — with one panel visible
there is nothing to spy on.

### 1.2 Cap every long list, with a counted expander

Never silently truncate: the count in the toggle is the disclosure.

| Surface | Cap | Toggle |
|---|---|---|
| Holdings (equity rows) | 10 | "Show all 50 holdings" |
| Claims registry | 5 | "Show all 179 claims" |
| Predictions vs outcomes | 8 open + 8 closed | "Show all N" |
| Notebook lessons | 4 | "Show all N pages" |
| Open orders | 12 | "Show all 44 orders" |

In-place expand, not new routes — the data is already loaded and adding pages
would mean new endpoints for no gain. Extract one shared helper
(`capList(container, rows, cap, render, noun)`) rather than five copies.

The options sub-table inside holdings is not capped: it runs 6 rows and is the
part of the book most worth seeing in full.

### 1.3 Open orders → summary line + table

`openOrderRow` currently renders 44 near-identical English sentences
("ABNB — protective stop at $168.00 on 3.00 shares"). Replace with:

- A summary line: *"44 orders resting at the broker — 44 protective stops
  covering $X of the book."*
- A compact table: **Symbol · Type · Trigger · Quantity · Placed**, capped at 12.
- Keep the GTC-expiry warning pill; it is real risk information.

---

## Phase 2 — fix the words

### 2.1 Label rewrites

| Where | Now | Becomes |
|---|---|---|
| Hero stat | `vs S&P 500 (price return)` | `vs S&P 500` (caveat moves to the honesty strip) |
| Hero stat | `Investments` / `50` | `Positions held` / `50` |
| Claims footer | `3 evidence ref(s)` | `3 pieces of evidence` / `1 piece of evidence` |
| Orders | `resting 1d` | `placed yesterday` / `placed today` / `placed 5 days ago` |
| Allocation legend | `SNOW260918C00340000` | `SNOW $340C 2026-09-18` |
| Equity axis | `103000.00` | `$103k` |

**Deviation from the artifact spec.** That spec proposed rendering the
benchmark stat as a sentence ("Behind the S&P 500 by 9.6%"). It is one of five
cells in a `<dl>` grid; making one a sentence breaks the row rhythm and the
tabular alignment that lets the five be compared. The real defect is that the
label is long enough to wrap onto two lines, which drops the value out of
alignment with its neighbours. Shortening the label fixes exactly that, and the
"price return, both sides" caveat belongs in the honesty strip where the other
standing disclosures already live.

The equity axis uses `compactPrice`, which already exists in `core/charts.js`
but is gated to `isNarrow()`. Two decimals on a six-figure axis are noise at
any width — apply it unconditionally.

### 2.2 The prediction block

`pickCard` renders the pre-registered prediction as three lowercase
key-colon-value chips. It is the most valuable thing on the page — a
falsifiable commitment made before the trade — and it looks like debug output.

Render as a labelled definition list instead:

| Chip today | Label |
|---|---|
| `predicts: …` | **The bet** |
| `horizon: 12 sessions` | **Checked by** — "12 trading sessions from now" |
| `abandon if: …` | **Called off if** |

Structured data already exists on `desk_decisions`; this is layout only. Keep
the "graded later in Predictions vs outcomes" explanation as a title on the
block rather than repeated on each of three chips.

### 2.3 Writing rules for the agent

`.claude/skills/trading-agent/SKILL.md`. Not a UI fix — this is what the agent
*writes*, and no stylesheet can reach it.

- Never `--` for an em dash. Use a real `—`, or rewrite the sentence.
- Never put command syntax in desk-visible prose. Claim C-2 currently reads
  "Every strategy pivot (state-set --bump) and every fill…" — a CLI invocation
  in a sentence meant for a person.
- Claims: two sentences maximum, and no semicolon standing in for a full stop.
  The registry is a public page now, not a note to its future self.
- The audience is the owner and anyone he shows the desk to — not the agent's
  next cycle.

---

## Phase 3 — visual system

### 3.1 Type scale for prose

`tokens.css` has `--t-fs-base: 13px` and nothing between it and `--t-fs-xl:
20px`. Hierarchy is carried by colour and weight alone, which is why the page
reads as one continuous texture.

**Deviation from the artifact spec.** That spec said raise body to 14px.
Raising `--t-fs-base` moves every data cell in every table — a large blast
radius for a change whose benefit is in *prose*. Add prose-specific steps
instead and leave the data ramp alone:

```
--t-fs-body:    15px   /* card subs, rationales, claim statements, summaries */
--t-fs-lede:    17px   /* the sub on a primary card */
--t-fs-display: 30px   /* tab-panel titles */
```

### 3.2 Three card tiers

Fourteen cards with identical border, radius and padding means nothing looks
more important than anything else.

- **Hero** — `--t-surface-2`, larger pad. Already visually distinct; keep.
- **Primary** (`.desk-card`) — decision, thinking, equity, holdings, orders:
  surface + border, standard pad, heading at `--t-fs-lg`.
- **Reference** (`.desk-card--ref`) — claims, notebook, lab, receipts,
  predictions: no border, no surface. A top rule and generous space above.

### 3.3 Split interactive accent from semantic green

`--t-accent` is `#26d49c`, identical to `--t-up`. Links, active tabs, the
thinking-feed phase label and a winning position all render the same, so green
has stopped meaning "up".

**Deviation from the artifact spec.** That spec said point `--t-accent` at the
existing blue. `--t-accent` is also the colour of the account equity curve
(`charts.js` `colors().accent`) and the brand mark — repainting the product's
signature curve blue is a bigger aesthetic change than the problem calls for,
and it was not what the owner asked for. Introduce `--t-link` instead
(`#58a6ff` dark / `#2f6fd0` light) and move only *interactive chrome* onto it:
links, `.c-link`, active tabs, the What's New button, the feed phase label.
Green stays the brand and the account curve; green-as-a-link is what goes away.

### 3.4 Donut → treemap

A 48-slice donut is unreadable by construction, and its single-column legend
with a large empty area to its right measures 1,023px — more than the decision
and thinking feed combined.

`components/treemap.js` already exists (squarified, positioned divs,
`treemap(el, nodes, {height, onClick})`). Render the top 12 holdings by weight
plus one pooled "38 smaller positions" tile and one cash tile. Click a tile →
the symbol page. Target height ~260px.

---

## Constraints

- Test gate before every commit:
  `DATABASE_URL= python -m pytest tests/ -q -m "not integration"`.
  Three failures pre-exist on `main` (`test_grade_alpaca`, `test_settings`,
  `test_streamer`) and are not mine to fix here.
- Version-bump `dashboard/app.py` per merge; commit format `[vX.Y.Z] …`.
- No agent tool, no `desk_*` write path, and no sacred table is touched. This
  is `dashboard/` plus one skill file.
- Verify against the local replica harness, never production.

## Order of work

Phase 1 → verify → Phase 2 → verify → Phase 3 → verify → full review → merge.
Phase 1 carries the return; if anything has to be dropped, drop from Phase 3.
