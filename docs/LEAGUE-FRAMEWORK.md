# League Framework (v2 direction)

The owner's rules for the league. These replace the v0.1 prototype's structure.

## The rules

- **10 teams, 9 kids each, 90 kids total.** No benches, no extras, no injuries, no trades. Every kid stays on their team forever.
- **No picking teams.** The Pick-Up Game draft mode goes away. The modes are **Exhibition** (any two teams, at the home team's field) and **Season**.
- **Every team has its own home field**, and every field has its own look and ground rules.
- **Everyone can pitch.** Any of the nine can take the mound. When a pitcher tires, they swap positions with a fielder. Nobody leaves the game.
- **Each kid has their own personality**: the mini-adult persona, bio and catchphrases carry over.

## Four traits per kid

Each trait is rated 1–10 and shown on the player card:

| Trait | What it does in the game |
|---|---|
| **Hitting** | How hard and how often they hit it (bat speed, exit velocity, sweet-spot size) |
| **Speed** | Running the bases and covering ground in the field |
| **Fielding** | Catching, range and throwing arm |
| **Pitching** | Pitch speed, movement, accuracy and stamina |

Rules that keep the league fair and interesting:

1. **No two kids share the same four numbers.** Every profile in the league is unique.
2. **Equal talent per team.** Each team's 36 trait points (9 kids × 4 traits) total within ±3 of every other team. Teams differ in *shape* (a slugging team, a speed team, a pitching team), not in overall strength.
3. **Everyone is good at something.** Each kid has at least one trait of 7 or higher, or is a true all-rounder (no trait below 5).
4. **Every team has at least three real pitchers** (Pitching 6+), so the rotation works.
5. Hidden flavor from personality (optional): a nudge like "free swinger" or "patient", taken from the persona rather than the stats.

**Mapping the current 72 kids:** Hitting = average of their old contact and power; Fielding = 60% old fielding + 40% old arm; Speed and Pitching carry over. Then nudge by ±1 until rules 1–4 hold. (A small script can do this and verify it automatically.)

**Engine impact:** small. The simulation already reads contact, power, speed, arm, fielding and pitching. Hitting feeds contact and power together, and Fielding feeds fielding and arm together. Rules, physics and AI stay the same.

## Pitcher rotation (all-pitch rule)

- Before each game, the manager (CPU, or you for your team) picks a starter. By default it's the most-rested kid with Pitching 6+.
- Pitch count builds fatigue. Past a limit, the pitcher swaps places with a fielder, usually the next-best pitcher.
- Season mode tracks rest days, so the ace can't pitch every game.

## The 10 teams and home fields

The existing eight stay as they are, each at its current yard. Two new teams join:

| # | Team | Division | Home field | Field hook |
|---|---|---|---|---|
| 1 | Maple Street Mudcats | Front Porch | Mudpuddle Meadow | Mud puddle stops balls dead |
| 2 | Cedar Lane Comets | Front Porch | Pool Party Paradise | Pool in right field: Splash Double |
| 3 | Willow Creek Frogs | Front Porch | Lily Pad Pond | Over the cattails is a pond home run |
| 4 | Elm Court Rockets | Front Porch | The Junk Lot | Old station wagon in right-center |
| 5 | **Riverbend Raccoons** *(new)* | Front Porch | **Raccoon Hollow** | Creek with a boat dock behind left; string lights for dusk games; raccoons "borrow" foul balls |
| 6 | Oak Hollow Owls | Back Fence | Treehouse Woods | Giant oak with a treehouse |
| 7 | Pine Ridge Pinecones | Back Fence | Grandma Bea's Garden | Tomato beds, gnomes, Grandma Bea |
| 8 | Birch Bay Bumblebees | Back Fence | Sandbox Stadium | Sandbox infield |
| 9 | Sunnyside Sparks | Back Fence | Sunflower Farm | Barn in center: Barn Burner |
| 10 | **Summit Street Yetis** *(new)* | Back Fence | **The Big Hill** | Steeply sloped outfield (balls roll back downhill); snow-cone stand. Only possible in true 3D |

**Playoffs:** two divisions of five. The top two in each division make the semifinals, and the winner takes the Lemonade Cup.
**Schedule:** each team plays the other nine once (9 games), or twice (18 games). Home and away are balanced.

## 18 new kids (sketch)

All mini adults, like the rest. The traits listed are each kid's standout, to be balanced under the rules above.

**Riverbend Raccoons**

| Kid | Persona | Standout |
|---|---|---|
| Audrey "The Auditor" Pratt | The IRS Auditor ("I'll need receipts for that double.") | Fielding |
| Rusty "Ranger" Okoye | The Park Ranger | Speed |
| Monty "Late Show" Carver | The Late-Night Talk Show Host | Hitting |
| Pip Lavalle | The Mime (silent; pantomimes every quip) | Pitching |
| Skip "Captain" Holloway | The Yacht Captain | Hitting |
| Bree "Believe It" Santos | The Motivational Speaker | All-rounder |
| Gil "Counselor" Nakashima | The Camp Counselor | Pitching |
| Harmony Brooks-Bell | The Barbershop Quartet Tenor | Speed |
| Ziggy Marlowe | The UFO Hunter | Pitching |

**Summit Street Yetis**

| Kid | Persona | Standout |
|---|---|---|
| Sven "Slopes" Albrecht | The Ski Instructor | Speed |
| Dolly "Ding-Ding" Ferreira | The Ice Cream Truck Driver | Hitting |
| Otto Polka | The Accordion Busker | Pitching |
| Frankie "Relish" Russo-Kim | The Hot Dog Vendor | Hitting |
| Stella "Steno" Abernathy-Ruiz | The Court Stenographer (types everything) | Fielding |
| Gus "Left at the Light" Tran | The Tour Bus Guide | All-rounder |
| Bonnie "Stop Sign" Lacroix | The Crossing Guard | Fielding |
| Mabel "Shhh" Okafor-Li | The Shushing Librarian | Pitching |
| Dex & Mr. Buttons | The Ventriloquist (the dummy does the trash talk) | Hitting |

Names and personas are drafts. Bios and catchphrases get written when the roster is built.
