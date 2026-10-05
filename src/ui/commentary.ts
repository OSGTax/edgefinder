import { Rng } from '../engine/rng';
import { kid } from '../data/kids';
import { POSITIONS, SPECIAL_INFO, type Kid, type Position } from '../data/types';
import type { MatchEvent } from '../sim/types';
import type { Match } from '../sim/match';

// Chet Valentine (play-by-play, age 10, clip-on tie, treats it like network
// TV) and Dottie Fairweather (color, age 12, "former big leaguer" from one
// season of tee-ball, brings it up constantly).
//
// Every line comes from a named pool; the booth remembers which lines of each
// pool it used recently (and the last few dozen lines overall) so a game
// doesn't repeat itself. All randomness goes through the booth's own seeded
// Rng so a replayed game says the same things.

export interface Line { who: 'Chet' | 'Dottie'; text: string }

const C = (text: string): Line => ({ who: 'Chet', text });
const D = (text: string): Line => ({ who: 'Dottie', text });

const nick = (id: string) => kid(id).nick;
const BASE = ['home', 'first', 'second', 'third'];
const POS_NAME: Record<Position, string> = {
  P: 'pitcher', C: 'catcher', '1B': 'first base', '2B': 'second base', '3B': 'third base',
  SS: 'shortstop', LF: 'left field', CF: 'center field', RF: 'right field',
};
/** Where a fielder stands, for "Dez at first base", "Inny on the mound". */
const POS_AT: Record<Position, string> = {
  P: 'on the mound', C: 'behind the plate', '1B': 'at first', '2B': 'at second', '3B': 'at third',
  SS: 'at short', LF: 'in left', CF: 'in center', RF: 'in right',
};
/** "Bo's", but "Pickles'" and "Gus-Gus'". */
const poss = (name: string) => (name.endsWith('s') ? `${name}'` : `${name}'s`);
const lower = (s: string) => s.charAt(0).toLowerCase() + s.slice(1);
const persona = (k: Kid) => lower(k.persona);
/** Use the long version if it fits the ticker, else the short one. */
const fit = (long: string, short: string, max = 110) => (long.length <= max ? long : short);

// ───────────────────────────────────────────── per-kid flavor
// Keyed by kid id; missing kids (future teams) simply get generic lines.
interface Flavor { up: string[]; hit?: string[]; k?: string[]; pitch?: string[]; field?: string[] }
const FLAVOR: Record<string, Flavor> = {
  bo: {
    up: ['Mudpie\'s knees are acting up again. He told me personally. Twice.', 'Bo says this bat "doesn\'t make \'em like they used to." It\'s from last Tuesday.', 'Mudpie brought his wrench to the on-deck circle. In case something leaks.'],
    hit: ['Mudpie fixed THAT leak!', 'Bo\'s gonna need an ice pack for those knees after all that running.'],
    k: ['Mudpie\'s blaming the pipes. There are no pipes.', 'Bo says strike three "was a clog." Sure, Bo.'],
    pitch: ['Mudpie\'s pitching. He says he\'ll "take a look at it" and charge us extra.'],
  },
  ines: {
    up: ['Inny\'s wearing sunglasses at the plate. It\'s for the drama.', 'Last week Inny fainted in the on-deck circle. The paramedic was a juice box.', 'Inny says this at-bat is "a two-part episode."'],
    hit: ['Inny\'s taking a bow at first base. Several bows.', 'Inny is blowing kisses to her public. Her public is four moms.'],
    k: ['Inny is lying in the dirt. She\'s fine. She\'s ACTING.', 'Inny says she was ROBBED. Dramatic music, please.'],
    pitch: ['Inny on the mound. Somebody cue the dramatic music.', 'Every Inny pitch is a season finale. Stay tuned.'],
  },
  toby: {
    up: ['Toby just blew his whistle at the pitcher. For what? Nobody knows.', 'Coach Toby wants everybody in the dugout to drop and give him twenty.', 'Toby\'s got a clipboard up there. He\'s grading his own swing.'],
    hit: ['Toby\'s yelling "HUSTLE" at himself while he runs. Inspiring.', 'Toby gives himself an A-plus for that one.'],
    k: ['Toby just gave himself detention.', 'That\'s a teachable moment for Toby. He\'s teaching it to himself.'],
    field: ['Toby blows the whistle on that one. You\'re OUT, mister!'],
  },
  wren: {
    up: ['Birdie spotted a cardinal on the way up to bat. That\'s 213!', 'Wren\'s got binoculars in her back pocket. Very professional.', 'Birdie\'s whispering so she doesn\'t scare the pitcher.'],
    hit: ['Birdie\'s on base! She\'s writing it in her bird journal.'],
    field: ['Birdie FLEW for that one! Get it? Birdie? I\'m here all week.', 'Spring Sneakers! Wren got more air than a blue jay.'],
  },
  dez: {
    up: ['Dez is up. Everybody lower your voices. Smoooooth.', 'You\'re listening to Dez FM, and Dez FM is at the plate.', 'Dez says this at-bat goes out to the ladies. The ladies are his grandma.'],
    hit: ['That was smooth like a saxophone solo, baby.', 'Dez FM, playing all the hits. Including that one.'],
    k: ['Dez keeps it mellow. Even after striking out. Respect.'],
    pitch: ['Dez on the mound. Keepin\' it mellow, baby.'],
  },
  priya: {
    up: ['Pree\'s ledger is in the dugout. She already pre-recorded this at-bat.', 'Priya calculated her odds before walking up. She won\'t tell me. Confidential.', 'Pree wears reading glasses at the plate. She reads the pitches.'],
    hit: ['Priya books that hit as an asset. Double-entry!', 'The numbers don\'t lie. That was a hit.'],
    k: ['Pree\'s writing that strikeout off. Non-deductible, though.'],
    pitch: ['Priya on the mound. Every pitch is itemized.'],
  },
  gus: {
    up: ['Gus-Gus is wearing flannel. In July. A true professional.', 'That beard is a cut-up sponge, folks. Do NOT tell Gus.', 'Gus-Gus says he ate fastballs for breakfast. Also waffles.'],
    hit: ['TIMBERRR! Gus-Gus chopped that one!', 'Gus split that one like firewood.'],
    field: ['Gus-Gus threw that like he was chucking a log!'],
    pitch: ['Gus-Gus on the hill. He\'s gonna chop some wood out there.'],
  },
  molly: {
    up: ['Pickles just took the umpire\'s lunch order. There\'s no umpire.', 'Molly\'s got a pencil behind her ear. Pickle on the side, presumably.', 'Number forty-two! Now serving: Pickles, at the plate.'],
    hit: ['Pickles delivers! Unlike her sandwiches.', 'Order up! One base hit, toasted.'],
    field: ['Order up! Pickles serves that one right into her glove.', 'Flypaper Glove! Nothing gets past the deli counter.'],
    pitch: ['Pickles is pitching. You want that knuckleball toasted?'],
  },
  jun: {
    up: ['Jun\'s selling his bat to the catcher. Three easy payments!', 'Jumpin\' Jun says if you call now, you get a SECOND at-bat. You don\'t.', 'Jun tried to sell me a blender before the game. I almost bought it.'],
    hit: ['But wait, there\'s MORE! Jun\'s on base!'],
    k: ['But wait... there\'s NO more. Jun goes down.', 'Operators are standing by to take Jun\'s complaints.'],
    pitch: ['Jun\'s on the mound. But wait, there\'s more!', 'Jun says this next pitch is "not available in stores."'],
  },
  kai: {
    up: ['It\'s Kaboom, at his own house! Home-yard advantage is real.', 'Mrs. Mendoza just yelled "aim left, sweetie." Kai did not hear. Nobody hears over Kai.', 'Kaboom announced his own at-bat. SUNDAY SUNDAY SUNDAY.'],
    hit: ['Mr. Mendoza just waved his spatula! That\'s his boy!', 'KA-BOOM! Kai\'s yelling it himself, so I don\'t have to.'],
    pitch: ['Kaboom on the mound. Cover your ears, folks.', 'Kai\'s pitching at his own house. He knows every bump in this lawn.'],
  },
  ruby: {
    up: ['Rooster just gave the traffic report. Light on Cedar Lane.', 'Ruby says it\'s a BEAUTIFUL day for an at-bat. She\'s been up since 5.', 'Good morning from Ruby! It\'s four in the afternoon.'],
    hit: ['Rooster\'s on base! Back to you, Chet. Thanks, Ruby!', 'Ruby\'s doing the weather from first base. Sunny!'],
    field: ['Rooster! Up early and catching everything.'],
  },
  ezra: {
    up: ['Easy E sold the catcher foul-tip insurance on the way up. Still not real.', 'Ezra\'s briefcase is in the dugout. It\'s empty. Business.', 'Easy E reminds everyone that line drives are NOT covered.'],
    hit: ['Ezra\'s safe. And insured.'],
    k: ['Ezra\'s filing a claim on that strikeout. Denied.', 'Easy E says strike three was "an act of nature."'],
    field: ['Easy E has that play fully covered. Low deductible.'],
  },
  maya: {
    up: ['Zoom Zoom! Fastest kid on Cedar Lane. She delivered the mail on the way to bat.', 'Maya\'s got a fanny pack full of stamps. Special delivery coming.', 'Neither rain nor sleet stops Zoom Zoom. A curveball might.'],
    hit: ['Special delivery! Zoom Zoom is on base, no signature required.', 'Zoom Zoom left a slip at first base: "Sorry I missed you!"'],
    field: ['Zoom Zoom ran that down like a late package.'],
  },
  leo: {
    up: ['Lefty Leo is not French, folks. He\'s from Cedar Lane.', 'Leo calls every pitch "un soufflé." Every single one.', 'Lefty Leo brought a rolling pin to bat. He\'s been asked to use the bat.'],
    hit: ['Magnifique! Leo cooked that one!', 'Leo says that hit needed more butter. It was perfect.'],
    pitch: ['Leo on the mound, serving up soufflés.', 'Lefty Leo says zis pitch is "fully baked."'],
  },
  anya: {
    up: ['Snowball just forecast a home run. Her forecasts are right ten percent of the time.', 'Anya says there\'s a cold front moving through the batter\'s box.', 'Snowball\'s five-day forecast: sunny with a chance of dingers.'],
    hit: ['Anya called that! Wait, she called snow. Close enough.'],
    field: ['Rocket Arm from Snowball! That throw had a wind chill.'],
    pitch: ['Anya\'s pitching. Ninety percent chance of strikes. So, ten percent.'],
  },
  darius: {
    up: ['D-King has objected to the pitcher. On what grounds? "Vibes."', 'Darius handed the catcher his business card. Attorney at Lawn.', 'D-King has filed a motion to make the strike zone smaller.'],
    hit: ['D-King rests his case. On first base.'],
    k: ['Darius would like to appeal that strikeout. To his mom.', 'OBJECTION! Overruled. Darius goes back to the dugout.'],
    pitch: ['D-King on the mound. Every pitch is "allegedly" a strike.'],
  },
  pepper: {
    up: ['Pepper\'s up! Do I hear a single? Do I hear a double?', 'Pepper accidentally sold Hank\'s bike last week. Hank is still sad.', 'Pepper talks so fast she finished this at-bat already. In her head.'],
    hit: ['SOLD! To the kid on first base!'],
    pitch: ['Pepper\'s on the mound. Going once, going twice...', 'Pepper\'s talking so fast the batter forgot what sport this is.'],
  },
  hank: {
    up: ['Here\'s Tank. Biggest kid in the league. Gentlest kid in the league.', 'Hank asked the base runners to please walk. They did not.', 'Tank\'s got his earpiece in. Code red at the plate.'],
    hit: ['Tank\'s jogging to first. No running in the mall.', 'Hank hit that like it shoplifted.'],
    k: ['Tank strikes out, and he apologizes to the catcher. Such a nice boy.'],
  },
};

// ───────────────────────────────────────────── running gags
const GRILL = [
  'Mr. Mendoza has been at that grill since 9 a.m. It\'s not a hobby, it\'s a lifestyle.',
  'Update from the grill: Mr. Mendoza flipped something. Possibly a burger.',
  'Mr. Mendoza\'s Hawaiian shirt has a brand-new ketchup stain. Breaking news.',
  'The grill smoke just drifted into center field. Smells like victory. And hot dogs.',
  'Mr. Mendoza says the burgers are "almost ready." He said that at noon.',
  'Mr. Mendoza turned around to watch the game. The hot dogs are on their own now.',
  'Grill report: the hot dogs have reached "crispy." Mr. Mendoza calls that "well done."',
  'Mr. Mendoza has made more burgers than this game has runs. Probably.',
  'The grill is still going. Scientists are baffled.',
];
const SPONSOR = [
  'This inning is brought to you by... nobody. We\'re still looking for sponsors.',
  'Our sponsor: Chet\'s Lemonade Stand. Twenty-five cents. Ice is extra.',
  'Sponsored by Dottie\'s brother\'s lawn-mowing business. Call now! Ask for Doug!',
  'A word from our sponsors: please return Mrs. Mendoza\'s lawn chairs.',
  'Brought to you by the ice cream truck. We can hear it! It\'s two streets over!',
  'Chet Valentine Broadcasting. We\'re on the air. The air is this backyard.',
];
const TEEBALL = [
  'Back in my big-league days, I batted ninth. Ninth is the cleanup spot for winners.',
  'I played one whole season of tee-ball. Some people call that a career.',
  'My tee-ball coach said I had a "unique stance." I think he meant amazing.',
  'People ask if I miss the big leagues. Every day, Chet. Every day.',
  'In tee-ball the ball doesn\'t even move. Talk about pressure.',
  'I still have my tee-ball trophy. Everybody got one. Mine\'s shinier.',
];
const POOL = [
  'Reminder, folks: the pool is not in play. Unless it is. Then it\'s a double.',
  'The flamingo floatie has been watching this whole game. Very focused.',
  'Mrs. Mendoza just reminded everybody to aim left. Again.',
  'Somebody left their goggles on the diving board. Not it.',
];
/** Chet sets them up, Dottie knocks them down. */
const RIFFS: [string, string][] = [
  ['Dottie, any thoughts so far?', 'Just one, Chet: I was a big leaguer.'],
  ['Folks, we\'ve got a world-class broadcast for you today.', 'We\'ve got a toy microphone and a lawn chair, Chet.'],
  ['Let\'s go to the replay.', 'We don\'t have replay. We have my memory. It\'s excellent.'],
  ['I\'d like to thank my tie for being here today.', 'It\'s a clip-on, Chet.'],
  ['Back to you, Dottie.', 'I\'m sitting right next to you.'],
  ['What a beautiful day for baseball.', 'The sun\'s in my eyes and a bee is following me. Sure.'],
  ['Dottie, what does it take to win in this league?', 'Snacks, Chet. It\'s mostly snacks.'],
  ['Chet Valentine here, coming to you live.', 'As opposed to what, Chet?'],
  ['Our producers tell me we\'re running long.', 'We don\'t have producers. That was your little sister.'],
  ['Dottie, as a former professional...', 'Thank you for noticing, Chet.'],
];

export class Booth {
  private rng: Rng;
  /** per pool: indices used recently */
  private recent = new Map<string, number[]>();
  /** the last few dozen lines said, in any pool */
  private said: string[] = [];
  private lastBatter = '';
  private introduced = new Set<string>();
  private lastContact: { ev: number; la: number; spray: number } | null = null;
  private playOuts = 0;
  private prevType = '';
  private maxDeficit: [number, number] = [0, 0];
  private atBatScore: [number, number] = [0, 0];
  private grill = 0;
  private splashes = 0;
  private batters = 0;
  /** a walk / HBP just happened: a run now means it was forced in */
  private walked = false;

  constructor(seed: number) {
    this.rng = new Rng(seed ^ 0xb00b);
  }

  /** Picks a line from a pool, avoiding the ones used recently. */
  private pick(key: string, opts: string[]): string {
    if (opts.length === 1) return opts[0];
    const used = this.recent.get(key) ?? [];
    let idx = opts.map((_, i) => i).filter((i) => !used.includes(i) && !this.said.includes(opts[i]));
    if (!idx.length) idx = opts.map((_, i) => i).filter((i) => i !== used[used.length - 1] && opts[i] !== this.said[this.said.length - 1]);
    if (!idx.length) idx = opts.map((_, i) => i);
    const i = this.rng.pick(idx);
    used.push(i);
    const memory = Math.max(1, Math.ceil(opts.length * 0.7));
    while (used.length > memory) used.shift();
    this.recent.set(key, used);
    return opts[i];
  }

  private chance(p: number) { return this.rng.chance(p); }

  /** Remembers what was said so the next lines avoid it. */
  private done(lines: Line[]): Line[] {
    for (const l of lines) this.said.push(l.text);
    if (this.said.length > 60) this.said.splice(0, this.said.length - 60);
    return lines;
  }

  /** Returns 0–2 lines for an event (most events say nothing). */
  react(e: MatchEvent, m: Match): Line[] {
    for (const s of [0, 1] as const) this.maxDeficit[s] = Math.max(this.maxDeficit[s], m.score[1 - s] - m.score[s]);
    const out = this.done(this.lines(e, m));
    if (e.type !== 'run') this.walked = e.type === 'walk';
    this.prevType = e.type;
    return out.slice(0, 2);
  }

  // ───────────────────────────────────────────── helpers

  private scoreText(m: Match) {
    return `${m.cfg.away.team.abbr} ${m.score[0]}, ${m.cfg.home.team.abbr} ${m.score[1]}`;
  }

  private isPoolParty(m: Match) {
    return m.cfg.yard.id === 'poolparty';
  }

  private positionOf(m: Match, id: string): Position | null {
    for (const s of [0, 1] as const) {
      const i = m.side(s).lineup.defense.indexOf(id);
      if (i >= 0) return POSITIONS[i] ?? null;
    }
    return null;
  }

  /** A kid's own lines that haven't been said lately. */
  private flavor(id: string, part: keyof Flavor): string[] {
    return (FLAVOR[id]?.[part] ?? []).filter((s) => !this.said.includes(s));
  }

  /** A running gag for an idle moment. */
  private gag(m: Match): Line {
    const pool = this.isPoolParty(m);
    const r = this.rng.next();
    if (pool && r < 0.3) return D(GRILL[this.grill++ % GRILL.length]);
    if (r < 0.55) return C(this.pick('sponsor', SPONSOR));
    if (pool && r < 0.7) return D(this.pick('pool', POOL));
    return D(this.pick('teeball', TEEBALL));
  }

  // ───────────────────────────────────────────── the booth

  private lines(e: MatchEvent, m: Match): Line[] {
    switch (e.type) {
      case 'batterUp': return this.batterUp(e.batter, m);
      case 'pitch': return this.pitch(e);
      case 'special': return this.special(e.kid, e.special, m);
      case 'call': return this.call(e.call, m);
      case 'contact':
        this.lastContact = { ev: e.ev, la: e.la, spray: e.spray };
        this.playOuts = 0;
        return [];
      case 'strikeout': return this.strikeout(e.batter, e.looking, m);
      case 'walk': return this.walk(e.batter, e.hbp, m);
      case 'hit': return this.hit(e.batter, e.bases, m);
      case 'homeRun': return this.homeRun(e.batter, e.runs, m);
      case 'groundRule': return this.groundRule(e.batter, e.why);
      case 'catch': return this.catch(e.fielder, e.fly, e.hard, m);
      case 'bobble':
        return this.chance(0.4) ? [C(this.pick('bobble', [
          `${nick(e.fielder)} juggles it!`, `Bobbled by ${nick(e.fielder)}!`,
          `${nick(e.fielder)} is doing a little hot-potato routine.`, `Uh-oh, ${nick(e.fielder)} can't find the handle!`,
          `${nick(e.fielder)} fumbles it like a wet bar of soap.`, `${nick(e.fielder)} double-clutches it!`,
        ]))] : [];
      case 'error': return this.error(e.fielder);
      case 'throw': return this.throwLine(e.fielder, e.base);
      case 'out': return this.outLine(e.kind, e.runner, e.fielder);
      case 'run': return this.runLine(e.runner, m);
      case 'dog':
        return [C(this.pick('dog', [
          'The dog is involved now! THE DOG IS INVOLVED!', 'It hit the doghouse! Biscuit is NOT happy.',
          'Off the doghouse! Somebody check on Biscuit.', 'Biscuit just barked at the baseball. Biscuit is a critic.',
        ]))];
      case 'tree':
        return this.chance(0.6) ? [C(this.pick('tree', [
          'Into the tree! Leaves everywhere!', 'It\'s in the branches!', 'Off the tree! A squirrel just filed a complaint.',
          'The tree gets a piece of it! Leaves are raining down!', 'Into the leaves! It\'s like a pinball machine in there.',
        ]))] : [];
      case 'fence': {
        if (e.cleared || !this.chance(0.4)) return [];
        const what = e.kind === 'hedge' ? 'hedge' : e.kind === 'picket' ? 'pickets' : 'fence';
        return [C(this.pick(`fence-${what}`, [
          `Off the ${what}!`, `Rattles around off the ${what}!`, `It bangs off the ${what}! Just short!`,
          `Off the ${what}! Inches from a home run!`,
        ]))];
      }
      case 'quip': {
        if (!this.chance(0.1)) return [];
        const n = nick(e.kid);
        return [D(fit(this.pick('quip', [
          `Did ${n} just say "${e.text}" Iconic.`, `"${e.text}" Words to live by, ${n}.`,
          `${n} says "${e.text}" Truly a poet.`, `Somebody write that down. "${e.text}"`,
        ]), `${n} has a way with words.`))];
      }
      case 'pitchingChange': return this.pitchingChange(e.from, e.to, m);
      case 'halfOver': return this.halfOver(e.inning, e.half, m);
      case 'gameOver': return this.gameOver(e.winner, m);
      default:
        return [];
    }
  }

  private batterUp(id: string, m: Match): Line[] {
    const k = kid(id);
    if (this.lastBatter === k.id) return [];
    this.lastBatter = k.id;
    this.atBatScore = [m.score[0], m.score[1]];
    this.playOuts = 0;
    this.lastContact = null;
    this.batters++;
    // now and then the booth just chats instead of introducing the batter
    if (this.batters > 2 && this.chance(0.06)) {
      const c = this.pick('riff', RIFFS.map((r) => r[0]));
      return [C(c), D(RIFFS.find((r) => r[0] === c)![1])];
    }
    const side = m.battingSide;
    const team = m.side(side).team;
    const p = m.pitcher;
    const box = m.box[k.id]?.bat;
    const hits = box?.h ?? 0, ab = box?.ab ?? 0;
    const today = ab > 0 ? ` ${hits} for ${ab} today.` : '';
    const pos = this.positionOf(m, k.id);
    const n = k.nick;
    const lead = m.score[side] - m.score[1 - side];
    const lastInning = m.inning >= m.cfg.innings;
    const [b1, b2, b3] = m.bases;

    let intro: string | null = null;
    if (this.chance(0.6)) {
      if (b1 && b2 && b3) intro = this.pick('up-loaded', [
        `Bases loaded for ${n}! Everybody's mom is standing up.`,
        `Ducks on the pond, all three of 'em, and here's ${n}.`,
        `The bases are juiced! ${n} could be a hero right here.`,
        `Bases loaded, and ${n} steps in. I can't look. I'm looking.`,
      ]);
      else if (m.outs === 2 && (b2 || b3)) intro = this.pick('up-clutch', [
        `Two outs, runner in scoring position. Big spot for ${n}.`,
        `${n} up with two down and a runner who'd really like to come home.`,
        `Two outs. ${n} at the plate. This is what you practice for in the driveway.`,
      ]);
      else if (lastInning && lead < 0 && lead >= -3) intro = this.pick('up-late', [
        `Last licks for the ${team.name}! ${n} needs to get something going.`,
        `It's now or never for the ${team.name}. Here's ${n}.`,
        `Down ${-lead} in the last inning. ${n} steps in. No pressure, kid.`,
      ]);
      else if (m.outs === 0 && !b1 && !b2 && !b3 && this.chance(0.5)) intro = this.pick('up-leadoff', [
        `Leading off the inning for the ${team.name}: ${n}.`,
        `${n} leads off. Get us started, ${k.first}!`,
        `First up this inning, it's ${n}.`,
      ]);
    }
    if (!intro) {
      const bats = k.bats === 'L' ? 'left' : k.bats === 'S' ? 'either' : 'right';
      intro = this.pick('up', [
        `Now batting: ${k.first} "${n}" ${k.last}.`,
        `Stepping in, it's ${n}, ${persona(k)}.`,
        `Here comes ${n} to the plate.`,
        `Batting for the ${team.name}: ${n}, age ${k.age}.`,
        `${n} digs in from the ${bats} side.`,
        `Ladies and gentlemen... ${k.first} ${k.last}!`,
        pos && pos !== 'P' ? `${n}, who plays ${POS_NAME[pos]} for the ${team.name}, steps in.` : `${n} steps in for the ${team.name}.`,
        `And the crowd goes mild! It's ${n}.`,
        `${n} takes a practice swing. Very professional.`,
        `Approaching the batter's box: ${k.persona}, ${k.first} ${k.last}.`,
        `From right here on ${team.street}... it's ${n}!`,
        `${n} versus ${p.nick}. Here we go.`,
      ]);
    }
    const out = [C(fit(intro + today, intro))];

    if (this.chance(0.4)) {
      const fresh = !this.introduced.has(k.id);
      const flavor = this.flavor(k.id, 'up');
      if (fresh && this.chance(0.6)) {
        this.introduced.add(k.id);
        out.push(D(this.pick('bio', [k.bio, fit(`${n}? ${k.bio}`, k.bio), fit(`Fun fact: ${lower(k.bio)}`, k.bio)])));
      } else if (flavor.length && this.chance(0.55)) {
        out.push(D(this.pick(`up-${k.id}`, flavor)));
      } else if (this.chance(0.5)) {
        out.push(D(this.pick('matchup', [
          `${p.nick} against ${n}. ${p.persona} versus ${persona(k)}. Classic.`,
          `I give ${n} a fifty-fifty chance. Maybe sixty-forty. I'm not great at math.`,
          `If I were ${p.nick}, I'd throw ${n} a curveball. I'd also be taller.`,
          `${n} has that look today. The look of a kid who had pancakes.`,
          `Eye on the ball, ${n}! That's what my tee-ball coach told me. A big leaguer.`,
          `${poss(n)} stance is a lot like mine was. Mine was better.`,
          `${p.nick} better be careful with ${n}. Trust me. I've seen things.`,
        ].map((s) => fit(s, `${n} versus ${p.nick}. This should be good.`)))));
      } else {
        out.push(this.gag(m));
      }
    }
    return out;
  }

  private pitch(e: Extract<MatchEvent, { type: 'pitch' }>): Line[] {
    if (e.special || !this.chance(0.09)) return [];
    const p = kid(e.pitcher), n = p.nick, mph = Math.round(e.mph);
    const flavor = this.flavor(p.id, 'pitch');
    if (flavor.length && this.chance(0.25)) return [D(this.pick(`pitch-${p.id}`, flavor))];
    const byType: Record<string, string[]> = {
      fastball: [`${n} brings the heat! ${mph} on the radar gun!`, `Fastball, ${mph}! That's faster than my bike.`, `Here's the cheese from ${n}!`, `${n} just reared back and threw it. Hard.`],
      curve: [`${n} spins a curveball. It bends like a garden hose!`, 'Curveball! That one took the scenic route.', `${n} drops in the curve. Loopy!`],
      changeup: [`Changeup from ${n}. Slow as a Sunday afternoon.`, 'Oh, the changeup! Sneaky, sneaky.', `${n} pulls the string on a changeup. ${mph} miles an hour.`],
      slider: [`${n} with the slider. It slides right on by.`, `Slider from ${n}! Sideways like a crab.`],
      sinker: [`Sinker from ${n}. That one dove like a kid off the diving board.`, `${n} throws the sinker. Down it goes!`],
      knuckler: [`Knuckleball! Nobody knows where that's going. Including ${n}.`, 'Here comes the knuckler, wobbling like a jelly donut.'],
    };
    const opts = byType[e.pitch] ?? [`${n} deals. ${mph} on the gun.`];
    const out = [C(this.pick(`pitch-${e.pitch}`, opts))];
    if (this.chance(0.15)) out.push(D(this.pick('pitch-d', [
      'In tee-ball we didn\'t have pitches. We had a tee. Simpler times.',
      `${poss(n)} got good stuff today. Not as good as mine was. But good.`,
      'The radar gun is my cousin\'s. He says it\'s "pretty accurate."',
    ])));
    return out;
  }

  private special(id: string, special: keyof typeof SPECIAL_INFO, m: Match): Line[] {
    const info = SPECIAL_INFO[special];
    const n = nick(id), L = info.label.toUpperCase(), l = info.label;
    const kindLines = info.kind === 'bat'
      ? [`${n} is loading up the ${l}! Outfielders, back up!`, `${n} is digging in for the ${L}!`]
      : info.kind === 'pitch'
        ? [`${n} winds up for the ${L}! Batter, good luck!`, `${n} has the ${l} ready. Uh-oh.`]
        : [`${n} is turning on the ${l}!`];
    const out = [C(this.pick(`special-${info.kind}`, [
      `${n} is going for the ${L}!`, `Here it comes: the ${L}! ${n} is powering up!`,
      `${n} has that look. It's ${L} time!`, ...kindLines,
    ]))];
    if (this.chance(0.5)) out.push(D(this.pick('special-d', [
      'Oh, this is gonna be good.', 'I invented that move, by the way.', 'Somebody get the camcorder!',
      'I did that once in tee-ball. The tee fell over.', 'I\'m not saying I taught them that. But I\'m not NOT saying it.',
      'Hold onto your juice boxes, folks!',
      ...(this.isPoolParty(m) ? ['Mr. Mendoza just put down his spatula. That\'s how you know it\'s serious.'] : []),
    ])));
    return out;
  }

  private call(call: 'ball' | 'strike' | 'foul' | 'swinging', m: Match): Line[] {
    // the deciding pitch is covered by the walk / strikeout lines
    if (m.balls >= 4 || m.strikes >= 3) return [];
    const b = m.batter.nick, p = m.pitcher.nick, bl = m.balls, s = m.strikes;
    if (call === 'foul') {
      if (!this.chance(0.13)) return [];
      return [C(this.pick('foul', [
        'Foul ball!', 'Fouled back. Somebody check the windows.', `${b} fights it off. Foul.`, 'Fouled away. Still alive!',
        ...(this.isPoolParty(m) ? ['Foul, into the flower bed. Mrs. Mendoza won\'t love that.', 'That foul went toward the grill! Mr. Mendoza didn\'t even flinch.'] : ['Foul, into the bushes.']),
        `${b} just gets a piece of it. Foul.`,
      ]))];
    }
    if (bl === 3 && s === 2 && this.chance(0.5)) return [C(this.pick('full', [
      'Full count! Everybody hold your breath.', `Three and two. ${p} and ${b}, eyeball to eyeball.`,
      'Full count! Mrs. Mendoza just covered her eyes.', 'Payoff pitch coming up. This is the good part.',
    ]))];
    if (!this.chance(0.1)) return [];
    if (call === 'ball') return [C(this.pick('ball', [
      'Ball. Just missed.', `Outside. Ball ${bl}.`, `${p} misses high.`, `Ball ${bl}. ${b} wasn't biting.`,
      `That's ball ${bl}. ${p} is shaking it off.`, `Low. ${b} lays off it.`,
    ]))];
    if (s === 2) return [C(this.pick('two-strikes', [
      `${b} is down to the last strike.`, `${bl} and 2. ${p} has ${b} right where ${p} wants.`,
      `Two strikes on ${b}. Choke up, kid!`, `${b} is in a hole now. ${bl} and 2.`,
    ]))];
    if (call === 'strike') return [C(this.pick('strike', [
      `Strike ${s}, looking.`, `${b} watches that one go by. Strike ${s}.`, 'Right down the middle. Strike!', `${p} paints the corner. Strike ${s}.`,
    ]))];
    return [C(this.pick('swinging', [
      'Swing and a miss!', `${b} cuts and misses. Strike ${s}.`, 'Whiff! The breeze feels nice, though.', `${b} swung so hard the hat came off!`,
    ]))];
  }

  private strikeout(id: string, looking: boolean, m: Match): Line[] {
    const n = nick(id), p = m.pitcher, pn = p.nick;
    const out = [C(looking
      ? this.pick('k-looking', [
        `${n} watches strike three go by!`, `Caught looking! ${n} is frozen like a popsicle.`,
        `Strike three called! ${n} didn't move a muscle.`, `${n} takes strike three. Bat never left the shoulder.`,
        `Called strike three! ${n} looks for the umpire. There is no umpire.`, `Ooh, painted the corner. ${n} is out looking.`,
        `Strike three! ${n} admired that one like it was in a museum.`,
      ])
      : this.pick('k-swinging', [
        `Struck out swinging! ${n} spun all the way around.`, `Strike three! ${n} swung at a ghost.`,
        `${n} went fishing and caught nothing. Strike three!`, `Swing and a miss, strike three! ${n} nearly drilled into the lawn.`,
        `${n} swings right through it! Sit down, partner.`, `Whiff! ${n} made a nice breeze for the infield.`,
        `Got 'em! ${pn} gets ${n} swinging.`,
      ]))];
    if (this.chance(0.4)) {
      const so = m.box[p.id]?.pitch.so ?? 0;
      const opts = [
        `${pn} is dealing today.`, 'In my day we called that "the windmill."', 'That pitch had some MUSTARD on it.',
        'Shake it off, kid. Shake it off.', 'Even I would\'ve struck out on that. And I played in the bigs. Tee-ball bigs.',
        'That ball moved like an ice cream truck going the wrong way.', 'Back to the dugout. A juice box will make it better.',
        ...this.flavor(id, 'k'), ...this.flavor(p.id, 'pitch'),
      ];
      out.push(so >= 4 && this.chance(0.5)
        ? D(this.pick('k-count', [`That's ${so} strikeouts for ${pn} today!`, `${pn} has ${so} K's. I'm drawing them on my hand.`, `Strikeout number ${so} for ${pn}. Somebody check that arm for batteries.`]))
        : D(this.pick('k-d', opts)));
    }
    return out;
  }

  private walk(id: string, hbp: boolean, m: Match): Line[] {
    const n = nick(id), p = m.pitcher.nick;
    const loaded = m.bases.every(Boolean);
    const out = [C(hbp
      ? this.pick('hbp', [
        `Ouch! ${n} takes one for the team.`, `Plunked! ${n} is rubbing it and heading to first.`,
        `${n} wears one! Take your base.`, `That one got ${n} right in the elbow pad. Which ${n} doesn't have.`,
        `Hit by pitch! ${n} is being very brave about it.`,
      ])
      : loaded ? this.pick('walk-loaded', [`Ball four, and the bases are loaded!`, `${n} walks, and now it's bases loaded!`, `${n} takes the free pass. Bases are full!`])
        : this.pick('walk', [
          `Ball four. ${n} takes a walk.`, `${n} draws the walk. Patience!`, `${n} trots to first. Ball four.`,
          `Ball four! ${n} didn't even have to swing.`, `${p} can't find the plate. ${n} walks.`, `Free pass for ${n}!`,
        ]))];
    if (this.chance(0.3)) out.push(D(hbp
      ? this.pick('hbp-d', ['That\'s gonna leave a mark. A cool mark, though.', 'Rub some dirt on it! Actually, don\'t. The lawn was just reseeded.', `${p} says sorry. ${p} better mean it.`])
      : this.pick('walk-d', [
        'A walk\'s as good as a hit. That\'s what my tee-ball coach said. We didn\'t have walks.',
        'Free base! Like a free sample at the grocery store.',
        `${p} needs to find the strike zone. It's the rectangle, buddy.`,
        'Good eye! I had good eyes too. Still do. Twenty-twenty. Look it up.',
      ])));
    return out;
  }

  /** The score line after runs come in on a hit, if they changed the story. */
  private runsStory(m: Match): { runs: number; lead: boolean; line: string | null } {
    const side = m.battingSide;
    const runs = m.score[side] - this.atBatScore[side];
    if (runs <= 0) return { runs: 0, lead: false, line: null };
    const before = this.atBatScore[side] - this.atBatScore[1 - side];
    const now = m.score[side] - m.score[1 - side];
    const team = m.side(side).team.name;
    const sc = this.scoreText(m);
    if (now > 0 && before <= 0) return { runs, lead: true, line: this.pick('go-ahead', [`And the ${team} take the lead! ${sc}.`, `That puts the ${team} on top! ${sc}.`, `Lead change! It's ${sc}.`]) };
    if (now === 0) return { runs, lead: true, line: this.pick('tied', [`Tie ballgame! ${sc}.`, `We're all square! ${sc}.`, `And this one is TIED. ${sc}.`]) };
    return { runs, lead: false, line: this.pick('rbi', [
      runs === 1 ? `A run scores! ${sc}.` : `${runs} runs score! ${sc}.`,
      runs === 1 ? `Here comes the run! ${sc}.` : `${runs} across! ${sc}.`,
      `That's ${runs === 1 ? 'an RBI' : `${runs} RBIs`}! ${sc}.`,
    ]) };
  }

  private hit(id: string, bases: number, m: Match): Line[] {
    const k = kid(id), n = k.nick;
    const kind = bases >= 3 ? 'triple' : bases === 2 ? 'double' : 'single';
    const la = this.lastContact?.la ?? 15;
    const shape = la < 8 ? 'ground' : la > 28 ? 'bloop' : 'liner';
    const pool: Record<string, string[]> = {
      single: [
        `Base hit for ${n}!`, `${n} pokes one through!`, `Single for ${n}. Nothing fancy, gets the job done.`,
        `${n} slaps it into the outfield. Base hit!`, `That's a knock! ${n} is aboard.`, `${n} finds a hole! Single.`,
        ...(shape === 'ground' ? [`Seeing-eye grounder gets through! ${n} on first.`, `${n} chops it past the infield!`]
          : shape === 'bloop' ? [`Little bloop... it falls in! ${n} has a single.`, `${n} dunks one in! A duck snort!`]
            : [`Line drive single for ${n}!`, `${n} ropes one up the middle!`]),
      ],
      double: [
        `${n} lines one into the gap. That's a double!`, `Two-bagger for ${n}!`, `${n} cruises into second with a double!`,
        `Into the corner! ${n} coasts into second.`, `${n} splits the outfielders! Stand-up double.`,
        `Double for ${n}! Somebody's buying the ice pops.`,
      ],
      triple: [
        `A TRIPLE for ${n}! Nobody hits triples!`, `${n} is flying around the bases! TRIPLE!`,
        `Three bases for ${n}! Somebody get that kid a water.`, `${n} slides into third! A triple!`,
        `The rarest hit in the backyard: a triple, by ${n}!`,
      ],
    };
    const out = [C(this.pick(`hit-${kind}-${kind === 'single' ? shape : ''}`, pool[kind]))];
    const story = this.runsStory(m);
    if (story.line && (story.runs > 1 || this.chance(0.7))) out.push(C(story.line));
    else if (this.chance(bases >= 2 ? 0.45 : 0.25)) {
      out.push(D(this.pick('hit-d', [
        'That\'s what we call a frozen rope. Or a hot rope. Some kind of rope.',
        `${poss(n)} mom is gonna put that on the fridge.`, 'I used to hit \'em like that. Off a tee, but still.',
        `Nice piece of hitting by ${n}. Textbook. I wrote the textbook. In crayon.`,
        `${m.pitcher.nick} won't want to see that one again.`,
        ...this.flavor(id, 'hit'),
      ])));
    }
    return out;
  }

  private homeRun(id: string, runs: number, m: Match): Line[] {
    const k = kid(id), n = k.nick;
    const c = this.lastContact;
    const spray = c?.spray ?? 0;
    const where = !this.isPoolParty(m) ? 'the fence' : spray > 12 ? 'the picket fence' : spray < -12 ? 'the hedge' : 'the fence in center';
    const extra = runs === 2 ? ' Two runs score!' : runs === 3 ? ' Three runs score!' : '';
    const call = runs >= 4
      ? this.pick('slam', [`GRAND SLAM! ${n} clears the bases!`, `It's a GRAND SLAM for ${n}! Four runs!`, `Bases loaded, and ${n} empties them! GRAND SLAM!`, `${n} hits a GRAND SLAM! Everybody touch 'em all!`])
      : this.pick('hr', [
        `IT'S OUTTA HERE! ${n} crushed it!`, `GOODBYE, BASEBALL! A home run for ${n}!`,
        `Somebody call the neighbors! ${n} hit it into the next zip code!`, `Going... going... GONE! ${n} with a home run!`,
        `Over ${where}! ${n} goes yard! In a yard!`, `${n} got all of that one! HOME RUN!`,
        `See ya! ${n} sends it over ${where}!`, `${n} hits a homer! Touch 'em all, ${k.first}!`,
      ]);
    const out = [C(fit(call + extra, call))];
    const story = this.runsStory(m);
    if (story.line && story.lead && this.chance(0.6)) { out.push(C(story.line)); return out; }
    const ev = c?.ev ?? 0;
    const far = ev >= 78, barely = ev > 0 && ev < 62;
    const opts = far
      ? ['That ball is still going, folks.', 'That one landed in the street. Somebody check the parked cars.', 'I think that ball is headed for the water tower.', `${n} hit that one to another neighborhood!`]
      : barely
        ? ['Just over! That one scraped the paint off the fence.', 'That barely cleared. I\'ll allow it.', 'Just enough! I could\'ve caught that. In my prime.']
        : ['Somebody\'s gonna have to knock on a door for that one.', 'I hit one like that once. In tee-ball. It was off the tee, but still.', `${n} is gonna be insufferable at lunch tomorrow.`, 'That\'s a souvenir for the neighbors.'];
    if (this.isPoolParty(m)) {
      opts.push('Mr. Mendoza is pointing his spatula at it. That\'s the highest honor.');
      if (id === 'kai') opts.push('Kaboom homers in his own backyard! Mrs. Mendoza is checking the windows.');
    }
    out.push(D(this.pick(`hr-d-${far ? 'far' : barely ? 'barely' : 'mid'}`, [...opts, ...this.flavor(id, 'hit')])));
    return out;
  }

  private groundRule(id: string, why: 'splash' | 'bounce'): Line[] {
    const n = nick(id);
    if (why === 'splash') {
      this.splashes++;
      const out = [C(this.pick('splash', [
        'SPLASH! It\'s in the water! That\'s a splash double!', 'Into the drink! Splash double!',
        `${n} goes swimming! Splash double!`, 'KER-SPLOOSH! Into the pool for a double!',
        `${n} finds the deep end! Two bases!`, 'Cannonball! That one\'s in the pool. Splash double!',
      ]))];
      out.push(D(this.splashes >= 2 && this.chance(0.4)
        ? this.pick('splash-count', [`Splash number ${this.splashes} today! The pool is undefeated.`, `That's ${this.splashes} in the pool today. Somebody get a net.`])
        : this.pick('splash-d', [
          'Somebody get the pool skimmer.', 'Hope nobody was swimming.', 'That ball needs a towel.',
          'Mrs. Mendoza did say aim left. Several times.', 'The flamingo floatie never saw it coming.',
          'That ball should\'ve waited thirty minutes after eating.', 'No running by the pool! Oh, it\'s a ball. Carry on.',
        ])));
      return out;
    }
    return [C(this.pick('bounce', [
      'Bounced over the fence! Ground-rule double.', 'One hop and over! That\'s two bases.',
      'Ground-rule double! It hopped the fence like a squirrel.', `${n} bounces one over. Ground-rule double!`,
    ]))];
  }

  private catch(id: string, fly: boolean, hard: boolean, m: Match): Line[] {
    const n = nick(id);
    const pos = this.positionOf(m, id);
    if (fly && hard) {
      const out = [C(this.pick('catch-hard', [
        `WHAT A CATCH by ${n}!`, `${n} lays out and MAKES THE GRAB!`, `Are you kidding me?! ${n} caught it!`,
        `${n} goes all out... and HAS IT!`, `Diving catch! ${n} is covered in grass stains!`,
        `${n} snags it! That's going on the highlight tape!`, `No way! ${n} just robbed that hit!`,
      ]))];
      if (this.chance(0.6)) out.push(D(this.pick('catch-d', [
        'Put that on the refrigerator!', 'Mom, are you filming?!', 'I made a catch like that once. It was a juice box, but still.',
        'Grass stains are a badge of honor. My mom disagrees.', `${poss(n)} getting a sticker for that one.`,
        ...this.flavor(id, 'field'),
      ])));
      return out;
    }
    if (fly) {
      if (!this.chance(0.3)) return [];
      return [C(this.pick('catch-fly', [
        `${n} squeezes it. Out.`, `Easy catch for ${n}.`, `Can of corn to ${n}.`,
        `${n} camps under it... and makes the catch.`, `High fly ball... ${n} has it.`,
        pos ? `Routine fly. ${n} ${POS_AT[pos]} puts it away.` : `${n} puts it away.`,
        `${n} calls for it, ${n} gets it.`, `Up, up... and down into ${poss(n)} glove.`,
      ]))];
    }
    if (!this.chance(0.12)) return [];
    return [C(this.pick('catch-ground', [
      `${n} scoops it up.`, `${n} gets a glove on it.`, `${n} charges it...`,
      pos ? `${n} ${POS_AT[pos]} picks it up.` : `${n} picks it up.`, `${n} knocks it down!`,
    ]))];
  }

  private error(id: string): Line[] {
    const n = nick(id);
    const out = [C(this.pick('error', [
      `Oh no, ${n} drops it!`, `It's off the glove! Error, ${n}!`, `${n} had it... and now doesn't.`,
      `Right through the legs! Error on ${n}!`, `${n} boots it!`, `Oops! ${n} would like that one back.`,
      `The ball squirts away from ${n}! Error!`,
    ]))];
    if (this.chance(0.5)) out.push(D(this.pick('error-d', [
      'That\'s going in the blooper reel.', 'Happens to the best of us. Mostly to the worst of us.',
      'Somebody needs a bigger glove.', `Shake it off, ${n}. I once missed a ball sitting on a tee.`,
      'The sun was in their eyes. Or a bee. Let\'s say a bee.', 'Errors build character. I have SO much character.',
    ])));
    return out;
  }

  private throwLine(id: string, base: number): Line[] {
    const k = kid(id), n = k.nick, b = BASE[base] ?? 'the base';
    if (k.special === 'rocketArm' && this.chance(0.3)) return [C(this.pick('throw-rocket', [
      `ROCKET ARM! ${poss(n)} throw has a vapor trail!`, `${n} fires a rocket to ${b}!`, `Did you SEE that throw from ${n}?!`,
    ]))];
    if (!this.chance(0.1)) return [];
    if (base === 0) return [C(this.pick('throw-home', [`${n} comes home with it!`, `Here's the throw to the plate from ${n}!`, `${n} with the long throw home!`]))];
    return [C(this.pick('throw', [`${n} fires to ${b}!`, `${n} comes up throwing... to ${b}!`, `Here's the throw to ${b}!`, `${n} throws to ${b}.`]))];
  }

  private outLine(kind: string, runner: string, fielder: string): Line[] {
    this.playOuts++;
    const r = nick(runner), f = nick(fielder);
    if (this.playOuts === 3) return [C(this.pick('tp', ['A TRIPLE PLAY?! I\'ve only seen that in cartoons!', 'TRIPLE PLAY! Somebody pinch me!']))];
    if (this.playOuts === 2) return [C(this.pick('dp', [
      'DOUBLE PLAY! Two for the price of one!', `Two outs on one play! ${f} turns it!`, 'Around the horn! It\'s a double play!',
      'Double play! That\'s a twin-killing, folks. That\'s a real term.',
    ]))];
    if (kind === 'doubledOff') return [C(this.pick('doubled-off', [`${r} gets doubled off! Should've stayed home!`, `${r} was halfway to the snack table! Doubled off!`, `Back, ${r}, back! Too late. Doubled off.`]))];
    if (kind === 'tag' && this.chance(0.5)) return [C(this.pick('tag', [
      `Tagged out! ${r} is out!`, `${f} applies the tag. Gotcha!`, `${r} tries to sneak by... tagged!`,
      `${f} slaps the tag on ${r}! Out!`, `Out! ${r} slid right into ${poss(f)} glove.`,
    ]))];
    if (kind === 'force' && this.chance(0.15)) return [C(this.pick('force', [`Got the force! ${r} is out.`, `${r} is forced out.`, `${f} steps on the bag. Force out!`]))];
    return [];
  }

  private runLine(id: string, m: Match): Line[] {
    const n = nick(id);
    if (this.walked) return [C(this.pick('walk-in', [`${n} walks in a run! Free run!`, `And ${n} trots home. A run walks in!`, `Bases-loaded walk! ${n} scores!`]))];
    if (m.phase !== 'live' || !this.chance(0.12)) return [];
    return [C(this.pick('run', [`${n} crosses the plate!`, `Here comes ${n} to score!`, `${n} touches home!`, `${n} scores! High-fives all around!`]))];
  }

  private pitchingChange(from: string, to: string, m: Match): Line[] {
    const f = nick(from), t = nick(to);
    const pos = this.positionOf(m, from);
    const out = [C(this.pick('pchange', [
      `Pitching change! ${f} heads out to the field, and ${t} takes the ball.`,
      `${f} is gassed. Here comes ${t} to pitch!`,
      pos ? `New arm on the mound: ${t}! ${f} moves to ${POS_NAME[pos]}.` : `New arm on the mound: ${t}!`,
      `${t} is the new pitcher. ${f}, go get some water.`,
      `${f} hands the ball to ${t}. That's a pitching change, folks.`,
    ]))];
    out.push(D(this.pick('pchange-d', [
      'Fresh arm! In my day we pitched until the streetlights came on.',
      `${poss(f)} arm is noodles. Wet noodles.`,
      `${t} has been warming up by throwing at the garage door. Very professional.`,
      ...this.flavor(to, 'pitch'),
    ])));
    return out;
  }

  private halfOver(inning: number, half: 0 | 1, m: Match): Line[] {
    const [a, h] = m.score;
    const n = m.cfg.innings;
    const sc = this.scoreText(m);
    const half$ = half === 0 ? 'top' : 'bottom';
    const ord = ordinal(inning);
    const out = [C(this.pick('half', [
      `That'll do it for the ${half$} of the ${ord}. ${sc}.`,
      `Three outs! End of the ${half$} of the ${ord}: ${sc}.`,
      `And that's the side. After ${inning}${half === 0 ? ' and a half' : ''}, it's ${sc}.`,
      `${half === 0 ? 'Middle' : 'End'} of the ${ord}. ${sc}.`,
      `Side retired. ${sc}, ${half$} of the ${ord} in the books.`,
    ]))];
    // the game-over line follows on its own
    if ((half === 0 && inning >= n && h > a) || (half === 1 && inning >= n && a !== h)) return out;
    if (!this.chance(0.75)) return out;
    const diff = Math.abs(a - h);
    const lead: 0 | 1 = a > h ? 0 : 1;
    const leader = m.side(lead).team.name, trailer = m.side((1 - lead) as 0 | 1).team.name;
    const comeback = diff > 0 && this.maxDeficit[lead] >= 3;
    if (half === 1 && inning >= n && a === h) {
      out.push(D(this.pick('extras', ['We\'re going to EXTRA INNINGS! Nobody tell our moms.', 'Extra innings! I\'ll need another juice box.', 'Free baseball! The streetlights aren\'t on yet. Keep going!'])));
    } else if (half === 0 && inning === n) {
      out.push(C(a === h ? this.pick('last-tied', [`Tied up, last inning. The ${m.cfg.home.team.name} can win it right here.`, 'All even, bottom of the last. Here we go.'])
        : this.pick('last-trail', [`Last licks for the ${m.cfg.home.team.name}, down ${diff}.`, `The ${m.cfg.home.team.name} need ${diff === 1 ? 'a run' : `${diff} runs`} to tie. Last chance!`])));
    } else if (comeback && this.chance(0.7)) {
      out.push(D(this.pick('comeback', [
        `What a comeback by the ${leader}! They were down ${this.maxDeficit[lead]} earlier!`,
        'Never count out a backyard team. NEVER.', `The ${leader} came all the way back. I'm getting goosebumps.`,
      ])));
    } else if (diff === 0) {
      out.push(D(this.pick('tie', ['All tied up! This is better than Saturday cartoons.', 'Tie game. Nobody blink.', 'Deadlocked! I\'m too nervous to finish my juice box.', 'Even steven. This is anybody\'s game.'])));
    } else if (diff >= 6) {
      out.push(D(this.pick('blowout', [
        `The ${leader} are running away with this one.`, `${trailer} fans, there's still... well, there's still snacks.`,
        'This is getting out of hand. Do we have a mercy rule? Somebody check.',
        `The ${trailer} might need a miracle. Or a really long inning.`,
      ])));
    } else if (diff <= 2 && this.chance(0.5)) {
      out.push(C(this.pick('close', [
        `The ${leader} hang on to a ${diff}-run lead.`, `Close one! The ${trailer} are right there.`,
        diff === 1 ? 'A one-run game... wait, let me count. Yep, one run.' : `Just ${diff} runs in it. Anybody's game.`,
      ])));
    } else if (half === 1 && inning === n - 1) {
      out.push(C(this.pick('to-last', [`Heading to the final inning! The ${leader} lead by ${diff}.`, `One inning left. The ${trailer} need ${diff} to tie.`])));
    } else {
      out.push(this.gag(m));
    }
    return out;
  }

  private gameOver(winner: 0 | 1 | -1, m: Match): Line[] {
    const [a, h] = m.score;
    const n = m.cfg.innings;
    const sc = this.scoreText(m);
    if (winner === -1) return [
      C(this.pick('tie-end', ['It\'s a tie! Everybody\'s mom is calling them in for dinner.', `The streetlights are on. We'll call it a tie, ${sc}.`, 'A tie! Same time tomorrow, folks?'])),
      D(this.pick('tie-end-d', ['A tie is like kissing your sister. Not that I would know.', 'Nobody lost! That\'s what my tee-ball coach said every game.'])),
    ];
    const w = m.side(winner).team, l = m.side((1 - winner) as 0 | 1).team;
    const diff = Math.abs(a - h);
    const walkOff = winner === 1 && this.prevType !== 'halfOver';
    let call: string;
    if (walkOff) call = this.pick('walkoff', [`WALK-OFF! The ${w.street} ${w.name} win it at home!`, `They win it! The ${w.name} walk it off, ${sc}!`, `Ballgame! A walk-off for the ${w.name}! Everybody dogpile!`]);
    else if (Math.min(a, h) === 0) call = this.pick('shutout', [`That's the ballgame! A shutout for the ${w.street} ${w.name}, ${sc}.`, `Final: ${sc}. The ${l.name} never got on the board!`]);
    else if (diff >= 6) call = this.pick('blowout-end', [`That's all, folks! The ${w.name} roll, ${sc}.`, `Final score: ${sc}. The ${w.name} ran away with it.`]);
    else if (m.inning > n) call = this.pick('extras-end', [`The ${w.name} win it in extra innings! ${sc}.`, `Free baseball pays off for the ${w.name}! Final: ${sc}.`]);
    else call = this.pick('win', [`That's the ballgame! The ${w.street} ${w.name} win it!`, `Final score: ${sc}. The ${w.name} win!`, `And that'll do it! The ${w.name} take this one, ${sc}.`, `Put it in the books: ${w.street} ${w.name} win, ${sc}.`]);
    return [C(call), D(this.pick('end-d', [
      'What a game. I need a juice box.', 'I\'ll be signing autographs by the swing set.', 'Same time tomorrow?',
      `Good game, ${l.name}. Line up for high-fives, everybody.`, 'That\'s a wrap! I gotta be home before the streetlights.',
      ...(this.isPoolParty(m) ? ['Last one in the pool is a rotten egg! Kidding. Mrs. Mendoza said no.', 'Mr. Mendoza says the burgers are finally ready. Finally!'] : []),
    ]))];
  }
}

export function ordinal(n: number) {
  const s = ['th', 'st', 'nd', 'rd'];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

export function personaLine(k: Kid) {
  return `${k.persona} · Age ${k.age}`;
}
