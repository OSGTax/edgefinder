import { Rng } from '../engine/rng';
import { KIDS, kid } from '../data/kids';
import { TEAMS } from '../data/teams';
import { POSITIONS, SPECIAL_INFO, type Kid, type Position } from '../data/types';
import type { MatchEvent } from '../sim/types';
import type { Match } from '../sim/match';

// Live from the Mendozas' backyard, on Channel 4½, Maple Hollow Public Access.
//
// Chet Valentine (play-by-play, age 10, clip-on tie, hair gel, a toy
// microphone; has been "in the business" since second grade and treats this
// like network TV) and Dottie Fairweather (color, age 12, "former big
// leaguer" from one season of tee-ball, and she will bring it up). The camera
// is Chet's dad's camcorder; the producer is Chet's little sister Bea, who
// holds the cue cards and mostly eats the snacks.
//
// The booth never says the same line twice in a game: once a line is used
// it's gone, and if a moment has nothing fresh left the booth just lets the
// play speak for itself. All randomness goes through the booth's own seeded
// Rng, so a replayed game says the same things.

export interface Line { who: 'Chet' | 'Dottie'; text: string }

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
const NUM = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'];
const num = (n: number) => NUM[n] ?? String(n);
/** Use the long version if it fits the ticker, else the short one. */
const fit = (long: string, short: string, max = 110) => (long.length <= max ? long : short);

/** Every name a line can be filled in with, longest first so "Gus-Gus" goes before "Gus". */
const NAMES = [...new Set([
  ...KIDS.flatMap((k) => [k.nick, k.first, k.last, k.persona, lower(k.persona), String(k.age)]),
  ...TEAMS.flatMap((t) => [t.name, t.street, t.abbr]),
])].sort((a, b) => b.length - a.length);
const NAME_RE = new RegExp(NAMES.map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'), 'g');
/**
 * A line with its names and numbers taken out: "From right here on Cedar Lane... it's
 * Kaboom!" and "...on Maple Street... it's Dez!" are the same line to a listener.
 */
export const shape = (text: string) =>
  text.replace(NAME_RE, '#').replace(/\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)(st|nd|rd|th)?\b/gi, '#');

// ───────────────────────────────────────────── per-kid flavor
// Keyed by kid id; missing kids (future teams) simply get the general lines.
interface Flavor {
  up: string[]; hit?: string[]; hr?: string[]; k?: string[]; walk?: string[];
  pitch?: string[]; field?: string[]; error?: string[];
}
const FLAVOR: Record<string, Flavor> = {
  bo: {
    up: ['Mudpie\'s knees are acting up again. He told me personally. Twice.', 'Bo says this bat "doesn\'t make \'em like they used to." It\'s from last Tuesday.', 'Mudpie brought his wrench to the on-deck circle. In case something leaks.', 'Bo is stretching his back. It takes a while. He\'s eleven.'],
    hit: ['Mudpie fixed THAT leak!', 'Bo\'s gonna need an ice pack for those knees after all that running.', 'Bo just said "oof" rounding first. Every step. Oof, oof, oof.'],
    hr: ['Mudpie\'s taking his home run trot at plumber speed. Billable hours, folks.', 'Bo hit that one into the next zip code. He\'ll charge them for the trip.'],
    k: ['Mudpie\'s blaming the pipes. There are no pipes.', 'Bo says strike three "was a clog." Sure, Bo.'],
    pitch: ['Mudpie\'s pitching. He says he\'ll "take a look at it" and charge us extra.', 'Bo throws every pitch with a little grunt. Retired-plumber stuff.'],
    error: ['Mudpie says the glove "has a leak." He\'ll fix it after the game. Maybe Tuesday.'],
  },
  ines: {
    up: ['Inny\'s wearing sunglasses at the plate. It\'s for the drama.', 'Last week Inny fainted in the on-deck circle. The paramedic was a juice box.', 'Inny says this at-bat is "a two-part episode."', 'Inny just checked her reflection in the batting helmet. Ready for her close-up.'],
    hit: ['Inny\'s taking a bow at first base. Several bows.', 'Inny is blowing kisses to her public. Her public is four moms.'],
    hr: ['Inny\'s rounding the bases like it\'s the season finale. Somebody cue the strings.', 'Inny touched home and fainted. Into a lawn chair. Perfectly on purpose.'],
    k: ['Inny is lying in the dirt. She\'s fine. She\'s ACTING.', 'Inny says she was ROBBED. Dramatic music, please.'],
    walk: ['Inny is walking to first like it\'s a red carpet.'],
    pitch: ['Inny on the mound. Somebody cue the dramatic music.', 'Every Inny pitch is a season finale. Stay tuned.', 'Inny just gave the batter a long, meaningful stare. Then she threw it.'],
    error: ['Inny dropped it, and now she\'s dropped to her knees. "WHY?!" she says.'],
  },
  toby: {
    up: ['Toby just blew his whistle at the pitcher. For what? Nobody knows.', 'Coach Toby wants everybody in the dugout to drop and give him twenty.', 'Toby\'s got a clipboard up there. He\'s grading his own swing.'],
    hit: ['Toby\'s yelling "HUSTLE" at himself while he runs. Inspiring.', 'Toby gives himself an A-plus for that one.'],
    k: ['Toby just gave himself detention.', 'That\'s a teachable moment for Toby. He\'s teaching it to himself.'],
    walk: ['Toby is jogging to first with his knees up. High knees! HIGH KNEES!'],
    pitch: ['Toby\'s pitching. He blew the whistle before the pitch. That\'s not a rule.'],
    field: ['Toby blows the whistle on that one. You\'re OUT, mister!', 'Toby made the play, then made everybody do jumping jacks.'],
  },
  wren: {
    up: ['Birdie spotted a cardinal on the way up to bat. That\'s 213!', 'Wren\'s got binoculars in her back pocket. Very professional.', 'Birdie\'s whispering so she doesn\'t scare the pitcher.'],
    hit: ['Birdie\'s on base! She\'s writing it in her bird journal.', 'Wren is on first, and she just pointed at a robin. Focus, Birdie!'],
    hr: ['Birdie hit one out, and she watched it fly like it was a hawk.'],
    k: ['Birdie was looking at a blue jay. Strike three went right by.'],
    field: ['Birdie FLEW for that one! Get it? Birdie? I\'m here all week.', 'Spring Sneakers! Wren got more air than a blue jay.', 'Wren tracked that ball like a rare warbler.'],
  },
  dez: {
    up: ['Dez is up. Everybody lower your voices. Smoooooth.', 'You\'re listening to Dez FM, and Dez FM is at the plate.', 'Dez says this at-bat goes out to the ladies. The ladies are his grandma.'],
    hit: ['That was smooth like a saxophone solo, baby.', 'Dez FM, playing all the hits. Including that one.'],
    hr: ['Dez is walking the bases. Not jogging. Walking. Smooth jazz pace.'],
    k: ['Dez keeps it mellow. Even after striking out. Respect.'],
    pitch: ['Dez on the mound. Keepin\' it mellow, baby.', 'Dez pitches like it\'s two in the morning on the radio.'],
  },
  priya: {
    up: ['Pree\'s ledger is in the dugout. She already pre-recorded this at-bat.', 'Priya calculated her odds before walking up. She won\'t tell me. Confidential.', 'Pree wears reading glasses at the plate. She reads the pitches.'],
    hit: ['Priya books that hit as an asset. Double-entry!', 'The numbers don\'t lie. That was a hit.'],
    k: ['Pree\'s writing that strikeout off. Non-deductible, though.'],
    walk: ['Priya takes the walk. She says it was "the fiscally responsible move."'],
    pitch: ['Priya on the mound. Every pitch is itemized.'],
    error: ['Pree made an error. She\'s marking it in her own ledger. In red pen.'],
  },
  gus: {
    up: ['Gus-Gus is wearing flannel. In July. A true professional.', 'That beard is a cut-up sponge, folks. Do NOT tell Gus.', 'Gus-Gus says he ate fastballs for breakfast. Also waffles.'],
    hit: ['TIMBERRR! Gus-Gus chopped that one!', 'Gus split that one like firewood.'],
    hr: ['Gus-Gus yelled TIMBER before it even landed. Confident.'],
    field: ['Gus-Gus threw that like he was chucking a log!'],
    pitch: ['Gus-Gus on the hill. He\'s gonna chop some wood out there.'],
  },
  molly: {
    up: ['Pickles just took the umpire\'s lunch order. There\'s no umpire.', 'Molly\'s got a pencil behind her ear. Pickle on the side, presumably.', 'Number forty-two! Now serving: Pickles, at the plate.'],
    hit: ['Pickles delivers! Unlike her sandwiches.', 'Order up! One base hit, toasted.'],
    field: ['Order up! Pickles serves that one right into her glove.', 'Flypaper Glove! Nothing gets past the deli counter.'],
    pitch: ['Pickles is pitching. You want that knuckleball toasted?'],
    walk: ['Pickles takes first base and asks if anybody wants mustard.'],
  },
  jun: {
    up: ['Jun\'s selling his bat to the catcher. Three easy payments!', 'Jumpin\' Jun says if you call now, you get a SECOND at-bat. You don\'t.', 'Jun tried to sell me a blender before the game. I almost bought it.'],
    hit: ['But wait, there\'s MORE! Jun\'s on base!'],
    hr: ['Jun is selling souvenir home-run balls. He doesn\'t have the ball.'],
    k: ['But wait... there\'s NO more. Jun goes down.', 'Operators are standing by to take Jun\'s complaints.'],
    pitch: ['Jun\'s on the mound. But wait, there\'s more!', 'Jun says this next pitch is "not available in stores."'],
  },
  kai: {
    up: ['It\'s Kaboom, at his own house! Home-yard advantage is real.', 'Mrs. Mendoza just yelled "aim left, sweetie." Kai did not hear. Nobody hears over Kai.', 'Kaboom announced his own at-bat. SUNDAY SUNDAY SUNDAY.'],
    hit: ['Mr. Mendoza just waved his spatula! That\'s his boy!', 'KA-BOOM! Kai\'s yelling it himself, so I don\'t have to.'],
    hr: ['Kaboom homers in his own backyard! Mrs. Mendoza is checking the windows.', 'Kai is announcing his own home run trot. Over the announcers. Rude.'],
    k: ['Kai struck out. He announced it anyway. "STRUCK OUT! SUNDAY!"'],
    pitch: ['Kaboom on the mound. Cover your ears, folks.', 'Kai\'s pitching at his own house. He knows every bump in this lawn.'],
  },
  ruby: {
    up: ['Rooster just gave the traffic report. Light on Cedar Lane.', 'Ruby says it\'s a BEAUTIFUL day for an at-bat. She\'s been up since 5.', 'Good morning from Ruby! It\'s four in the afternoon.'],
    hit: ['Rooster\'s on base! Back to you, Chet. Thanks, Ruby!', 'Ruby\'s doing the weather from first base. Sunny!'],
    k: ['Ruby struck out and said "and THAT\'S the news!" Very upbeat about it.'],
    field: ['Rooster! Up early and catching everything.'],
  },
  ezra: {
    up: ['Easy E sold the catcher foul-tip insurance on the way up. Still not real.', 'Ezra\'s briefcase is in the dugout. It\'s empty. Business.', 'Easy E reminds everyone that line drives are NOT covered.'],
    hit: ['Ezra\'s safe. And insured.'],
    k: ['Ezra\'s filing a claim on that strikeout. Denied.', 'Easy E says strike three was "an act of nature."'],
    field: ['Easy E has that play fully covered. Low deductible.'],
    error: ['Easy E says that error is covered under his policy. It is not.'],
  },
  maya: {
    up: ['Zoom Zoom! Fastest kid on Cedar Lane. She delivered the mail on the way to bat.', 'Maya\'s got a fanny pack full of stamps. Special delivery coming.', 'Neither rain nor sleet stops Zoom Zoom. A curveball might.'],
    hit: ['Special delivery! Zoom Zoom is on base, no signature required.', 'Zoom Zoom left a slip at first base: "Sorry I missed you!"'],
    walk: ['Maya takes the walk and makes it look like a sprint.'],
    field: ['Zoom Zoom ran that down like a late package.'],
  },
  leo: {
    up: ['Lefty Leo is not French, folks. He\'s from Cedar Lane.', 'Leo calls every pitch "un soufflé." Every single one.', 'Lefty Leo brought a rolling pin to bat. He\'s been asked to use the bat.'],
    hit: ['Magnifique! Leo cooked that one!', 'Leo says that hit needed more butter. It was perfect.'],
    hr: ['Leo kissed his fingers rounding second. Chef\'s kiss. On the basepaths.'],
    pitch: ['Leo on the mound, serving up soufflés.', 'Lefty Leo says zis pitch is "fully baked."'],
  },
  anya: {
    up: ['Snowball just forecast a home run. Her forecasts are right ten percent of the time.', 'Anya says there\'s a cold front moving through the batter\'s box.', 'Snowball\'s five-day forecast: sunny with a chance of dingers.'],
    hit: ['Anya called that! Wait, she called snow. Close enough.'],
    hr: ['Snowball forecast a home run and GOT one. Her first correct forecast all summer.'],
    field: ['Rocket Arm from Snowball! That throw had a wind chill.'],
    pitch: ['Anya\'s pitching. Ninety percent chance of strikes. So, ten percent.'],
  },
  darius: {
    up: ['D-King has objected to the pitcher. On what grounds? "Vibes."', 'Darius handed the catcher his business card. Attorney at Lawn.', 'D-King has filed a motion to make the strike zone smaller.'],
    hit: ['D-King rests his case. On first base.'],
    k: ['Darius would like to appeal that strikeout. To his mom.', 'OBJECTION! Overruled. Darius goes back to the dugout.'],
    walk: ['Darius says the walk proves his client was innocent. He is his client.'],
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
    hr: ['Tank is rounding the bases at a gentle mall-cop stroll. Excuse me, pardon me.'],
    k: ['Tank strikes out, and he apologizes to the catcher. Such a nice boy.'],
    error: ['Hank dropped it and said "sorry" to the ball. Gentle giant.'],
  },
};

/** Batter vs pitcher: these two have history. Key: "batter|pitcher". */
const MATCHUP: Record<string, string[]> = {
  'hank|pepper': ['Pepper sold Hank\'s bike last week. Hank would like it back. Pepper would like to sell it again.'],
  'pepper|hank': ['Pepper versus Hank. She sold his bike, folks. He\'s still being very polite about it.'],
  'ruby|anya': ['The morning show versus the weather lady. This is a ratings war.'],
  'anya|ruby': ['Snowball against Rooster. Two TV personalities. Only one microphone.'],
  'darius|toby': ['D-King has objected to Toby\'s whistle. Toby has blown the whistle at the objection.'],
  'toby|darius': ['Toby versus D-King. Gym class versus the courtroom.'],
  'priya|ezra': ['The CPA against the insurance man. Somebody\'s getting audited.'],
  'ezra|priya': ['Easy E tried to sell Pree a policy. She read the fine print. Out loud.'],
  'leo|molly': ['The fancy chef against the deli counter. Soufflé versus pickle. Huge.'],
  'molly|leo': ['Pickles versus Lefty Leo. Leo says her sandwiches lack "finesse." Them\'s fighting words.'],
  'jun|pepper': ['Two salespeople. One at-bat. Somebody\'s going home with a blender.'],
  'pepper|jun': ['Pepper against Jun. Fastest mouths on two streets.'],
  'dez|kai': ['The quietest kid in the league against the loudest. Dez is whispering. Kai is NOT.'],
  'kai|dez': ['Kaboom against Dez. Volume eleven against volume two.'],
  'bo|hank': ['Two big guys. Bo\'s knees versus Hank\'s good manners.'],
  'hank|bo': ['Tank against Mudpie. The heaviest at-bat in league history. Physically.'],
  'kai|jun': ['Kai versus Jun: two kids who announce their own pitches. It\'s loud out there.'],
  'wren|anya': ['Birdie versus Snowball. Wren says Anya\'s forecasts scare the birds.'],
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
  'Mr. Mendoza just tested a burger. For quality. He\'s testing another one.',
];
const SPONSOR = [
  'This inning is brought to you by... nobody. We\'re still looking for sponsors.',
  'Our sponsor: Chet\'s Lemonade Stand. Twenty-five cents. Ice is extra.',
  'Sponsored by Dottie\'s brother\'s lawn-mowing business. Call now! Ask for Doug!',
  'A word from our sponsors: please return Mrs. Mendoza\'s lawn chairs.',
  'Brought to you by the ice cream truck. We can hear it! It\'s two streets over!',
  'You\'re watching Channel 4½, Maple Hollow Public Access. Don\'t touch that dial.',
  'This broadcast is coming to you on my dad\'s camcorder. Hi, Dad. Sorry about the tape.',
  'Our producer, my little sister Bea, says we have to mention the bake sale. Saturday. Bring quarters.',
];
const TEEBALL = [
  'Back in my big-league days, I batted ninth. Ninth is the cleanup spot for winners.',
  'I played one whole season of tee-ball. Some people call that a career.',
  'My tee-ball coach said I had a "unique stance." I think he meant amazing.',
  'People ask if I miss the big leagues. Every day, Chet. Every day.',
  'In tee-ball the ball doesn\'t even move. Talk about pressure.',
  'I still have my tee-ball trophy. Everybody got one. Mine\'s shinier.',
  'My tee-ball team was the Fighting Ladybugs. We fought. Mostly each other.',
];
const POOL = [
  'Reminder, folks: the pool is not in play. Unless it is. Then it\'s a double.',
  'The flamingo floatie has been watching this whole game. Very focused.',
  'Mrs. Mendoza just reminded everybody to aim left. Again.',
  'Somebody left their goggles on the diving board. Not it.',
  'The garden gnome by home plate has not moved all game. Great discipline.',
  'That right-field picket fence is short. Suspiciously short. I\'m just saying.',
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
  ['Bea is holding up a cue card. It says... "snack time."', 'Bea is eating the cue card, Chet.'],
  ['I want to say hi to my mom, who is watching at home.', 'Your mom is right there, Chet. In the lawn chair. Waving.'],
];

export class Booth {
  private rng: Rng;
  /** every line said this game, as shapes (names and numbers out): none gets said again */
  private used = new Set<string>();
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
  /** how each kid's last trip to the plate went */
  private lastTrip = new Map<string, 'k' | 'hit' | 'hr' | 'walk' | 'out'>();
  /** batters in a row the current pitcher has retired */
  private retired = 0;
  private retiredBy = '';
  private jinxed = false;
  private homers = new Map<string, number>();

  constructor(seed: number) {
    this.rng = new Rng(seed ^ 0xb00b);
  }

  /** A fresh line from a pool, or null when the pool has nothing left that hasn't been said. */
  private pick(_key: string, opts: string[]): string | null {
    const fresh = opts.filter((o) => !this.used.has(shape(o)));
    if (!fresh.length) return null;
    return this.rng.pick(fresh);
  }

  private chance(p: number) { return this.rng.chance(p); }

  /** Remembers what was said so nothing is said twice. */
  private done(lines: Line[]): Line[] {
    const out: Line[] = [];
    for (const l of lines) {
      const sh = shape(l.text);
      if (this.used.has(sh)) continue;
      this.used.add(sh);
      out.push(l);
    }
    return out;
  }

  /** Returns 0–2 lines for an event (most events say nothing). */
  react(e: MatchEvent, m: Match): Line[] {
    for (const s of [0, 1] as const) this.maxDeficit[s] = Math.max(this.maxDeficit[s], m.score[1 - s] - m.score[s]);
    this.track(e, m);
    const out = this.done(this.lines(e, m).filter((l) => !!l.text));
    if (e.type !== 'run') this.walked = e.type === 'walk';
    this.prevType = e.type;
    return out.slice(0, 2);
  }

  /** Keep the little bits of memory the lines lean on. */
  private track(e: MatchEvent, m: Match): void {
    const p = m.pitcher.id;
    if (p !== this.retiredBy) { this.retiredBy = p; this.retired = 0; }
    switch (e.type) {
      case 'strikeout': this.lastTrip.set(e.batter, 'k'); this.retired++; break;
      case 'hit': this.lastTrip.set(e.batter, 'hit'); this.retired = 0; break;
      case 'homeRun': this.lastTrip.set(e.batter, 'hr'); this.retired = 0; this.homers.set(e.batter, (this.homers.get(e.batter) ?? 0) + 1); break;
      case 'groundRule': this.lastTrip.set(e.batter, 'hit'); this.retired = 0; break;
      case 'walk': this.lastTrip.set(e.batter, 'walk'); this.retired = 0; break;
      case 'error': this.retired = 0; break;
      case 'out': if (e.runner === m.batter.id || e.kind === 'fly') { this.lastTrip.set(e.runner, 'out'); this.retired++; } break;
      default: break;
    }
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

  /** A kid's own lines that haven't been said yet. */
  private flavor(id: string, part: keyof Flavor): string[] {
    return (FLAVOR[id]?.[part] ?? []).filter((s) => !this.used.has(shape(s)));
  }

  private C(text: string | null): Line { return { who: 'Chet', text: text ?? '' }; }
  private D(text: string | null): Line { return { who: 'Dottie', text: text ?? '' }; }

  /** A running gag for an idle moment. */
  private gag(m: Match): Line {
    const pool = this.isPoolParty(m);
    const r = this.rng.next();
    if (pool && r < 0.3) {
      while (this.grill < GRILL.length && this.used.has(shape(GRILL[this.grill]))) this.grill++;
      if (this.grill < GRILL.length) return this.D(GRILL[this.grill++]);
    }
    if (r < 0.55) return this.C(this.pick('sponsor', SPONSOR));
    if (pool && r < 0.7) return this.D(this.pick('pool', POOL));
    return this.D(this.pick('teeball', TEEBALL));
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
      case 'bobble': {
        if (!this.chance(0.4)) return [];
        const f = nick(e.fielder);
        return [this.C(this.pick('bobble', [
          `${f} juggles it! Like a hot potato fresh off Mr. Mendoza's grill.`, `Bobbled by ${f}! It's in there, it's out, it's in!`,
          `${f} fumbles it like a wet bar of soap.`, `${f} double-clutches it! Hold on to it, ${kid(e.fielder).first}!`,
          `Uh-oh, ${f} can't find the handle!`,
        ]))];
      }
      case 'error': return this.error(e.fielder);
      case 'throw': return this.throwLine(e.fielder, e.base);
      case 'out': return this.outLine(e.kind, e.runner, e.fielder);
      case 'run': return this.runLine(e.runner, m);
      case 'dog':
        return [this.C(this.pick('dog', [
          'The dog is involved now! THE DOG IS INVOLVED!', 'It hit the doghouse! Biscuit is NOT happy.',
          'Off the doghouse! Somebody check on Biscuit.', 'Biscuit just barked at the baseball. Biscuit is a critic.',
        ]))];
      case 'tree':
        return this.chance(0.6) ? [this.C(this.pick('tree', [
          'Into the big tree! Leaves everywhere!', 'It\'s up in the branches!', 'Off the tree! A squirrel just filed a complaint.',
          'The tree gets a piece of it! Leaves are raining down on left field!', 'Into the leaves! It\'s like a pinball machine in there.',
        ]))] : [];
      case 'fence': {
        if (e.cleared || !this.chance(0.45)) return [];
        if (e.kind === 'house') return [this.C(this.pick('fence-house', [
          'Off the back of the house! Mrs. Mendoza is at the window!', 'It bangs off the siding! Everybody look innocent.',
          'Off the house! That\'s the kitchen window it just missed. Barely.',
        ]))];
        if (e.kind === 'hedge') return [this.C(this.pick('fence-hedge', [
          'Into the hedge! The hedge eats it.', 'The hedge in left swallows that one. It does that.',
          'Off the hedge! Somebody\'s going in there after it. Not me.',
        ]))];
        return [this.C(this.pick(`fence-${e.kind}`, [
          'Off the pickets! Rattles every slat on the way down.', 'It clacks off the picket fence! Inches from a home run!',
          'Off the short fence in right! The suspiciously short fence!', 'Off the fence! Just short!',
        ]))];
      }
      case 'quip': {
        if (!this.chance(0.12)) return [];
        const n = nick(e.kid);
        return [this.D(fit(this.pick('quip', [
          `Did ${n} just say "${e.text}" Iconic.`, `"${e.text}" Words to live by, ${n}.`,
          `${n} says "${e.text}" Truly a poet.`, `Somebody write that down. "${e.text}"`,
          `Bea, put "${e.text}" on a cue card.`,
        ]) ?? '', `${n} has a way with words.`))];
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
    if (this.batters > 2 && this.chance(0.07)) {
      const c = this.pick('riff', RIFFS.map((r) => r[0]));
      if (c) return [this.C(c), this.D(RIFFS.find((r) => r[0] === c)![1])];
    }
    const side = m.battingSide;
    const team = m.side(side).team;
    const p = m.pitcher;
    const box = m.box[k.id]?.bat;
    const hits = box?.h ?? 0, ab = box?.ab ?? 0;
    // how they've done today, now and then, when it says something
    const today = ab >= 2 && hits === 0 && this.chance(0.3) ? ` Still looking for a hit today.`
      : ab >= 1 && hits >= 1 && this.chance(0.35) ? ` ${hits} for ${ab} so far.` : '';
    const pos = this.positionOf(m, k.id);
    const n = k.nick;
    const lead = m.score[side] - m.score[1 - side];
    const lastInning = m.inning >= m.cfg.innings;
    const [b1, b2, b3] = m.bases;
    const last = this.lastTrip.get(k.id);

    let intro: string | null = null;
    if (this.chance(0.65)) {
      if (b1 && b2 && b3) intro = this.pick('up-loaded', [
        `Bases loaded for ${n}! Everybody's mom is standing up.`,
        `Ducks on the pond, all three of 'em, and here's ${n}.`,
        `The bases are juiced! ${n} could be a hero right here.`,
        `Bases loaded, and ${n} steps in. I can't look. I'm looking.`,
        `Bases loaded. Even Mr. Mendoza put the spatula down for ${n}.`,
      ]);
      else if (m.outs === 2 && (b2 || b3)) intro = this.pick('up-clutch', [
        `Two outs, runner in scoring position. Big spot for ${n}.`,
        `${n} up with two down and a runner who'd really like to come home.`,
        `Two outs. ${n} at the plate. This is what you practice for in the driveway.`,
        `Two down, ${b3 ? nick(b3) : nick(b2!)} in scoring position, and ${n} at the plate. Bea, hold the snacks.`,
      ]);
      else if (lastInning && lead < 0 && lead >= -3) intro = this.pick('up-late', [
        `Last licks for the ${team.name}! ${n} needs to get something going.`,
        `It's now or never for the ${team.name}. Here's ${n}.`,
        `Down ${num(-lead)} in the last inning. ${n} steps in. No pressure, kid.`,
        `The streetlights will be on soon. The ${team.name} need ${n} to start something.`,
      ]);
      else if (last === 'k' && this.chance(0.6)) intro = this.pick('up-after-k', [
        `${n} struck out last time up. Looking for a little payback.`,
        `${n} is back after that strikeout. ${p.nick} remembers. ${n} remembers.`,
        `Round two: ${n} versus ${p.nick}. Last time, ${p.nick} won.`,
      ]);
      else if (last === 'hr' && this.chance(0.8)) intro = this.pick('up-after-hr', [
        `${n} hit one out of here last time. ${p.nick} would like that not to happen again.`,
        `Here's ${n}, who homered last time up. The outfielders are backing up. Way up.`,
        `${n} again. Last time, that ball ended up in the neighbors' yard.`,
      ]);
      else if (hits >= 2 && this.chance(0.7)) intro = this.pick('up-hot', [
        `${n} is ${hits} for ${ab} today. Hottest bat in the yard.`,
        `${n} has ${num(hits)} hits already. Somebody check that bat for batteries.`,
        `${n} is on fire today, ${hits} for ${ab}. Not literally. We checked.`,
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
        `${n} kicks the dirt, taps the plate, adjusts the helmet. The full routine.`,
        `Up next, wearing number ${k.age}... wait, that's ${poss(n)} age. Anyway, it's ${n}.`,
        `${n} to the plate. ${p.nick} looks in.`,
        `${n} steps up and points the bat at the hedge. Calling it? Bold.`,
        `Here's ${n}. Mom's in the front row. The front row is a lawn chair.`,
        `${n} is up. Somebody on the ${team.name} bench is chanting. It's ${poss(n)} little brother.`,
        `And now, presenting ${persona(k)} ${k.first} ${k.last}!`,
        `${n} walks up swinging two bats. Drops one. Picks it up. Very smooth.`,
        `Next up for the ${team.name}, ${k.first} ${k.last}. Friends call ${k.first} "${n}."`,
      ]);
    }
    const out: Line[] = [];
    if (intro) out.push(this.C(fit(intro + today, intro)));

    // the no-hitter jinx: Chet can't help himself
    const fs = m.fieldingSide;
    if (!this.jinxed && m.hits[side] === 0 && m.inning >= Math.max(3, Math.ceil(m.cfg.innings / 2)) && m.outs === 0 && this.chance(0.6)) {
      this.jinxed = true;
      out.push(this.C(`Folks, the ${m.side(side).team.name} don't have a hit yet. That's called a no-hit...`));
      out.push(this.D('Don\'t say it, Chet. You\'ll jinx it. Everybody knows that.'));
      return out;
    }
    if (this.retired >= 6 && this.chance(0.5)) {
      const r = this.pick('retired', [
        `${p.nick} has set down ${num(this.retired)} in a row. ${persona(p)} is cruising.`,
        `That's ${num(this.retired)} straight for ${p.nick}. The ${m.side(fs).team.name} are on a roll.`,
      ]);
      if (r) { out.push(this.D(r)); return out; }
    }

    const matchup = MATCHUP[`${k.id}|${p.id}`]?.filter((s) => !this.used.has(shape(s))) ?? [];
    if (matchup.length && this.chance(0.8)) {
      out.push(this.D(this.rng.pick(matchup)));
    } else if (this.chance(0.42)) {
      const fresh = !this.introduced.has(k.id);
      const flavor = this.flavor(k.id, 'up');
      if (fresh && this.chance(0.6)) {
        this.introduced.add(k.id);
        // the bios are notes on a cue card ("Argues every call."), so they need the name in front
        // unless Chet just said it
        const named = fit(`${n}? ${k.bio}`, fit(`${n}: ${k.bio}`, k.bio, 140));
        out.push(this.D(this.pick('bio', out.length && this.chance(0.4) ? [k.bio] : [named, fit(`About ${n}. ${k.bio}`, named)])));
      } else if (flavor.length && this.chance(0.6)) {
        out.push(this.D(this.rng.pick(flavor)));
      } else if (this.chance(0.5)) {
        out.push(this.D(this.pick('matchup', [
          `${p.nick} against ${n}. ${p.persona} versus ${persona(k)}. Classic.`,
          `I give ${n} a fifty-fifty chance. Maybe sixty-forty. I'm not great at math.`,
          `If I were ${p.nick}, I'd throw ${n} a curveball. I'd also be taller.`,
          `${n} has that look today. The look of a kid who had pancakes.`,
          `Eye on the ball, ${n}! That's what my tee-ball coach told me. A big leaguer.`,
          `${poss(n)} stance is a lot like mine was. Mine was better.`,
          `${p.nick} better be careful with ${n}. Trust me. I've seen things.`,
          `${n} is choking up on the bat. Smart. I did that. In the bigs.`,
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
    if (flavor.length && this.chance(0.3)) return [this.D(this.rng.pick(flavor))];
    const byType: Record<string, string[]> = {
      fastball: [`${n} brings the heat! ${mph} on the radar gun!`, `Fastball, ${mph}! That's faster than my bike.`, `Here's the cheese from ${n}!`, `${n} just reared back and threw it. Hard. ${mph}.`, `${mph} miles an hour from ${n}. The radar gun is my cousin's, so, roughly.`],
      curve: [`${n} spins a curveball. It bends like a garden hose!`, `Curveball from ${n}! That one took the scenic route.`, `${n} drops in the curve. Loopy!`, `${n} throws a curve that goes around the gnome. Almost.`],
      changeup: [`Changeup from ${n}. Slow as a Sunday afternoon.`, `Oh, the changeup from ${n}! Sneaky, sneaky.`, `${n} pulls the string on a changeup. ${mph} miles an hour.`],
      slider: [`${n} with the slider. It slides right on by.`, `Slider from ${n}! Sideways like a crab.`],
      sinker: [`Sinker from ${n}. That one dove like a kid off the diving board.`, `${n} throws the sinker. Down it goes!`],
      knuckler: [`Knuckleball! Nobody knows where that's going. Including ${n}.`, `Here comes ${poss(n)} knuckler, wobbling like a jelly donut.`],
    };
    const opts = byType[e.pitch] ?? [`${n} deals. ${mph} on the gun.`];
    const out = [this.C(this.pick(`pitch-${e.pitch}`, opts))];
    if (this.chance(0.15)) out.push(this.D(this.pick('pitch-d', [
      'In tee-ball we didn\'t have pitches. We had a tee. Simpler times.',
      `${poss(n)} got good stuff today. Not as good as mine was. But good.`,
      'The radar gun is my cousin\'s. He says it\'s "pretty accurate."',
      'You can hear that one pop the mitt from up here. And up here is a lawn chair.',
    ])));
    return out;
  }

  private special(id: string, special: keyof typeof SPECIAL_INFO, m: Match): Line[] {
    const info = SPECIAL_INFO[special];
    const n = nick(id), L = info.label.toUpperCase(), l = info.label;
    const kindLines = info.kind === 'bat'
      ? [`${n} is loading up the ${l}! Outfielders, back up! Back up more!`, `${n} is digging in for the ${L}!`]
      : info.kind === 'pitch'
        ? [`${n} winds up for the ${L}! Batter, good luck!`, `${n} has the ${l} ready. Uh-oh.`]
        : [`${n} is turning on the ${l}!`];
    const out = [this.C(this.pick(`special-${info.kind}`, [
      `${n} is going for the ${L}!`, `Here it comes: the ${L}! ${n} is powering up!`,
      `${n} has that look. It's ${L} time!`, ...kindLines,
    ]))];
    if (this.chance(0.5)) out.push(this.D(this.pick('special-d', [
      'Oh, this is gonna be good.', 'I invented that move, by the way.', 'Somebody get the camcorder! Oh, it\'s on. Hi, Dad.',
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
      return [this.C(this.pick('foul', [
        `Fouled back. Somebody check the windows.`, `${b} fights it off. Foul.`, `${b} stays alive with a foul.`,
        ...(this.isPoolParty(m) ? [
          'Foul, into the flower bed. Mrs. Mendoza won\'t love that.', 'That foul went toward the grill! Mr. Mendoza didn\'t even flinch.',
          'Foul, off the gutter! That\'s a new sound.', 'Fouled into the lawn chairs. Everybody duck. Everybody\'s fine.',
          `Foul ball past the garden gnome. The gnome did not react.`,
        ] : ['Foul, into the bushes.']),
        `${b} just gets a piece of it. Foul.`,
        s === 2 ? `${b} fouls off another one with two strikes. Pesky!` : `${bl} and ${s} after the foul.`,
      ]))];
    }
    if (bl === 3 && s === 2 && this.chance(0.5)) return [this.C(this.pick('full', [
      'Full count! Everybody hold your breath.', `Three and two. ${p} and ${b}, eyeball to eyeball.`,
      'Full count! Mrs. Mendoza just covered her eyes.', 'Payoff pitch coming up. This is the good part.',
      `Three balls, two strikes, and ${b} is chewing on the batting glove.`,
    ]))];
    if (!this.chance(0.1)) return [];
    if (call === 'ball') return [this.C(this.pick('ball', [
      `Outside. Ball ${bl}.`, `${p} misses high.`, `Ball ${bl}. ${b} wasn't biting.`,
      `That's ball ${bl}. ${p} is shaking it off.`, `Low. ${b} lays off it.`, `${p} bounces one. Ball ${bl}.`,
      `Ball ${bl}. That one nearly hit the gnome.`,
    ]))];
    if (s === 2) return [this.C(this.pick('two-strikes', [
      `${b} is down to the last strike.`, `${bl} and 2. ${p} has ${b} right where ${p} wants.`,
      `Two strikes on ${b}. Choke up, kid!`, `${b} is in a hole now. ${bl} and 2.`,
    ]))];
    if (call === 'strike') return [this.C(this.pick('strike', [
      `Strike ${s}, looking.`, `${b} watches that one go by. Strike ${s}.`, `Right down the middle from ${p}. Strike!`, `${p} paints the corner. Strike ${s}.`,
    ]))];
    return [this.C(this.pick('swinging', [
      'Swing and a miss!', `${b} cuts and misses. Strike ${s}.`, `Whiff! ${b} made a nice breeze, though.`, `${b} swung so hard the helmet came off!`,
    ]))];
  }

  private strikeout(id: string, looking: boolean, m: Match): Line[] {
    const n = nick(id), p = m.pitcher, pn = p.nick;
    const out = [this.C(looking
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
        `Got 'em! ${pn} gets ${n} swinging.`, `${pn} blows it past ${n}! Strike three!`,
      ]))];
    if (this.chance(0.42)) {
      const so = m.box[p.id]?.pitch.so ?? 0;
      const opts = [
        `${pn} is dealing today.`, 'In my day we called that "the windmill."', 'That pitch had some MUSTARD on it.',
        'Shake it off, kid. Shake it off.', 'Even I would\'ve struck out on that. And I played in the bigs. Tee-ball bigs.',
        'That ball moved like an ice cream truck going the wrong way.', 'Back to the dugout. A juice box will make it better.',
        ...this.flavor(id, 'k'), ...this.flavor(p.id, 'pitch'),
      ];
      out.push(so >= 4 && this.chance(0.5)
        ? this.D(this.pick('k-count', [`That's ${num(so)} strikeouts for ${pn} today!`, `${pn} has ${so} K's. I'm drawing them on my hand.`, `Strikeout number ${so} for ${pn}. Somebody check that arm for batteries.`]))
        : this.D(this.pick('k-d', opts)));
    }
    return out;
  }

  private walk(id: string, hbp: boolean, m: Match): Line[] {
    const n = nick(id), p = m.pitcher.nick;
    const loaded = m.bases.every(Boolean);
    const out = [this.C(hbp
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
    if (this.chance(0.3)) out.push(this.D(hbp
      ? this.pick('hbp-d', ['That\'s gonna leave a mark. A cool mark, though.', 'Rub some dirt on it! Actually, don\'t. The lawn was just reseeded.', `${p} says sorry. ${p} better mean it.`])
      : this.pick('walk-d', [
        'A walk\'s as good as a hit. That\'s what my tee-ball coach said. We didn\'t have walks.',
        'Free base! Like a free sample at the grocery store.',
        `${p} needs to find the strike zone. It's the rectangle, buddy.`,
        'Good eye! I had good eyes too. Still do. Twenty-twenty. Look it up.',
        ...this.flavor(id, 'walk'),
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
    if (now > 0 && before <= 0) return { runs, lead: true, line: this.pick('go-ahead', [`And the ${team} take the lead! ${sc}.`, `That puts the ${team} on top! ${sc}.`, `Lead change! It's ${sc}.`, `The ${team} go ahead, ${sc}! Somebody tell Mr. Mendoza!`]) };
    if (now === 0) return { runs, lead: true, line: this.pick('tied', [`Tie ballgame! ${sc}.`, `We're all square! ${sc}.`, `And this one is TIED. ${sc}.`]) };
    return { runs, lead: false, line: this.pick('rbi', [
      runs === 1 ? `A run scores! ${sc}.` : `${runs} runs score! ${sc}.`,
      runs === 1 ? `Here comes the run! ${sc}.` : `${runs} across! ${sc}.`,
      `That's ${runs === 1 ? 'an RBI' : `${runs} RBIs`}! ${sc}.`,
      `The ${team} add on. ${sc}.`,
    ]) };
  }

  private hit(id: string, bases: number, m: Match): Line[] {
    const k = kid(id), n = k.nick;
    const kind = bases >= 3 ? 'triple' : bases === 2 ? 'double' : 'single';
    const la = this.lastContact?.la ?? 15;
    const spray = this.lastContact?.spray ?? 0;
    const shape = la < 8 ? 'ground' : la > 28 ? 'bloop' : 'liner';
    const field = spray > 15 ? 'right' : spray < -15 ? 'left' : 'center';
    const pool: Record<string, string[]> = {
      single: [
        `${n} pokes one into ${field}!`, `Single for ${n}. Nothing fancy, gets the job done.`,
        `${n} slaps it into ${field} field. Base hit!`, `That's a knock! ${n} is aboard.`, `${n} finds a hole! Single.`,
        `${n} punches one through. ${n} on first, looking very pleased.`, `Base hit, ${n}! The bench is banging on the cooler.`,
        `${n} drops one in front of the ${field} fielder. That'll play.`, `There's a hit for ${n}, and a high five from the first-base kid.`,
        ...(shape === 'ground' ? [`Seeing-eye grounder gets through! ${n} on first.`, `${n} chops it past the infield!`, `A worm-burner from ${n}, and it squirts through!`]
          : shape === 'bloop' ? [`Little bloop... it falls in! ${n} has a single.`, `${n} dunks one in! A duck snort!`, `${n} plops one between three fielders. Nobody called it!`]
            : [`Line drive single for ${n}!`, `${n} ropes one up the middle!`, `${n} lines it right over the shortstop's glove!`]),
      ],
      double: [
        `${n} lines one into the gap. That's a double!`, `Two-bagger for ${n}!`, `${n} cruises into second with a double!`,
        `Into the corner in ${field}! ${n} coasts into second.`, `${n} splits the outfielders! Stand-up double.`,
        `Double for ${n}! Somebody's buying the ice pops.`,
      ],
      triple: [
        `A TRIPLE for ${n}! Nobody hits triples!`, `${n} is flying around the bases! TRIPLE!`,
        `Three bases for ${n}! Somebody get that kid a water.`, `${n} slides into third! A triple!`,
        `The rarest hit in the backyard: a triple, by ${n}!`,
      ],
    };
    const out = [this.C(this.pick(`hit-${kind}`, pool[kind]))];
    const story = this.runsStory(m);
    const h = m.box[id]?.bat.h ?? 0;
    if (story.line && (story.runs > 1 || this.chance(0.7))) out.push(this.C(story.line));
    else if (h >= 3 && this.chance(0.6)) {
      out.push(this.D(this.pick('hit-streak', [`That's hit number ${num(h)} for ${n} today!`, `${n} is ${h} for ${m.box[id].bat.ab}. Hand that kid a trophy. I'll give mine.`])));
    } else if (m.hits[m.battingSide] === 1 && this.jinxed) {
      out.push(this.D(this.pick('jinx-broken', ['There goes the no-hitter. Told you not to say it, Chet.', 'Chet jinxed it. On camera. Bea, keep that tape.'])));
    } else if (this.chance(bases >= 2 ? 0.45 : 0.25)) {
      out.push(this.D(this.pick('hit-d', [
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
    const pool = this.isPoolParty(m);
    const where = !pool ? 'the fence' : spray > 12 ? 'the picket fence' : spray < -12 ? 'the hedge' : 'the fence in center';
    const extra = runs === 2 ? ' Two runs score!' : runs === 3 ? ' Three runs score!' : '';
    const nth = this.homers.get(id) ?? 1;
    const call = runs >= 4
      ? this.pick('slam', [`GRAND SLAM! ${n} clears the bases!`, `It's a GRAND SLAM for ${n}! Four runs!`, `Bases loaded, and ${n} empties them! GRAND SLAM!`, `${n} hits a GRAND SLAM! Everybody touch 'em all!`])
      : nth >= 2 && this.chance(0.7)
        ? this.pick('hr-again', [`${n} does it AGAIN! Home run number ${num(nth)}!`, `ANOTHER ONE! ${n} has ${num(nth)} homers today!`, `${n} goes deep for the ${ordinal(nth)} time! Somebody stop this kid!`])
        : this.pick('hr', [
          `IT'S OUTTA HERE! ${n} crushed it!`, `GOODBYE, BASEBALL! A home run for ${n}!`,
          `Somebody call the neighbors! ${n} hit it into the next zip code!`, `Going... going... GONE! ${n} with a home run!`,
          `Over ${where}! ${n} goes yard! In a yard!`, `${n} got all of that one! HOME RUN!`,
          `See ya! ${n} sends it over ${where}!`, `${n} hits a homer! Touch 'em all, ${k.first}!`,
        ]);
    const out = [this.C(call ? fit(call + extra, call) : null)];
    const story = this.runsStory(m);
    if (story.line && story.lead && this.chance(0.6)) { out.push(this.C(story.line)); return out; }
    const ev = c?.ev ?? 0;
    const far = ev >= 78, barely = ev > 0 && ev < 62;
    const opts = far
      ? ['That ball is still going, folks.', 'That one landed in the street. Somebody check the parked cars.', 'I think that ball is headed for the water tower.', `${n} hit that one to another neighborhood!`, 'The ball just went over a house. A whole other house.']
      : barely
        ? ['Just over! That one scraped the paint off the fence.', 'That barely cleared. I\'ll allow it.', 'Just enough! I could\'ve caught that. In my prime.']
        : ['Somebody\'s gonna have to knock on a door for that one.', 'I hit one like that once. In tee-ball. It was off the tee, but still.', `${n} is gonna be insufferable at lunch tomorrow.`, 'That\'s a souvenir for the neighbors.'];
    if (pool) {
      opts.push('Mr. Mendoza is pointing his spatula at it. That\'s the highest honor.');
      if (spray > 12) opts.push('Over the picket fence in right. Mrs. Mendoza did say aim left. She meant the other way.');
    }
    out.push(this.D(this.pick(`hr-d`, [...opts, ...this.flavor(id, 'hr'), ...this.flavor(id, 'hit')])));
    return out;
  }

  private groundRule(id: string, why: 'splash' | 'bounce'): Line[] {
    const n = nick(id);
    if (why === 'splash') {
      this.splashes++;
      const out = [this.C(this.pick('splash', [
        'SPLASH! It\'s in the water! That\'s a splash double!', 'Into the drink! Splash double!',
        `${n} goes swimming! Splash double!`, 'KER-SPLOOSH! Into the pool for a double!',
        `${n} finds the deep end! Two bases!`, 'Cannonball! That one\'s in the pool. Splash double!',
      ]))];
      out.push(this.D(this.splashes >= 2 && this.chance(0.5)
        ? this.pick('splash-count', [`Splash number ${num(this.splashes)} today! The pool is undefeated.`, `That's ${num(this.splashes)} in the pool today. Somebody get a net.`, `${num(this.splashes)} splashes! Mr. Mendoza has the skimmer out again.`])
        : this.pick('splash-d', [
          'Somebody get the pool skimmer.', 'Hope nobody was swimming.', 'That ball needs a towel.',
          'Mrs. Mendoza did say aim left. Several times.', 'The flamingo floatie never saw it coming.',
          'That ball should\'ve waited thirty minutes after eating.', 'No running by the pool! Oh, it\'s a ball. Carry on.',
          'Mr. Mendoza is fishing it out with the skimmer. Like a pro.',
        ])));
      return out;
    }
    return [this.C(this.pick('bounce', [
      'Bounced over the fence! Ground-rule double.', 'One hop and over! That\'s two bases.',
      'Ground-rule double! It hopped the fence like a squirrel.', `${n} bounces one over. Ground-rule double!`,
    ]))];
  }

  private catch(id: string, fly: boolean, hard: boolean, m: Match): Line[] {
    const n = nick(id);
    const pos = this.positionOf(m, id);
    if (fly && hard) {
      const out = [this.C(this.pick('catch-hard', [
        `WHAT A CATCH by ${n}!`, `${n} lays out and MAKES THE GRAB!`, `Are you kidding me?! ${n} caught it!`,
        `${n} goes all out... and HAS IT!`, `Diving catch! ${n} is covered in grass stains!`,
        `${n} snags it! That's going on the highlight tape!`, `No way! ${n} just robbed that hit!`,
        `${n} leaps... and comes down WITH IT!`, `Back, back... ${n} reaches up and takes it away!`,
        `Full extension from ${n}! I felt that in my knees!`, `${n} with the snow-cone catch! The ball's sticking out of the glove!`,
      ]))];
      if (this.chance(0.6)) out.push(this.D(this.pick('catch-d', [
        'Put that on the refrigerator!', 'Mom, are you filming?! Oh, Chet\'s dad is. Good.', 'I made a catch like that once. It was a juice box, but still.',
        'Grass stains are a badge of honor. My mom disagrees.', `${n} is getting a gold star sticker for that one.`,
        ...this.flavor(id, 'field'),
      ])));
      return out;
    }
    if (fly) {
      if (!this.chance(0.3)) return [];
      return [this.C(this.pick('catch-fly', [
        `${n} squeezes it. Out.`, `Easy catch for ${n}.`, `Can of corn to ${n}.`,
        `${n} camps under it... and makes the catch.`, `High fly ball... ${n} has it.`,
        pos ? `Routine fly. ${n} ${POS_AT[pos]} puts it away.` : `${n} puts it away.`,
        `${n} calls for it, ${n} gets it.`, `Up, up... and down into ${poss(n)} glove.`,
        `${n} loses it in the sun... finds it... catches it! Phew.`,
      ]))];
    }
    if (!this.chance(0.12)) return [];
    return [this.C(this.pick('catch-ground', [
      `${n} scoops it up.`, `${n} gets a glove on it.`, `${n} charges it...`,
      pos ? `${n} ${POS_AT[pos]} picks it up.` : `${n} picks it up.`, `${n} knocks it down!`,
    ]))];
  }

  private error(id: string): Line[] {
    const n = nick(id);
    const out = [this.C(this.pick('error', [
      `Oh no, ${n} drops it!`, `It's off the glove! Error, ${n}!`, `${n} had it... and now doesn't.`,
      `Right through the legs! Error on ${n}!`, `${n} boots it!`, `Oops! ${n} would like that one back.`,
      `The ball squirts away from ${n}! Error!`,
    ]))];
    if (this.chance(0.5)) out.push(this.D(this.pick('error-d', [
      'That\'s going in the blooper reel.', 'Happens to the best of us. Mostly to the worst of us.',
      'Somebody needs a bigger glove.', `Shake it off, ${n}. I once missed a ball sitting on a tee.`,
      'The sun was in their eyes. Or a bee. Let\'s say a bee.', 'Errors build character. I have SO much character.',
      ...this.flavor(id, 'error'),
    ])));
    return out;
  }

  private throwLine(id: string, base: number): Line[] {
    const k = kid(id), n = k.nick, b = BASE[base];
    if (!b) return [];
    if (k.special === 'rocketArm' && this.chance(0.3)) return [this.C(this.pick('throw-rocket', [
      `ROCKET ARM! ${poss(n)} throw has a vapor trail!`, `${n} fires a rocket to ${b}!`, `Did you SEE that throw from ${n}?!`,
    ]))];
    if (!this.chance(0.1)) return [];
    if (base === 0) return [this.C(this.pick('throw-home', [`${n} comes home with it!`, `Here's the throw to the plate from ${n}!`, `${n} with the long throw home!`]))];
    return [this.C(this.pick('throw', [`${n} fires to ${b}!`, `${n} comes up throwing... to ${b}!`, `Here's the throw to ${b} from ${n}!`, `${n} throws to ${b}.`]))];
  }

  private outLine(kind: string, runner: string, fielder: string): Line[] {
    this.playOuts++;
    const r = nick(runner), f = nick(fielder);
    if (this.playOuts === 3) return [this.C(this.pick('tp', ['A TRIPLE PLAY?! I\'ve only seen that in cartoons!', 'TRIPLE PLAY! Somebody pinch me!']))];
    if (this.playOuts === 2) return [this.C(this.pick('dp', [
      'DOUBLE PLAY! Two for the price of one!', `Two outs on one play! ${f} turns it!`, 'Around the horn! It\'s a double play!',
      'Double play! That\'s a twin-killing, folks. That\'s a real term.', `${f} starts the double play! Textbook!`,
    ]))];
    if (kind === 'doubledOff') return [this.C(this.pick('doubled-off', [`${r} gets doubled off! Should've stayed home!`, `${r} was halfway to the snack table! Doubled off!`, `Back, ${r}, back! Too late. Doubled off.`]))];
    if (kind === 'tag' && this.chance(0.5)) return [this.C(this.pick('tag', [
      `Tagged out! ${r} is out!`, `${f} applies the tag. Gotcha!`, `${r} tries to sneak by... tagged!`,
      `${f} slaps the tag on ${r}! Out!`, `Out! ${r} slid right into ${poss(f)} glove.`,
    ]))];
    if (kind === 'force' && this.chance(0.15)) return [this.C(this.pick('force', [`Got the force! ${r} is out.`, `${r} is forced out.`, `${f} steps on the bag. Force out!`]))];
    return [];
  }

  private runLine(id: string, m: Match): Line[] {
    const n = nick(id);
    if (this.walked) return [this.C(this.pick('walk-in', [`${n} walks in a run! Free run!`, `And ${n} trots home. A run walks in!`, `Bases-loaded walk! ${n} scores!`]))];
    if (m.phase !== 'live' || !this.chance(0.12)) return [];
    return [this.C(this.pick('run', [`${n} crosses the plate!`, `Here comes ${n} to score!`, `${n} touches home!`, `${n} scores! High-fives all around!`, `${n} slides in safe at home! Grass stains!`]))];
  }

  private pitchingChange(from: string, to: string, m: Match): Line[] {
    const f = nick(from), t = nick(to);
    const pos = this.positionOf(m, from);
    const out = [this.C(this.pick('pchange', [
      `Pitching change! ${f} heads out to the field, and ${t} takes the ball.`,
      `${f} is gassed. Here comes ${t} to pitch!`,
      pos ? `New arm on the mound: ${t}! ${f} moves to ${POS_NAME[pos]}.` : `New arm on the mound: ${t}!`,
      `${t} is the new pitcher. ${f}, go get some water.`,
      `${f} hands the ball to ${t}. That's a pitching change, folks.`,
    ]))];
    out.push(this.D(this.pick('pchange-d', [
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
    const out = [this.C(this.pick('half', [
      `That'll do it for the ${half$} of the ${ord}. ${sc}.`,
      `Three outs! End of the ${half$} of the ${ord}: ${sc}.`,
      `And that's the side. After ${inning}${half === 0 ? ' and a half' : ''}, it's ${sc}.`,
      `${half === 0 ? 'Middle' : 'End'} of the ${ord}. ${sc}.`,
      `Side retired. ${sc}, ${half$} of the ${ord} in the books.`,
      `The ${m.side(half).team.name} are done in the ${ord}. It's ${sc}.`,
      `Out number three, and we'll turn it over. ${sc}.`,
      `That's three. Bea, flip the scoreboard cards: ${sc}.`,
      `${half === 0 ? 'Halfway through' : 'All done with'} the ${ord}. On the cardboard scoreboard: ${sc}.`,
      `And the ${m.side(half).team.name} head out to the field. ${sc}.`,
      `Three up, three... well, three outs anyway. ${sc}.`,
      `Change sides! Everybody find your glove. ${sc}.`,
      `That retires the side in the ${ord}. ${sc}.`,
      `${ord.replace(/^./, (c) => c.toUpperCase())} inning, ${half$} half, done. ${sc}.`,
      `The ${m.side(half).team.name} leave it there. ${sc} after ${inning}${half === 0 ? ' and a half' : ''}.`,
      `Grab a juice box, folks. ${sc}, ${half$} of the ${ord}.`,
    ]))];
    // the game-over line follows on its own
    if ((half === 0 && inning >= n && h > a) || (half === 1 && inning >= n && a !== h)) return out;
    if (!this.chance(0.75)) return out;
    const diff = Math.abs(a - h);
    const lead: 0 | 1 = a > h ? 0 : 1;
    const leader = m.side(lead).team.name, trailer = m.side((1 - lead) as 0 | 1).team.name;
    const comeback = diff > 0 && this.maxDeficit[lead] >= 3;
    const inningRuns = m.line[half]?.[inning - 1] ?? 0;
    if (half === 1 && inning >= n && a === h) {
      out.push(this.D(this.pick('extras', ['We\'re going to EXTRA INNINGS! Nobody tell our moms.', 'Extra innings! I\'ll need another juice box.', 'Free baseball! The streetlights aren\'t on yet. Keep going!'])));
    } else if (half === 0 && inning === n) {
      out.push(this.C(a === h ? this.pick('last-tied', [`Tied up, last inning. The ${m.cfg.home.team.name} can win it right here.`, 'All even, bottom of the last. Here we go.'])
        : this.pick('last-trail', [`Last licks for the ${m.cfg.home.team.name}, down ${num(diff)}.`, `The ${m.cfg.home.team.name} need ${diff === 1 ? 'a run' : `${num(diff)} runs`} to tie. Last chance!`])));
    } else if (inningRuns >= 4) {
      const big = m.side(half).team.name;
      out.push(this.D(this.pick('big-inning', [
        `${inningRuns} runs that inning for the ${big}! The scoreboard kid needs more chalk.`,
        `That was a ${num(inningRuns)}-run inning. Mr. Mendoza ran out of buns just watching it.`,
      ])));
    } else if (comeback && this.chance(0.7)) {
      out.push(this.D(this.pick('comeback', [
        `What a comeback by the ${leader}! They were down ${num(this.maxDeficit[lead])} earlier!`,
        'Never count out a backyard team. NEVER.', `The ${leader} came all the way back. I'm getting goosebumps.`,
      ])));
    } else if (diff === 0) {
      out.push(this.D(this.pick('tie', ['All tied up! This is better than Saturday cartoons.', 'Tie game. Nobody blink.', 'Deadlocked! I\'m too nervous to finish my juice box.', 'Even steven. This is anybody\'s game.'])));
    } else if (diff >= 6) {
      out.push(this.D(this.pick('blowout', [
        `The ${leader} are running away with this one.`, `${trailer} fans, there's still... well, there's still snacks.`,
        'This is getting out of hand. Do we have a mercy rule? Somebody check.',
        `The ${trailer} might need a miracle. Or a really long inning.`,
      ])));
    } else if (diff <= 2 && this.chance(0.5)) {
      out.push(this.C(this.pick('close', [
        `The ${leader} hang on to a ${num(diff)}-run lead.`, `Close one! The ${trailer} are right there.`,
        diff === 1 ? 'A one-run game... wait, let me count. Yep, one run.' : `Just ${num(diff)} runs in it. Anybody's game.`,
      ])));
    } else if (half === 1 && inning === n - 1) {
      out.push(this.C(this.pick('to-last', [`Heading to the final inning! The ${leader} lead by ${num(diff)}.`, `One inning left. The ${trailer} need ${num(diff)} to tie.`])));
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
      this.C(this.pick('tie-end', ['It\'s a tie! Everybody\'s mom is calling them in for dinner.', `The streetlights are on. We'll call it a tie, ${sc}.`, 'A tie! Same time tomorrow, folks?'])),
      this.D(this.pick('tie-end-d', ['A tie is like kissing your sister. Not that I would know.', 'Nobody lost! That\'s what my tee-ball coach said every game.'])),
    ];
    const w = m.side(winner).team, l = m.side((1 - winner) as 0 | 1).team;
    const diff = Math.abs(a - h);
    const walkOff = winner === 1 && this.prevType !== 'halfOver';
    let call: string | null;
    if (walkOff) call = this.pick('walkoff', [`WALK-OFF! The ${w.street} ${w.name} win it at home!`, `They win it! The ${w.name} walk it off, ${sc}!`, `Ballgame! A walk-off for the ${w.name}! Everybody dogpile!`]);
    else if (Math.min(a, h) === 0) call = this.pick('shutout', [`That's the ballgame! A shutout for the ${w.street} ${w.name}, ${sc}.`, `Final: ${sc}. The ${l.name} never got on the board!`]);
    else if (diff >= 6) call = this.pick('blowout-end', [`That's all, folks! The ${w.name} roll, ${sc}.`, `Final score: ${sc}. The ${w.name} ran away with it.`]);
    else if (m.inning > n) call = this.pick('extras-end', [`The ${w.name} win it in extra innings! ${sc}.`, `Free baseball pays off for the ${w.name}! Final: ${sc}.`]);
    else call = this.pick('win', [`That's the ballgame! The ${w.street} ${w.name} win it!`, `Final score: ${sc}. The ${w.name} win!`, `And that'll do it! The ${w.name} take this one, ${sc}.`, `Put it in the books: ${w.street} ${w.name} win, ${sc}.`]);
    return [this.C(call), this.D(this.pick('end-d', [
      'What a game. I need a juice box.', 'I\'ll be signing autographs by the swing set.', 'Same time tomorrow?',
      `Good game, ${l.name}. Line up for high-fives, everybody.`, 'That\'s a wrap! I gotta be home before the streetlights.',
      'For Channel 4½, I\'m Dottie Fairweather. Former big leaguer.',
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
