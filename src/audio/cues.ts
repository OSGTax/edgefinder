/**
 * What the game sounds like, event by event: the crowd sized to the moment,
 * the ball on whatever it lands on, kids shouting in their own voices, the
 * booth talking under its captions, and the between-innings jingle.
 *
 * GameScreen hands every batch of match events to `GameSound.events()` and
 * every caption to `caption()`. The stingers (strike, out, safe), the bat
 * crack, mitt pops and the like are still played by GameScreen itself.
 *
 * The sounds go to a `CueSink` (the real audio module by default), so tests
 * can run a whole game through here and look at what it would have played.
 */
import { kid } from '../data/kids';
import type { Match } from '../sim/match';
import type { MatchEvent } from '../sim/types';
import { reportError } from './context';
import { playMusic } from './music';
import { playSfx } from './sfx';
import type { MusicTrack, PlayOpts, SfxName } from './types';
import { speak, type SpeakOpts } from './voice';

export interface CueSink {
  play(name: SfxName, opts?: PlayOpts, delay?: number): void;
  speak(who: string, text: string, opts?: SpeakOpts): number;
  music(track: MusicTrack): void;
}

const live: CueSink = {
  play: (name, opts, delay) => {
    if (delay && delay > 0) setTimeout(() => playSfx(name, opts), delay * 1000);
    else playSfx(name, opts);
  },
  speak: (who, text, opts) => speak(who, text, opts),
  music: (track) => playMusic(track),
};

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));

/** Short shouts (never shown on screen: they only give each bark its rhythm and vowels). */
const BARKS = {
  hit: ['Yeah!', 'Let\'s go!', 'Woo-hoo!', 'Yes, yes, yes!', 'Ha HA!', 'Did you see that?'],
  bigHit: ['Woooo!', 'YEAH, baby!', 'Look at it GO!', 'Whoa-ho-ho!'],
  homer: ['YES! YES! YES!', 'Woo-HOO! Ha ha!', 'See ya LATER!', 'Get outta here!'],
  whiffK: ['Aw, man!', 'No way!', 'Rats!', 'Ugh, seriously?'],
  lookK: ['That was outside!', 'Are you kidding?', 'Hey! No way!', 'Oh, come on!'],
  catch: ['Got it!', 'Mine!', 'I got it, I got it!', 'Yoink!'],
  oops: ['Oops!', 'Uh-oh!', 'Whoops-a-daisy!', 'Ah, nuts!'],
  ouch: ['Ow!', 'Ow, ow, ow!', 'Ouchie!'],
  bench: ['Let\'s go, NAME!', 'You got this, NAME!', 'Come on, NAME!', 'Hit it to the moon, NAME!', 'Eye on the ball, NAME!'],
  house: ['KAI! Aim LEFT, sweetie!', 'My WINDOWS!', 'Who hit the house?', 'Not the windows, kids!'],
} as const;

/** Persona barks for the moments a kid is known for; they win over the generic ones. */
const PERSONA: Record<string, Partial<Record<keyof typeof BARKS, readonly string[]>>> = {
  bo: { hit: ['Still got it!', 'Ha! Fixed it!'], whiffK: ['Oh, my back.', 'Bah!'] },
  ines: { hit: ['Thank you, thank you!'], whiffK: ['How DARE you!', 'Robbed! ROBBED!'], lookK: ['I was ROBBED!'] },
  toby: { hit: ['HUSTLE!', 'Move it, move it!'], catch: ['That\'s how it\'s DONE!'], whiffK: ['Detention! For me!'] },
  wren: { catch: ['Shh! Got it.'], hit: ['Oh! Oh!'] },
  dez: { hit: ['Smooooth.', 'Mm-hmm.'], homer: ['Smooooth, baby.'] },
  priya: { hit: ['Per my calculations!'], whiffK: ['Noted.'] },
  gus: { hit: ['TIMBERRR!'], homer: ['TIMBERRRR!'], catch: ['Yup.'] },
  molly: { catch: ['Order UP!'], hit: ['Pickle on the side!'] },
  jun: { hit: ['But wait, there\'s MORE!'], homer: ['Call NOW!'] },
  kai: { hit: ['KA-BOOM!'], homer: ['SUNDAY! SUNDAY! SUNDAY!'], whiffK: ['NOOOO!'] },
  ruby: { hit: ['Good MORNING!'], catch: ['Back to you, Chet!'] },
  ezra: { catch: ['Fully covered!'], oops: ['Act of nature!'] },
  maya: { hit: ['Special delivery!'], catch: ['Signed for!'] },
  leo: { hit: ['Magnifique!'], homer: ['Ooh la la!'], whiffK: ['Zut!'] },
  anya: { catch: ['Cold front!'], hit: ['Sunny skies!'] },
  darius: { lookK: ['OBJECTION!', 'I object! I OBJECT!'], hit: ['I rest my case.'] },
  pepper: { hit: ['SOLD-sold-sold!'], whiffK: ['No-sale-no-sale!'] },
  hank: { hit: ['Oh! Okay!'], catch: ['Gotcha, sir.'], oops: ['Sorry! Sorry!'] },
};

export class GameSound {
  private bounceT = -1;
  private now = 0;
  /** each kid's turn through their own barks, so nobody says the same thing twice in a row */
  private barked = new Map<string, number>();
  private benchT = -99;
  private houseT = -99;
  private boothFree = 0;
  private scoreBefore: [number, number] = [0, 0];

  constructor(private readonly sink: CueSink = live) {}

  /** Advance the clock (seconds). The screen's frame time is fine. */
  tick(dt: number): void {
    this.now += dt;
  }

  /**
   * How big this moment is, 0..1: late, close, runners on, two outs.
   * A routine single in the 1st is ~0.3; a tying hit in the last inning ~0.9.
   */
  moment(m: Match): number {
    const lateness = clamp01((m.inning - 1) / Math.max(1, m.cfg.innings - 1));
    const diff = Math.abs(m.score[0] - m.score[1]);
    const close = clamp01(1 - diff / 5);
    const on = m.bases.filter(Boolean).length;
    return clamp01(0.2 + 0.4 * close * (0.4 + 0.6 * lateness) + 0.08 * on + (m.outs === 2 ? 0.08 : 0));
  }

  /** Feed every event of one frame, before the screen clears them. */
  events(list: readonly MatchEvent[], m: Match, humanSide: -1 | 0 | 1): void {
    try {
      // a kid with a speech bubble this frame does the talking; skip the barks
      const quipping = new Set(list.filter((e) => e.type === 'quip').map((e) => (e as { kid: string }).kid));
      for (const e of list) this.event(e, m, humanSide, quipping);
    } catch (err) {
      reportError(err); // sound must never break the game
    }
  }

  /** An announcer's caption just went up. */
  caption(who: 'Chet' | 'Dottie', text: string): void {
    // one booth voice at a time: a caption arriving mid-sentence just shows
    if (this.now < this.boothFree) return;
    try {
      const d = this.sink.speak(who, text, { pan: who === 'Chet' ? -0.12 : 0.12, duck: true });
      this.boothFree = this.now + d;
    } catch (err) {
      reportError(err);
    }
  }

  private bark(id: string, moment: keyof typeof BARKS, opts: SpeakOpts = {}, name = ''): void {
    const own = PERSONA[id]?.[moment] ?? [];
    const pool = [...own, ...own, ...BARKS[moment]]; // persona lines come up twice as often
    const n = this.barked.get(id) ?? Math.floor(Math.random() * pool.length);
    this.barked.set(id, n + 1);
    const text = pool[n % pool.length].replace('NAME', name);
    this.sink.speak(id, text, { pan: (Math.random() - 0.5) * 0.5, ...opts });
  }

  private event(e: MatchEvent, m: Match, human: -1 | 0 | 1, quipping: Set<string>): void {
    // is this good news for the side the player is on? (CPU vs CPU: cheer everything)
    const good = (battingGood: boolean) => (human < 0 ? true : (m.battingSide === human) === battingGood);
    const mo = this.moment(m);
    switch (e.type) {
      case 'batterUp':
        this.scoreBefore = [m.score[0], m.score[1]];
        // now and then a teammate on the bench yells encouragement
        if (this.now - this.benchT > 25 && Math.random() < 0.18) {
          const mates = m.side(m.battingSide).team.roster.filter((id) => id !== e.batter);
          const mate = mates[Math.floor(Math.random() * mates.length)];
          if (mate) {
            this.benchT = this.now;
            this.bark(mate, 'bench', { level: 0.55, delay: 0.6, pan: m.battingSide === 0 ? 0.5 : -0.5 }, kid(e.batter).nick);
          }
        }
        break;
      case 'contact':
        // a long fly ball: the crowd rises with it
        if (e.ev > 70 && e.la > 18 && e.la < 45 && e.quality > 0.5) this.sink.play('ooh', { intensity: clamp01(0.3 + (e.ev - 70) / 30 + mo * 0.3) }, 0.25);
        break;
      case 'foulTip':
        // just ticks the bat and pops into the mitt
        this.sink.play('batTink', { intensity: 0.15 });
        this.sink.play('mittPop', { intensity: 0.6 }, 0.06);
        break;
      case 'hit': {
        const i = clamp01(0.15 + 0.17 * e.bases + 0.4 * mo);
        this.sink.play(good(true) ? 'cheer' : 'aww', { intensity: good(true) ? i : i * 0.7 });
        if (!quipping.has(e.batter) && Math.random() < 0.45) this.bark(e.batter, e.bases >= 2 ? 'bigHit' : 'hit', { delay: 0.3 });
        break;
      }
      case 'run': {
        // a run that ties it or takes the lead gets the whole yard on its feet
        const s = m.battingSide;
        const was = this.scoreBefore[s] - this.scoreBefore[1 - s];
        const now = m.score[s] - m.score[1 - s];
        if ((was <= 0 && now >= 0) || (was < 0 && now > 0)) {
          this.sink.play(good(true) ? 'cheer' : 'aww', { intensity: 0.95 }, 0.15);
        }
        this.scoreBefore = [m.score[0], m.score[1]];
        break;
      }
      case 'homeRun':
        this.sink.play(good(true) ? 'bigCheer' : 'aww', { intensity: good(true) ? clamp01(0.5 + 0.12 * e.runs + 0.3 * mo) : 1 });
        if (!quipping.has(e.batter)) this.bark(e.batter, 'homer', { delay: 0.5, level: 1.2 });
        break;
      case 'strikeout':
        this.sink.play(good(false) ? 'cheer' : 'aww', { intensity: clamp01(0.2 + 0.5 * mo) });
        if (!quipping.has(e.batter) && Math.random() < 0.5) this.bark(e.batter, e.looking ? 'lookK' : 'whiffK', { delay: 0.4 });
        break;
      case 'walk':
        if (e.hbp) {
          this.sink.play('bonk', { intensity: 0.7 });
          this.sink.play('dizzy', { intensity: 0.6 }, 0.25);
          this.bark(e.batter, 'ouch', { level: 1.1, delay: 0.2 });
        }
        break;
      case 'catch':
        if (e.hard) {
          this.sink.play(good(false) ? 'cheer' : 'aww', { intensity: clamp01(0.45 + 0.4 * mo) }, 0.1);
          if (!quipping.has(e.fielder) && Math.random() < 0.7) this.bark(e.fielder, 'catch', { delay: 0.15 });
        }
        break;
      case 'bobble':
        this.sink.play('boing', { intensity: 0.5 });
        this.sink.play('giggle', { intensity: 0.5 }, 0.3);
        if (Math.random() < 0.4) this.bark(e.fielder, 'oops');
        break;
      case 'error':
        this.sink.play(good(true) ? 'cheer' : 'aww', { intensity: 0.35 });
        if (!quipping.has(e.fielder) && Math.random() < 0.5) this.bark(e.fielder, 'oops', { delay: 0.2 });
        break;
      case 'bounce': {
        // one thump per bounce, whatever it's on (the bounces on a roller come fast)
        if (this.now - this.bounceT < 0.12 || e.surface === 'water') break;
        this.bounceT = this.now;
        const name: SfxName = e.surface === 'patio' ? 'bouncePatio' : e.surface === 'dirt' || e.surface === 'sand' ? 'bounceDirt' : 'bounce';
        this.sink.play(name, { intensity: clamp01(e.speed / 40) });
        break;
      }
      case 'fence': {
        if (e.cleared) break;
        const name: SfxName = e.kind === 'picket' ? 'picket' : e.kind === 'hedge' || e.kind === 'sunflower' || e.kind === 'reeds' ? 'hedge' : e.kind === 'house' || e.kind === 'garage' ? 'houseWall' : 'fence';
        this.sink.play(name, { intensity: 0.8 });
        this.sink.play('ooh', { intensity: 0.4 + 0.3 * mo });
        // off the house: Mrs. Mendoza comes to the back door to check on her windows
        if (name === 'houseWall' && this.now - this.houseT > 40) {
          this.houseT = this.now;
          this.sink.play('screenDoor', { intensity: 0.6, pan: -0.3 }, 0.7);
          const line = BARKS.house[Math.floor(Math.random() * BARKS.house.length)];
          this.sink.speak('mrsMendoza', line, { delay: 1.3, pan: -0.3, level: 0.9 });
        }
        break;
      }
      case 'splash':
        this.sink.play('giggle', { intensity: 0.7 }, 0.5);
        break;
      case 'quip':
        // the kid in the speech bubble, in their own voice
        this.sink.speak(e.kid, e.text, { level: 1.1, duck: true });
        break;
      case 'halfOver':
        // between innings somebody plays the toy keyboard (except after the final out)
        if (!(e.half === 1 && e.inning >= m.cfg.innings && m.score[0] !== m.score[1]) &&
          !(e.half === 0 && e.inning >= m.cfg.innings && m.score[1] > m.score[0])) {
          this.sink.music('inning');
        }
        break;
      default:
        break;
    }
  }
}
