import { Rng } from '../engine/rng';
import { kid } from '../data/kids';
import { SPECIAL_INFO, type Kid } from '../data/types';
import type { MatchEvent } from '../sim/types';
import type { Match } from '../sim/match';

// Chet Valentine (play-by-play, age 10) and Dottie Fairweather (color,
// age 12, "former big leaguer"). They are kids pretending to be a network
// broadcast booth.

export interface Line { who: 'Chet' | 'Dottie'; text: string }

const nick = (id: string) => kid(id).nick;

export class Booth {
  private rng: Rng;
  private lastBatterLine = '';
  constructor(seed: number) {
    this.rng = new Rng(seed ^ 0xb00b);
  }

  private pick(...opts: string[]) {
    return this.rng.pick(opts);
  }

  /** Returns 0–2 lines for an event (most events say nothing). */
  react(e: MatchEvent, m: Match): Line[] {
    const P = (s: string): Line => ({ who: 'Chet', text: s });
    const D = (s: string): Line => ({ who: 'Dottie', text: s });
    switch (e.type) {
      case 'batterUp': {
        const k = kid(e.batter);
        if (this.lastBatterLine === k.id) return [];
        this.lastBatterLine = k.id;
        const box = m.box[k.id]?.bat;
        const hits = box?.h ?? 0, ab = box?.ab ?? 0;
        const today = ab > 0 ? ` ${hits} for ${ab} today.` : '';
        const intro = this.pick(
          `Now batting: ${k.first} "${k.nick}" ${k.last}.${today}`,
          `Stepping in, it's ${k.nick}, ${k.persona.toLowerCase()}.${today}`,
          `Here comes ${k.nick} to the plate.${today}`,
          `Batting for ${m.side(m.battingSide).team.name}: ${k.nick}, age ${k.age}.${today}`,
        );
        const out = [P(intro)];
        if (this.rng.chance(0.4)) out.push(D(this.pick(k.bio, `${k.nick}? ${k.bio}`, `Folks, ${k.bio.charAt(0).toLowerCase()}${k.bio.slice(1)}`)));
        return out;
      }
      case 'special': {
        const info = SPECIAL_INFO[e.special];
        return [P(`${nick(e.kid)} is going for the ${info.label.toUpperCase()}!`), ...(this.rng.chance(0.5) ? [D(this.pick('Oh, this is gonna be good.', 'I invented that move, by the way.', 'Somebody get the camcorder!'))] : [])];
      }
      case 'strikeout':
        return [P(e.looking
          ? this.pick(`${nick(e.batter)} watches strike three go by!`, `Caught looking! ${nick(e.batter)} is frozen like a popsicle.`)
          : this.pick(`Struck out swinging! ${nick(e.batter)} spun all the way around.`, `Strike three! ${nick(e.batter)} swung at a ghost.`, `He went fishing and caught nothing. Strike three!`.replace('He', nick(e.batter)))),
        ...(this.rng.chance(0.35) ? [D(this.pick(`${m.pitcher.nick} is dealing today.`, 'In my day we called that "the windmill."', 'That pitch had some MUSTARD on it.', 'Shake it off, kid. Shake it off.'))] : [])];
      case 'walk':
        return [P(e.hbp ? this.pick(`Ouch! ${nick(e.batter)} takes one for the team.`, `Plunked! ${nick(e.batter)} is rubbing it and heading to first.`) : this.pick(`Ball four. ${nick(e.batter)} takes a walk.`, `${nick(e.batter)} draws the walk. Patience!`))];
      case 'homeRun': {
        const k = kid(e.batter);
        const extra = e.runs > 1 ? ` That's ${e.runs} runs!` : '';
        return [P(this.pick(`IT'S OUTTA HERE! ${k.nick} crushed it!${extra}`, `GOODBYE, BASEBALL! A home run for ${k.nick}!${extra}`, `Somebody call the neighbors! ${k.nick} hit it into the next zip code!${extra}`)),
          D(this.pick('Somebody\'s gonna have to go knock on a door for that one.', 'I hit one like that once. In tee-ball. It was off the tee, but still.', 'That ball is still going, folks.', `${k.nick} is gonna be insufferable at lunch tomorrow.`))];
      }
      case 'hit': {
        const k = kid(e.batter);
        const kind = e.bases >= 3 ? 'triple' : e.bases === 2 ? 'double' : 'single';
        const lines = {
          single: [`Base hit for ${k.nick}!`, `${k.nick} pokes one through!`, `Single for ${k.nick}. Nothing fancy, gets the job done.`],
          double: [`${k.nick} lines one into the gap. That's a double!`, `Two-bagger for ${k.nick}!`, `${k.nick} cruises into second with a double!`],
          triple: [`A TRIPLE for ${k.nick}! Nobody hits triples!`, `${k.nick} is flying around the bases! TRIPLE!`],
        }[kind];
        return [P(this.rng.pick(lines))];
      }
      case 'groundRule':
        if (e.why === 'splash') return [P(this.pick('SPLASH! It\'s in the water! That\'s a splash double!', 'Into the drink! Splash double!')), D(this.pick('Somebody get the pool skimmer.', 'Hope nobody was swimming.', 'That ball needs a towel.'))];
        return [P('Bounced over the fence! Ground-rule double.')];
      case 'out':
        if (e.kind === 'doubledOff') return [P(`${nick(e.runner)} gets doubled off! Should've stayed home!`)];
        if (e.kind === 'tag' && !this.rng.chance(0.5)) return [P(this.pick(`Tagged out! ${nick(e.runner)} is out!`, `${nick(e.fielder)} applies the tag. Gotcha!`))];
        return [];
      case 'catch':
        if (e.fly && e.hard) return [P(this.pick(`WHAT A CATCH by ${nick(e.fielder)}!`, `${nick(e.fielder)} lays out and MAKES THE GRAB!`, `Are you kidding me?! ${nick(e.fielder)} caught it!`)), D(this.pick('Put that on the refrigerator!', 'Mom, are you filming?!'))];
        if (e.fly && this.rng.chance(0.35)) return [P(this.pick(`${nick(e.fielder)} squeezes it. Out.`, `Easy catch for ${nick(e.fielder)}.`, `Can of corn to ${nick(e.fielder)}.`))];
        return [];
      case 'error':
        return [P(this.pick(`Oh no, ${nick(e.fielder)} drops it!`, `It's off the glove! Error, ${nick(e.fielder)}!`, `${nick(e.fielder)} had it... and now doesn't.`)), ...(this.rng.chance(0.5) ? [D(this.pick('That\'s going in the blooper reel.', 'Happens to the best of us. Mostly to the worst of us.', 'Somebody needs a bigger glove.'))] : [])];
      case 'dog':
        return [P(this.pick('The dog is involved now! THE DOG IS INVOLVED!', 'It hit the doghouse! Biscuit is NOT happy.'))];
      case 'tree':
        return this.rng.chance(0.6) ? [P(this.pick('Into the tree! Leaves everywhere!', 'It\'s in the branches!'))] : [];
      case 'fence':
        return e.cleared ? [] : this.rng.chance(0.4) ? [P(this.pick('Off the fence!', 'Rattles around off the wall!'))] : [];
      case 'halfOver': {
        const [a, hScore] = m.score;
        return [P(`That'll do it for the ${e.half === 0 ? 'top' : 'bottom'} of the ${ordinal(e.inning)}. ${m.cfg.away.team.abbr} ${a}, ${m.cfg.home.team.abbr} ${hScore}.`)];
      }
      case 'gameOver': {
        const w = e.winner === 0 ? m.cfg.away.team : e.winner === 1 ? m.cfg.home.team : null;
        return [P(w ? `That's the ballgame! The ${w.street} ${w.name} win it!` : 'It\'s a tie! Everybody\'s mom is calling them in for dinner.'), D(this.pick('What a game. I need a juice box.', 'I\'ll be signing autographs by the swing set.', 'Same time tomorrow?'))];
      }
      default:
        return [];
    }
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
