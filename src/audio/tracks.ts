/**
 * The soundtrack: original tunes written for this game, played by the kids'
 * garage band (see instruments.ts). Chord strings give one chord per half
 * bar; melodies are bar-checked.
 */
import { arp, bassLine, beat, chain, comp, notes, rest, song, type Song } from './notation';
import type { MusicTrack } from './types';

const rep = (bar: string, n: number): string => Array(n).fill(bar).join('|');

// ── title: "Grass Stain Rag". G major, 120 BPM, 16 bars: A (8) + B (8). ─────
// The A tune is hummed on a kazoo over a strummed ukulele; somebody whistles
// the B section while a toy glockenspiel answers.
const TITLE_CHORDS =
  'G G | Em Em | C C | D D | G G | Em Em | C D | G G |' + // A
  'C C | D D | Bm Bm | Em Em | C C | D D | G G | D7 D7'; // B

const TITLE_A = notes(
  'B4/2 D5/2 G5/4 F#5/2 G5/2 A5/2 G5/2 | E5/4 ./2 E5/2 G5/2 F#5/2 E5/2 D5/2 |' +
    'C5/2 E5/2 G5/4 A5/2 G5/2 E5/2 C5/2 | D5/6 ./2 A4/2 B4/2 C5/2 D5/2 |' +
    'B4/2 D5/2 G5/4 F#5/2 G5/2 B5/2 A5/2 | G5/4 E5/2 ./2 E5/2 F#5/2 G5/2 E5/2 |' +
    'C5/2 E5/2 A5/4 D5/2 F#5/2 A5/4 | G5/8 ./4 D5/2 E5/2',
);
const TITLE_B = notes(
  'G5/3 G5/1 E5/2 G5/2 A5/4 G5/4 | F#5/3 F#5/1 D5/2 F#5/2 A5/4 F#5/4 |' +
    'D5/2 F#5/2 B5/4 A5/2 F#5/2 D5/4 | E5/2 G5/2 B5/2 A5/2 G5/4 ./4 |' +
    'C6/4 B5/2 A5/2 G5/4 E5/4 | D5/2 E5/2 F#5/2 G5/2 A5/4 D6/4 |' +
    'B5/4 A5/2 G5/2 D5/4 B4/4 | A4/2 B4/2 C5/2 D5/2 F#5/2 A5/2 ./4',
);
// the glockenspiel kid only knows a few notes, and plays them on the off-beats
const TITLE_GLOCK = notes(
  './6 G6/2 ./6 B6/2 | ./16 | ./6 E6/2 ./6 G6/2 | ./16 | ./6 F#6/2 ./6 A6/2 | ./16 | ./6 G6/2 ./4 B6/2 D7/2 | ./16',
);

const title = song({
  bpm: 120,
  level: 0.7,
  feel: 0.6,
  loop: true,
  parts: [
    ['kazoo', 0.2, chain(TITLE_A, rest(8))],
    ['whistle', 0.2, chain(rest(8), TITLE_B)],
    ['glock', 0.12, chain(rest(8), TITLE_GLOCK)],
    ['uke', 0.34, comp(TITLE_CHORDS, 'x..x..x.', 3, 60)],
    ['ukeBass', 0.42, bassLine(TITLE_CHORDS, [[0, 3], [0, 1], [7, 2], [12, 2]])],
    ['box', 0.42, beat('x.......x.x.....')],
    ['claps', 0.12, beat('....x.......x...')],
    // the B section picks up a 16th shaker
    ['shaker', 0.07, beat(rep('................', 8) + '|' + rep('.o.o.o.o.o.o.o.o', 8))],
  ],
});

// ── game: light and sparse under play, F major, 100 BPM. 8 bars of tune, 8 of air.
// A picked ukulele and a toy glockenspiel; nothing that fights the bat cracks.
const GAME_CHORDS = rep('F F | Dm Dm | Bb Bb | C C | F F | Dm Dm | Gm Gm | C C', 2);

const GAME_TUNE = notes(
  './4 A5/2 C6/2 ./8 | ./4 D6/2 C6/2 A5/4 ./4 | ./8 F5/2 G5/2 A5/2 Bb5/2 | C6/6 ./10 |' +
    './4 A5/2 C6/2 F6/4 ./4 | E6/2 D6/2 ./4 A5/4 ./4 | ./4 Bb5/2 A5/2 G5/4 D6/4 | C6/4 E5/2 G5/2 ./8 |' +
    './16 | ./8 F6/2 E6/2 D6/4 | ./16 | ./8 E6/2 D6/2 C6/4 |' +
    './16 | ./8 A6/2 G6/2 F6/4 | ./8 D6/2 E6/2 F6/4 | E6/4 G6/4 ./8',
);

const game = song({
  bpm: 100,
  level: 0.75, // sparse, so it sits under the bat cracks and mitt pops anyway
  feel: 0.5,
  loop: true,
  parts: [
    ['glock', 0.17, GAME_TUNE],
    ['uke', 0.26, arp(GAME_CHORDS, '0.1.2.1.', 3, 57)],
    ['ukeBass', 0.34, bassLine(GAME_CHORDS, [[0, 3], [null, 3], [7, 2]])],
    ['box', 0.28, beat('x.........x.....')],
    ['shaker', 0.06, beat('..o...o...o...o.')],
  ],
});

// ── inning: between innings somebody plays the toy keyboard, with its
// built-in bossa beat. They hit a wrong note, stop, and fix it. ~6 s, then
// hands back to the game music.
const inning = song({
  bpm: 120,
  level: 0.6,
  feel: 1,
  loop: false,
  then: true,
  parts: [
    ['toy', 0.2, notes('C5/2 E5/2 G5/2 E5/2 A5/4 G5/4 | F5/2 E5/2 D5/2 F#5/1 ./2 F5/1 E5/2 D5/4 | E5/2 D5/2 C5/2 G4/2 C5/8')],
    ['toy', 0.08, notes('C4+E4+G4/2 ./6 C4+E4+G4/2 ./6 | F3+A3+C4/2 ./6 G3+B3+D4/2 ./6 | C4+E4+G4/2 ./6 C4+E4+G4/8')],
    ['toy', 0.12, notes('C3/2 ./4 G3/2 ./8 | F3/2 ./6 G3/2 ./6 | C3/2 ./6 C3/8')],
    ['toyKick', 0.32, beat('x......xx.......|x......xx.......|x.......x.......')],
    ['toySnare', 0.1, beat('x..x..x...x..x..|x..x..x...x..x..|x..x..x.........')],
    ['toyHat', 0.05, beat('x.x.x.x.x.x.x.x.|x.x.x.x.x.x.x.x.|x.x.x.x.........')],
  ],
});

// ── season: relaxed swung hub groove, Bb major, 92 BPM. 16 bars. ────────────
const SEASON_CHORDS =
  'Bb Bb | Gm Gm | Eb Eb | F F | Bb Bb | Gm Gm | Cm F | Bb Bb |' +
  'Eb Eb | F F | Dm Dm | Gm Gm | Eb Eb | F F | Bb Bb | F7 F7';

const SEASON_LEAD = notes(
  'D5/3 F5/3 D5/2 C5/2 Bb4/2 ./4 | ./2 G4/2 Bb4/2 D5/4 C5/2 Bb4/4 |' +
    'G5/3 F5/3 Eb5/2 D5/2 Eb5/2 ./4 | C5/6 ./2 A4/2 Bb4/2 C5/4 |' +
    'D5/3 F5/3 Bb5/2 A5/2 F5/2 ./4 | G5/4 F5/2 D5/2 ./2 Bb4/2 D5/4 |' +
    'Eb5/4 D5/2 C5/2 A4/4 C5/4 | Bb4/10 ./6 |' +
    './4 G4/2 Bb4/2 Eb5/4 D5/4 | ./4 A4/2 C5/2 F5/4 Eb5/4 |' +
    'D5/4 F5/2 A5/2 ./2 G5/2 F5/4 | G5/8 ./4 F5/2 G5/2 |' +
    'Bb5/4 G5/2 Eb5/2 ./2 F5/2 G5/4 | A5/4 F5/2 C5/2 ./2 D5/2 Eb5/4 |' +
    'D5/6 C5/2 Bb4/4 F4/4 | A4/4 C5/4 Eb5/4 ./4',
);

const season = song({
  bpm: 92,
  level: 0.7,
  swing: 0.35,
  feel: 0.7,
  loop: true,
  parts: [
    ['whistle', 0.24, SEASON_LEAD],
    ['uke', 0.3, comp(SEASON_CHORDS, '..x...x.', 2, 58)],
    ['ukeBass', 0.4, bassLine(SEASON_CHORDS, [[0, 3], [7, 3], [12, 2]])],
    ['box', 0.36, beat('x.........x.....')],
    ['claps', 0.1, beat('....x.......x...')],
    ['shaker', 0.045, beat('x.o.x.o.x.o.x.o.')],
  ],
});

// ── victory: ~4 s kazoo-band fanfare in C, plays once. ──────────────────────
const victory = song({
  bpm: 132,
  level: 0.8,
  feel: 0.5,
  loop: false,
  parts: [
    ['kazoo', 0.22, notes('G4/2 C5/2 E5/2 G5/4 E5/2 G5/2 A5/2 | G5/2 A5/2 B5/2 C6/10')],
    ['kazoo', 0.13, notes('E4/2 G4/2 C5/2 E5/4 C5/2 E5/2 F5/2 | D5/2 F5/2 G5/2 E5/10')],
    ['ukeBass', 0.42, notes('C3/2 C3/2 G3/2 C4/2 F2/2 F3/2 C3/2 F3/2 | G2/2 G3/2 B2/2 C3/10')],
    ['uke', 0.3, notes('C4+E4+G4+C5/3 ./5 F4+A4+C5/3 ./5 | G4+B4+D5/3 ./3 C4+E4+G4+C5/10')],
    ['box', 0.42, beat('x...x...x...x...|x.x.x.X.........')],
    ['claps', 0.15, beat('....x.......x.x.|x.xxx...........')],
    ['glock', 0.12, notes('./16 | ./6 C7/2 E7/2 G7/6')],
    ['lid', 0.5, beat('................|......X.........')],
  ],
});

// ── defeat: ~5 s, a music box and a soft whistle. Chin up: it ends in major.
const defeat = song({
  bpm: 96,
  level: 0.7,
  feel: 0.8,
  loop: false,
  parts: [
    ['musicbox', 0.3, notes('A5/4 G5/2 F5/2 D5/4 C5/4 | D5/2 F5/2 E5/2 C5/2 F5/8')],
    ['whistle', 0.12, notes('./8 F4/4 E4/4 | F4/4 G4/4 A4/8')],
    ['uke', 0.18, notes('D4+F4+A4/8 Bb3+D4+F4/8 | C4+E4+G4/8 F4+A4+C5/8')],
    ['ukeBass', 0.3, notes('D3/8 Bb2/8 | C3/8 F2/8')],
  ],
});

export const SONGS: Record<MusicTrack, Song> = { title, game, inning, victory, defeat, season };
