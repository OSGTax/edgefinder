/**
 * The soundtrack: original tunes written for this game.
 * Chord strings give one chord per half bar; melodies are bar-checked.
 */
import { bassLine, beat, comp, notes, song, type Song } from './notation';
import type { MusicTrack } from './types';

const rep = (bar: string, n: number): string => Array(n).fill(bar).join('|');

// ── title: sunny, bouncy, G major, 120 BPM. 16 bars: A (8) + B (8). ──────────
const TITLE_CHORDS =
  'G G | Em Em | C C | D D | G G | Em Em | C D | G G |' + // A
  'C C | D D | Bm Bm | Em Em | C C | D D | G G | D7 D7'; // B

const TITLE_LEAD = notes(
  // A
  'B4/2 D5/2 G5/4 F#5/2 G5/2 A5/2 G5/2 | E5/4 ./2 E5/2 G5/2 F#5/2 E5/2 D5/2 |' +
    'C5/2 E5/2 G5/4 A5/2 G5/2 E5/2 C5/2 | D5/6 ./2 A4/2 B4/2 C5/2 D5/2 |' +
    'B4/2 D5/2 G5/4 F#5/2 G5/2 B5/2 A5/2 | G5/4 E5/2 ./2 E5/2 F#5/2 G5/2 E5/2 |' +
    'C5/2 E5/2 A5/4 D5/2 F#5/2 A5/4 | G5/8 ./4 D5/2 E5/2 |' +
    // B
    'G5/3 G5/1 E5/2 G5/2 A5/4 G5/4 | F#5/3 F#5/1 D5/2 F#5/2 A5/4 F#5/4 |' +
    'D5/2 F#5/2 B5/4 A5/2 F#5/2 D5/4 | E5/2 G5/2 B5/2 A5/2 G5/4 ./4 |' +
    'C6/4 B5/2 A5/2 G5/4 E5/4 | D5/2 E5/2 F#5/2 G5/2 A5/4 D6/4 |' +
    'B5/4 A5/2 G5/2 D5/4 B4/4 | A4/2 B4/2 C5/2 D5/2 F#5/2 A5/2 ./4',
);

const title = song({
  bpm: 120,
  level: 0.7,
  loop: true,
  parts: [
    ['lead', 0.2, TITLE_LEAD],
    ['bass', 0.26, bassLine(TITLE_CHORDS, [[0, 3], [0, 1], [7, 2], [12, 2]])],
    ['keys', 0.08, comp(TITLE_CHORDS, '..x...x.')],
    ['kick', 0.42, beat('x.......x.x.....')],
    ['clap', 0.16, beat('....x.......x...')],
    ['hat', 0.05, beat('o.x.o.x.o.x.o.x.')],
    // the B section picks up a 16th shaker
    ['shaker', 0.08, beat(rep('................', 8) + '|' + rep('.o.o.o.o.o.o.o.o', 8))],
  ],
});

// ── game: light and sparse under play, F major, 100 BPM. 8 bars of tune, 8 of air.
const GAME_CHORDS = rep('F F | Dm Dm | Bb Bb | C C | F F | Dm Dm | Gm Gm | C C', 2);

const GAME_TUNE = notes(
  './4 A4/2 C5/2 ./8 | ./4 D5/2 C5/2 A4/4 ./4 | ./8 F4/2 G4/2 A4/2 Bb4/2 | C5/6 ./10 |' +
    './4 A4/2 C5/2 F5/4 ./4 | E5/2 D5/2 ./4 A4/4 ./4 | ./4 Bb4/2 A4/2 G4/4 D5/4 | C5/4 E4/2 G4/2 ./8 |' +
    './16 | ./8 F5/2 E5/2 D5/4 | ./16 | ./8 E5/2 D5/2 C5/4 |' +
    './16 | ./8 A5/2 G5/2 F5/4 | ./8 D5/2 E5/2 F5/4 | E5/4 G5/4 ./8',
);

const game = song({
  bpm: 100,
  level: 0.5, // sits well under the bat cracks and mitt pops
  loop: true,
  parts: [
    ['bell', 0.2, GAME_TUNE],
    ['bass', 0.22, bassLine(GAME_CHORDS, [[0, 3], [null, 3], [7, 2]])],
    ['keys', 0.05, comp(GAME_CHORDS, 'x.......', 6)],
    ['kick', 0.3, beat('x.........x.....')],
    ['shaker', 0.07, beat('..o...o...o...o.')],
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
  loop: true,
  parts: [
    ['soft', 0.25, SEASON_LEAD],
    ['bass', 0.25, bassLine(SEASON_CHORDS, [[0, 3], [7, 3], [12, 2]])],
    ['keys', 0.075, comp(SEASON_CHORDS, '......x.')],
    ['kick', 0.36, beat('x.........x.....')],
    ['clap', 0.12, beat('....x.......x...')],
    ['shaker', 0.045, beat('x.o.x.o.x.o.x.o.')],
  ],
});

// ── victory: ~4 s fanfare in C, plays once. ─────────────────────────────────
const victory = song({
  bpm: 132,
  level: 0.8,
  loop: false,
  parts: [
    ['lead', 0.22, notes('G4/2 C5/2 E5/2 G5/4 E5/2 G5/2 A5/2 | G5/2 A5/2 B5/2 C6/10')],
    ['soft', 0.18, notes('E4/2 G4/2 C5/2 E5/4 C5/2 E5/2 F5/2 | D5/2 F5/2 G5/2 E5/10')],
    ['bass', 0.27, notes('C3/2 C3/2 G3/2 C4/2 F2/2 F3/2 C3/2 F3/2 | G2/2 G3/2 B2/2 C3/10')],
    ['keys', 0.07, notes('C4+E4+G4/3 ./5 F4+A4+C5/3 ./5 | G4+B4+D5/3 ./3 C4+E4+G4+C5/10')],
    ['kick', 0.37, beat('x...x...x...x...|x.x.x.X.........')],
    ['clap', 0.15, beat('....x.......x.x.|x.xxx...........')],
    ['hat', 0.055, beat('x.x.x.x.x.x.x.x.|x.x.x...........')],
    ['crash', 0.2, beat('................|......X.........')],
  ],
});

export const SONGS: Record<MusicTrack, Song> = { title, game, victory, season };
