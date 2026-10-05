import type { Kid } from '../data/types';

// The grown-ups who hang around the yard. They're built with the same
// character kit as the kids (with a Hawaiian shirt instead of a jersey) and
// scaled up — big cartoon heads and all.

export const MR_MENDOZA: Kid = {
  id: 'mrMendoza', first: 'Hector', last: 'Mendoza', nick: 'Mr. Mendoza', age: 44, bats: 'R', throws: 'R',
  traits: { contact: 1, power: 1, speed: 1, fielding: 1, arm: 1, pitching: 1, control: 1 },
  pitches: ['fastball'], special: 'flypaper', persona: 'Grill Sergeant',
  look: {
    skin: 3, hair: 'sidepart', hairColor: 0, head: 'wide', mouth: 'grin', hat: 'none' as never, eyewear: 'aviators',
    face: 'walrus', neck: 'none', extra: 'none', body: 'apron', holding: 'spatula', freckles: false, height: 1, build: 1,
  },
  bio: 'Has been at the grill since 9 a.m. Nobody has seen a burger yet.',
  quips: ['Burgers in five!', 'Watch the pool!', 'Who wants a hot dog?', 'That\'s my boy!'],
};

export const GROWNUP_SCALE = 1.3;
