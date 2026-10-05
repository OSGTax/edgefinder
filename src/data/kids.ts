import type { Bats, Hand, Kid, KidLook, PitchType, Special } from './types';

// Every kid in the league is a mini adult: an 8-to-12-year-old who dresses,
// talks and plays like a very specific kind of grown-up.

type Look = Pick<KidLook, 'hair' | 'hairColor' | 'skin'> & Partial<KidLook>;

const DEFAULT_LOOK: Omit<KidLook, 'hair' | 'hairColor' | 'skin'> = {
  head: 'round',
  mouth: 'grin',
  hat: 'cap',
  eyewear: 'none',
  face: 'none',
  neck: 'none',
  extra: 'none',
  body: 'none',
  holding: 'none',
  freckles: false,
  height: 0.5,
  build: 0.5,
};

function K(
  id: string,
  first: string,
  last: string,
  nick: string,
  age: number,
  bats: Bats,
  throws: Hand,
  [contact, power, speed, fielding, arm, pitching, control]: [number, number, number, number, number, number, number],
  pitches: PitchType[],
  special: Special,
  persona: string,
  look: Look,
  bio: string,
  quips: string[],
): Kid {
  return {
    id, first, last, nick, age, bats, throws,
    traits: { contact, power, speed, fielding, arm, pitching, control },
    pitches, special, persona,
    look: { ...DEFAULT_LOOK, ...look },
    bio, quips,
  };
}


// Traits: [contact, power, speed, fielding, arm, pitching, control], each 1–10;
// no two kids share the same seven numbers.
export const KIDS: Kid[] = [
  // ── Maple Street Mudcats ────────────────────────────────────────────
  K('bo', 'Bo', 'Rutherford', 'Mudpie', 11, 'R', 'R', [6, 10, 3, 5, 7, 3, 3], ['fastball', 'changeup'], 'moonshot', 'The Retired Plumber',
    { hair: 'messy', hairColor: 2, skin: 1, build: 0.95, height: 0.6, head: 'wide', mouth: 'frown', face: 'walrus', body: 'suspenders', holding: 'wrench' },
    'Eleven years old. Complains about his knees. Says "they don\'t make \'em like they used to" about juice boxes.',
    ['Oh, my back.', 'I\'ve seen worse clogs.', 'That\'s gonna cost ya extra.', 'Kids today, I tell ya.']),
  K('ines', 'Ines', 'Calderón', 'Inny', 11, 'R', 'R', [5, 3, 6, 6, 6, 8, 10], ['fastball', 'curve', 'changeup'], 'loopy', 'The Soap Opera Star',
    { hair: 'long', hairColor: 0, skin: 3, height: 0.55, mouth: 'smirk', eyewear: 'shades', neck: 'scarf', holding: 'microphone' },
    'Treats every pitch like the season finale. Has fainted dramatically on the mound four times this year.',
    ['How DARE you.', 'My public adores me.', 'I was ROBBED!', 'Get my good side!']),
  K('toby', 'Tobias', 'Grant', 'Toby', 10, 'R', 'R', [6, 5, 9, 8, 7, 4, 5], ['fastball', 'slider'], 'zoomies', 'The Gym Teacher',
    { hair: 'sidepart', hairColor: 4, skin: 0, height: 0.45, hat: 'visor', neck: 'whistle', extra: 'sweatband', holding: 'clipboard' },
    'Blows his whistle at everything. Gave his own mom a tardy slip.',
    ['Hustle, people!', 'Give me twenty laps!', 'That\'s a teachable moment.', 'No running in the— oh wait.']),
  K('wren', 'Wren', 'Halloway', 'Birdie', 9, 'L', 'L', [6, 3, 9, 8, 6, 3, 4], ['fastball', 'changeup'], 'springs', 'The Bird Watcher',
    { hair: 'pigtails', hairColor: 3, skin: 1, height: 0.3, build: 0.25, freckles: true, hat: 'bucket', holding: 'binoculars' },
    'Keeps a life list of every bird she\'s seen from center field. Currently at 212.',
    ['Shh! You\'ll scare it!', 'A rare Red-Breasted Fastball!', 'Magnificent wingspan.', 'Noted in my journal.']),
  K('dez', 'Desmond', 'Pike', 'Dez', 12, 'L', 'L', [9, 5, 4, 5, 5, 4, 6], ['fastball', 'sinker'], 'laser', 'The Late-Night Jazz DJ',
    { hair: 'afro', hairColor: 0, skin: 5, height: 0.85, build: 0.6, face: 'goatee', eyewear: 'shades', holding: 'microphone' },
    'Talks in a voice two octaves lower than his real one. Calls everyone "baby."',
    ['Smoooooth.', 'You\'re listening to Dez FM.', 'Keep it mellow, baby.', 'That one\'s goin\' out to the ladies.']),
  K('priya', 'Priya', 'Raman', 'Pree', 10, 'R', 'R', [9, 4, 6, 7, 6, 3, 7], ['changeup', 'curve'], 'eagleEye', 'The CPA',
    { hair: 'braids', hairColor: 0, skin: 3, height: 0.4, eyewear: 'reading', body: 'pocketProtector', holding: 'calculator' },
    'Keeps the stats for every game in a ledger. Double-entry. In pen. Has already filed her taxes for next year.',
    ['That run is non-deductible.', 'Let\'s reconcile this inning.', 'I\'m filing an extension.', 'The numbers don\'t lie.']),
  K('gus', 'Gus', 'Pelletier', 'Gus-Gus', 11, 'R', 'R', [5, 8, 4, 5, 10, 6, 4], ['fastball', 'sinker'], 'rocketArm', 'The Lumberjack',
    { hair: 'buzz', hairColor: 4, skin: 0, build: 0.8, head: 'square', mouth: 'open', face: 'beard', hat: 'trucker' },
    'Wears flannel in July. Fake beard is made of a cut-up yellow sponge.',
    ['TIMBERRR!', 'I eat fastballs for breakfast.', 'Splittin\' wood, baby!', 'That one\'s goin\' down.']),
  K('molly', 'Molly', 'Fitch', 'Pickles', 9, 'R', 'R', [5, 4, 6, 9, 3, 6, 7], ['changeup', 'knuckler'], 'flypaper', 'The Deli Counter Lady',
    { hair: 'bob', hairColor: 6, skin: 0, freckles: true, height: 0.35, mouth: 'tongue', body: 'apron', extra: 'pencil' },
    'Takes everyone\'s order between innings. Nobody asked her to. Nobody has ever gotten their food.',
    ['Order up!', 'Pickle on the side?', 'Number forty-two!', 'You want that toasted?']),
  K('jun', 'Jun', 'Watanabe', 'Jumpin\' Jun', 10, 'R', 'R', [5, 5, 7, 6, 7, 9, 6], ['fastball', 'slider', 'changeup'], 'heater', 'The Infomercial Host',
    { hair: 'spiky', hairColor: 0, skin: 1, height: 0.5, hat: 'capBack', extra: 'headset', mouth: 'open' },
    'Narrates his own pitches like they\'re on sale. Has tried to sell the umpire a blender.',
    ['But wait, there\'s MORE!', 'Call in the next ten minutes!', 'Operators are standing by!', 'Act now!']),

  // ── Cedar Lane Comets ───────────────────────────────────────────────
  K('kai', 'Kai', 'Mendoza', 'Kaboom', 12, 'R', 'R', [5, 9, 6, 5, 6, 10, 5], ['fastball', 'sinker', 'changeup'], 'heater', 'The Monster Truck Announcer',
    { hair: 'spiky', hairColor: 0, skin: 3, height: 0.8, build: 0.7, mouth: 'open', eyewear: 'shades', holding: 'microphone' },
    'Cannot speak below full volume. Announces his own pitches. Every one is "SUNDAY SUNDAY SUNDAY."',
    ['SUNDAY! SUNDAY! SUNDAY!', 'BE THERE!', 'MAXIMUM POWER!', 'KA-BOOOOM!']),
  K('ruby', 'Ruby', 'Santangelo', 'Rooster', 10, 'R', 'R', [9, 5, 7, 8, 7, 3, 4], ['fastball', 'curve'], 'eagleEye', 'The Morning Show Host',
    { hair: 'curly', hairColor: 3, skin: 1, height: 0.4, mouth: 'grin', neck: 'scarf', holding: 'coffee' },
    'Gets up at 5 a.m. Is VERY chipper about it. Does the weather and traffic before every at-bat.',
    ['Gooood MORNING, Cedar Lane!', 'Stay tuned!', 'Back to you, Chet!', 'What a beautiful day!']),
  K('ezra', 'Ezra', 'Goldfarb', 'Easy E', 11, 'R', 'R', [7, 5, 3, 9, 7, 4, 6], ['fastball', 'changeup'], 'flypaper', 'The Insurance Salesman',
    { hair: 'curly', hairColor: 1, skin: 1, height: 0.55, build: 0.7, eyewear: 'glasses', neck: 'tie', holding: 'briefcase' },
    'Has sold three of the other catchers "foul tip insurance." It is not a real thing.',
    ['Are you covered for that?', 'Let\'s talk deductibles.', 'Read the fine print.', 'Act of nature, not my fault.']),
  K('maya', 'Maya', 'Thompson', 'Zoom Zoom', 10, 'L', 'R', [7, 2, 10, 8, 6, 2, 3], ['fastball', 'changeup'], 'zoomies', 'The Mail Carrier',
    { hair: 'braids', hairColor: 0, skin: 4, height: 0.45, build: 0.25, hat: 'visor', body: 'fannyPack', holding: 'newspaper' },
    'Fastest kid on Cedar Lane. Delivers the base hit, signs for it, and leaves a slip if you\'re not home.',
    ['Special delivery!', 'Neither rain nor sleet!', 'Sign here, please.', 'Return to sender!']),
  K('leo', 'Leo', 'Fontaine', 'Lefty Leo', 11, 'L', 'L', [8, 6, 4, 5, 4, 6, 7], ['fastball', 'curve', 'changeup'], 'eagleEye', 'The Fancy French Chef',
    { hair: 'bowl', hairColor: 4, skin: 0, height: 0.6, face: 'handlebar', neck: 'bandana', holding: 'rollingPin' },
    'Is not French. Has never been to France. Calls every pitch "un soufflé."',
    ['Magnifique!', 'Zis pitch is overcooked!', 'Bon appétit!', 'More butter!']),
  K('anya', 'Anya', 'Lindqvist', 'Snowball', 11, 'R', 'R', [6, 6, 5, 6, 10, 5, 3], ['fastball', 'slider'], 'rocketArm', 'The TV Weather Lady',
    { hair: 'pigtails', hairColor: 5, skin: 0, height: 0.6, extra: 'bow', neck: 'pearls', mouth: 'grin' },
    'Gives a full five-day forecast before every throw. Is wrong about 90% of the time.',
    ['Ninety percent chance of OUT!', 'A cold front moving in!', 'Sunny with a chance of dingers.', 'Back to you in the studio!']),
  K('darius', 'Darius', 'King', 'D-King', 11, 'R', 'R', [8, 5, 6, 6, 6, 6, 5], ['fastball', 'curve', 'changeup'], 'laser', 'The Big-Shot Lawyer',
    { hair: 'buzz', hairColor: 0, skin: 5, height: 0.6, eyewear: 'shades', neck: 'tie', holding: 'briefcase' },
    'Argues every call. Has a business card that says "Attorney at Lawn."',
    ['OBJECTION!', 'My client was SAFE.', 'I rest my case.', 'You\'ll hear from my mom.']),
  K('pepper', 'Pepper', 'Lin', 'Pepper', 9, 'R', 'R', [5, 2, 7, 6, 6, 8, 9], ['changeup', 'fastball', 'curve'], 'loopy', 'The Auctioneer',
    { hair: 'bob', hairColor: 0, skin: 1, height: 0.3, mouth: 'open', hat: 'cowboy', holding: 'gavel' },
    'Talks so fast while she pitches that batters forget to swing. Accidentally sold Hank\'s bike.',
    ['Do-I-hear-strike-one-strike-one-STRIKE!', 'SOLD!', 'Going once, going twice!', 'Fifty-fifty-who\'ll-give-sixty!']),
  K('hank', 'Hank', 'Bristow', 'Tank', 12, 'R', 'R', [7, 10, 2, 5, 7, 4, 3], ['fastball', 'sinker'], 'moonshot', 'The Mall Security Guard',
    { hair: 'buzz', hairColor: 2, skin: 2, height: 0.9, build: 1, head: 'wide', body: 'badge', extra: 'earpiece', face: 'mustache' },
    'Biggest kid in the league. Gentlest kid in the league. Asks runners to please walk.',
    ['Sir, this is a no-running zone.', 'Move along, folks.', 'I\'m gonna need to see a receipt.', 'Code red at second!']),
];

export const KID_BY_ID: Record<string, Kid> = Object.fromEntries(KIDS.map((k) => [k.id, k]));

export function kid(id: string): Kid {
  const k = KID_BY_ID[id];
  if (!k) throw new Error(`unknown kid ${id}`);
  return k;
}

export const displayName = (k: Kid) => k.nick || k.first;
export const fullName = (k: Kid) => `${k.first} "${k.nick}" ${k.last}`;
