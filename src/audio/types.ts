export type SfxName =
  | 'batCrack' // solid wooden-bat contact; opts.intensity 0..1 scales loudness/brightness (1 = home-run crush)
  | 'batTink' // weak/off-center contact, dull thud with a stinging buzz
  | 'whiff' // swing and miss: airy whoosh
  | 'mittPop' // ball smacking into a catcher's mitt (deep leathery pop)
  | 'catch' // softer glove catch in the field
  | 'bounce' // ball bouncing on grass (soft thump); intensity scales
  | 'bounceDirt' // ball bouncing on the base paths (thud + grit)
  | 'bouncePatio' // ball bouncing on the pool deck (hard, bright tock)
  | 'fence' // ball hitting a solid wooden fence (hollow knock + rattle)
  | 'picket' // ball hitting the white picket fence (light clack, every slat chattering)
  | 'hedge' // ball into the hedge (thump swallowed by leaves)
  | 'houseWall' // ball off the back of the house (hollow bonk, window buzz)
  | 'splash' // ball landing in the backyard pool
  | 'leaves' // ball crashing through tree leaves (rustle)
  | 'cheer' // the crowd cheering; intensity = how big the moment is
  | 'bigCheer' // bigger, longer cheer for home runs (~2.5s)
  | 'aww' // disappointed groan; intensity = how much it hurts
  | 'ooh' // the crowd rising with a long fly ball
  | 'giggle' // two kids on the bench trying not to laugh (bobbles)
  | 'strike' // two notes up the toy xylophone
  | 'out' // three notes down the xylophone, mallet bouncing on the last
  | 'safe' // a bright glockenspiel run
  | 'homeRun' // slide whistle + kazoo-band fanfare (~2s), original melody
  | 'special' // slide whistle winding up, kazoo ta-da, sparkles: a special move
  | 'uiTap' // menu button tap (fingertip on cardboard)
  | 'uiBack' // menu back (poster board sliding away)
  | 'uiSelect' // choosing something (a bottle cap set down)
  | 'whistle' // Coach Toby's pea whistle, for inning change
  | 'dogBark' // a dog bark, used when a ball lands near a doghouse
  | 'screenDoor' // the back screen door creaking open and slapping shut
  | 'throw' // whoosh of a hard throw
  // cartoon accents, used sparingly for the comic moments
  | 'boing' // a door-stop spring
  | 'bonk' // wood-block knock + cartoon pitch drop (hit by pitch, a kid bumping into something)
  | 'zip' // zip-up whoosh (a kid taking off)
  | 'dizzy' // stars circling a kid's head
  | 'squeak' // sneakers skidding to a stop
  | 'pop'; // a cork pop, for a comic pop-up landing on screen

export type MusicTrack = 'title' | 'game' | 'inning' | 'victory' | 'defeat' | 'season';

export interface PlayOpts {
  /** 0..1, default 0.7. */
  intensity?: number;
  /** -1 (left) .. 1 (right), default 0. */
  pan?: number;
}
