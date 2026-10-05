export type SfxName =
  | 'batCrack' // solid wooden-bat contact; opts.intensity 0..1 scales loudness/brightness (1 = home-run crush)
  | 'batTink' // weak/off-center contact, dull thud-tick
  | 'whiff' // swing and miss: airy whoosh
  | 'mittPop' // ball smacking into a catcher's mitt (deep leathery pop)
  | 'catch' // softer glove catch in the field
  | 'bounce' // ball bouncing on grass (soft thump); intensity scales
  | 'fence' // ball hitting a wooden/picket fence (hollow knock + rattle)
  | 'splash' // ball landing in a backyard pool
  | 'leaves' // ball crashing through tree leaves (rustle)
  | 'cheer' // small crowd of kids cheering (short, ~1.2s)
  | 'bigCheer' // bigger, longer cheer for home runs (~2.5s)
  | 'aww' // disappointed kid crowd groan
  | 'strike' // a playful "strike" stinger (two-note blip, not a voice)
  | 'out' // short descending "out" stinger
  | 'safe' // short bright "safe" stinger
  | 'homeRun' // celebratory jingle (~2s), original melody
  | 'special' // magical power-up whoosh/sparkle for activating a special move
  | 'uiTap' // menu button tap
  | 'uiBack' // menu back
  | 'uiSelect' // choosing something (a kid in a draft), slightly more emphatic than tap
  | 'whistle' // short slide-whistle with a pea-whistle trill, for inning change
  | 'dogBark' // a cartoon dog bark, used when a ball lands near a doghouse
  | 'throw'; // whoosh of a hard throw

export type MusicTrack = 'title' | 'game' | 'victory' | 'season';

export interface PlayOpts {
  /** 0..1, default 0.7. */
  intensity?: number;
  /** -1 (left) .. 1 (right), default 0. */
  pan?: number;
}
