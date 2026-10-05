import type { Team } from './types';

export const TEAMS: Team[] = [
  {
    id: 'mudcats', street: 'Maple Street', name: 'Mudcats', abbr: 'MUD',
    colors: { primary: '#7a4a2a', secondary: '#f08a24', accent: '#fff3e0' },
    icon: 'mudcat', yardId: 'mudpuddle', division: 'Front Porch',
    roster: ['ines', 'bo', 'dez', 'priya', 'gus', 'toby', 'molly', 'wren', 'jun'],
  },
  {
    id: 'comets', street: 'Cedar Lane', name: 'Comets', abbr: 'COM',
    colors: { primary: '#23395d', secondary: '#6ec6ff', accent: '#eaf6ff' },
    icon: 'comet', yardId: 'poolparty', division: 'Front Porch',
    roster: ['kai', 'ezra', 'leo', 'darius', 'anya', 'ruby', 'hank', 'maya', 'pepper'],
  },
  {
    id: 'frogs', street: 'Willow Creek', name: 'Frogs', abbr: 'FRG',
    colors: { primary: '#5b3b8c', secondary: '#7ccf3f', accent: '#f1ffe6' },
    icon: 'frog', yardId: 'lilypad', division: 'Front Porch',
    roster: ['sora', 'ada', 'biglou', 'sammy', 'isla', 'nico', 'coop', 'fern', 'lola'],
  },
  {
    id: 'rockets', street: 'Elm Court', name: 'Rockets', abbr: 'RKT',
    colors: { primary: '#c8312f', secondary: '#c9d3dc', accent: '#fff0ef' },
    icon: 'rocket', yardId: 'junklot', division: 'Front Porch',
    roster: ['jett', 'wrench', 'brock', 'tamsin', 'omar', 'q', 'effie', 'mars', 'cyrus'],
  },
  {
    id: 'owls', street: 'Oak Hollow', name: 'Owls', abbr: 'OWL',
    colors: { primary: '#2f6d43', secondary: '#f2c14e', accent: '#fffbe8' },
    icon: 'owl', yardId: 'treehouse', division: 'Back Fence',
    roster: ['teddy', 'hazel', 'marcus', 'lulu', 'sadie', 'felix', 'rafi', 'ozzie', 'noodle'],
  },
  {
    id: 'pinecones', street: 'Pine Ridge', name: 'Pinecones', abbr: 'PIN',
    colors: { primary: '#1f5e4b', secondary: '#d9a35f', accent: '#fdf3e4' },
    icon: 'pinecone', yardId: 'grandmabea', division: 'Back Fence',
    roster: ['opal', 'benny', 'amari', 'clem', 'josie', 'tess', 'milo', 'ravi', 'winnie'],
  },
  {
    id: 'bumblebees', street: 'Birch Bay', name: 'Bumblebees', abbr: 'BEE',
    colors: { primary: '#2a2522', secondary: '#f5c518', accent: '#fff9db' },
    icon: 'bee', yardId: 'sandbox', division: 'Back Fence',
    roster: ['honey', 'wally', 'trudy', 'arlo', 'naomi', 'zeke', 'gideon', 'bianca', 'paloma'],
  },
  {
    id: 'sparks', street: 'Sunnyside', name: 'Sparks', abbr: 'SPK',
    colors: { primary: '#14857f', secondary: '#ffd23f', accent: '#e9fffd' },
    icon: 'lightning', yardId: 'sunflower', division: 'Back Fence',
    roster: ['june', 'tilly', 'ford', 'kofi', 'poppy', 'lucky', 'eli', 'dash', 'goldie'],
  },
];

export const TEAM_BY_ID: Record<string, Team> = Object.fromEntries(TEAMS.map((t) => [t.id, t]));

export function team(id: string): Team {
  const t = TEAM_BY_ID[id];
  if (!t) throw new Error(`unknown team ${id}`);
  return t;
}

export const teamName = (t: Team) => `${t.street} ${t.name}`;

/** The broadcast booth: two kids who think they're network announcers. */
export const ANNOUNCERS = {
  play: { name: 'Chet Valentine', blurb: 'Age 10. Hair gel, clip-on tie, toy microphone. Has been "in the business" since second grade.' },
  color: { name: 'Dottie Fairweather', blurb: 'Age 12. "Former big leaguer" (tee-ball, one season). Brings it up constantly.' },
};
