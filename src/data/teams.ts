import type { Team } from './types';

export const TEAMS: Team[] = [
  {
    id: 'mudcats', street: 'Maple Street', name: 'Mudcats', abbr: 'MUD',
    colors: { primary: '#7a4a2a', secondary: '#f08a24', accent: '#fff3e0' },
    icon: 'mudcat', yardId: 'poolparty', division: 'Front Porch',
    roster: ['ines', 'bo', 'dez', 'priya', 'gus', 'toby', 'molly', 'wren', 'jun'],
  },
  {
    id: 'comets', street: 'Cedar Lane', name: 'Comets', abbr: 'COM',
    colors: { primary: '#23395d', secondary: '#6ec6ff', accent: '#eaf6ff' },
    icon: 'comet', yardId: 'poolparty', division: 'Front Porch',
    roster: ['kai', 'ezra', 'leo', 'darius', 'anya', 'ruby', 'hank', 'maya', 'pepper'],
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
