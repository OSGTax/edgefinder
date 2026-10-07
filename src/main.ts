import { shell } from './shell';
import { ensureDefs, installTextures } from './ui/look';

const root = document.getElementById('app')!;
const hash = new URLSearchParams(location.hash.slice(1));

installTextures();
ensureDefs();
shell.init();

if (import.meta.env.DEV && hash.has('gallery')) {
  import('./dev/gallery3d').then((m) => m.devGallery(root, hash));
} else if (import.meta.env.DEV && hash.has('dev')) {
  import('./dev/view3d').then((m) => m.devView(root, hash.get('dev') || 'field'));
} else if (import.meta.env.DEV && hash.has('look')) {
  import('./ui/look/specimen').then((m) => m.specimen(root));
} else if (hash.has('sounds')) {
  // the sound board (src/dev/soundboard.ts): kept in production builds so it works on a phone
  import('./dev/soundboard').then((m) => m.soundBoard(root));
} else {
  import('./ui/app').then((m) => m.startApp(root));
}
