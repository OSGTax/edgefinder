import { ensureDefs, installTextures } from './ui/look';

const root = document.getElementById('app')!;
const hash = new URLSearchParams(location.hash.slice(1));

installTextures();
ensureDefs();

if (import.meta.env.DEV && hash.has('gallery')) {
  import('./dev/gallery3d').then((m) => m.devGallery(root, hash));
} else if (import.meta.env.DEV && hash.has('dev')) {
  import('./dev/view3d').then((m) => m.devView(root, hash.get('dev') || 'field'));
} else if (import.meta.env.DEV && hash.has('look')) {
  import('./ui/look/specimen').then((m) => m.specimen(root));
} else {
  import('./ui/app').then((m) => m.startApp(root));
}
