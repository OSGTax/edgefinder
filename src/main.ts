const root = document.getElementById('app')!;
const hash = new URLSearchParams(location.hash.slice(1));

if (import.meta.env.DEV && hash.has('gallery')) {
  import('./dev/gallery3d').then((m) => m.devGallery(root, hash));
} else if (import.meta.env.DEV && hash.has('dev')) {
  import('./dev/view3d').then((m) => m.devView(root, hash.get('dev') || 'field'));
} else if (hash.has('sounds')) {
  // the sound board (src/dev/soundboard.ts): kept in production builds so it works on a phone
  import('./dev/soundboard').then((m) => m.soundBoard(root));
} else {
  import('./ui/app').then((m) => m.startApp(root));
}
