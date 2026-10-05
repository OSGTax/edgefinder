const root = document.getElementById('app')!;
const hash = new URLSearchParams(location.hash.slice(1));

if (import.meta.env.DEV && hash.has('dev')) {
  import('./dev/view3d').then((m) => m.devView(root, hash.get('dev') || 'field'));
} else {
  root.textContent = 'Loading…';
}
