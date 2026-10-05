import { App } from './ui/app';

const root = document.getElementById('app')!;
const hash = new URLSearchParams(location.hash.slice(1));

// dev-only views for eyeballing the procedural art: #gallery, #scene=<yard>&cam=bat|field|overview
if (import.meta.env.DEV && hash.has('gallery')) import('./dev/gallery').then((m) => m.showGallery(root));
else if (import.meta.env.DEV && hash.has('scene')) import('./dev/scenes').then((m) => m.showScene(root, hash.get('scene')!, hash.get('cam') ?? 'bat'));
else new App(root);
