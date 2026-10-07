import { defineConfig } from 'vitest/config';
import { pwa } from './build/pwa';

// `--mode pages` builds the playable site into docs/; `npm run deploy` copies it
// to the gh-pages branch, which GitHub Pages serves. The pwa() plugin adds the
// manifest, code-generated icons and the offline service worker (build/pwa.ts).
export default defineConfig(({ mode }) => ({
  base: './',
  plugins: [pwa()],
  build: {
    outDir: mode === 'pages' ? 'docs' : 'dist',
    emptyOutDir: true,
    target: 'es2020',
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks: (id) => (id.includes('node_modules/three') ? 'three' : undefined),
      },
    },
  },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
  },
}));
