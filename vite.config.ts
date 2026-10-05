import { defineConfig } from 'vitest/config';

// `--mode pages` builds the playable site into docs/; `npm run deploy` copies it
// to the gh-pages branch, which GitHub Pages serves.
export default defineConfig(({ mode }) => ({
  base: './',
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
