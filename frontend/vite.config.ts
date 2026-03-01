import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },

  build: {
    target: 'es2020',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          deckgl: [
            'deck.gl',
            '@deck.gl/core',
            '@deck.gl/layers',
            '@deck.gl/aggregation-layers',
            '@deck.gl/mapbox',
          ],
          maplibre: ['maplibre-gl'],
          d3: ['d3'],
        },
      },
    },
  },

  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
