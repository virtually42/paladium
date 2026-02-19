import { defineConfig } from 'vite';

export default defineConfig({
  resolve: {
    alias: {
      // Point to Mill's Scala.js output
      'scalajs': '../../out/web-frontend/fastLinkJS.dest'
    }
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8080'
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
});
