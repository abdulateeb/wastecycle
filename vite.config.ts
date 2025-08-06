import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: './',
  plugins: [react()],
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
  server: {
    port: 5000,
    host: true,
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'ui-libs': ['framer-motion', 'lucide-react'],
        },
      },
    },
    sourcemap: false,
    minify: 'terser',
  },
  define: {
    __DEV__: JSON.stringify(process.env.NODE_ENV === 'development'),
  },
});
