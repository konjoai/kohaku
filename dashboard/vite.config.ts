import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@konjoai/ui': '/Users/wesleyscholl/ui/packages/ui/src/index.ts' },
    dedupe: ['react', 'react-dom', 'motion', 'motion/react'],
  },
  server: {
    port: 5180,
    proxy: {
      '/api': { target: 'http://localhost:8002', changeOrigin: true },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: true,
  },
})
