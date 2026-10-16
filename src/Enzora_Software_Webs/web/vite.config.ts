import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

// Dev proxy target: the GPU-server backend (or a local one for backend authors).
// Set VITE_PROXY_TARGET in web/.env, e.g. VITE_PROXY_TARGET=http://192.168.1.50:8000
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const proxyTarget = env.VITE_PROXY_TARGET || "http://127.0.0.1:8000";
  return {
    base: env.VITE_BASE_PATH || "/",
    plugins: [react()],
    server: {
      host: true,
      fs: {
        allow: [
          fileURLToPath(new URL(".", import.meta.url)),
          fileURLToPath(new URL("../../../resource", import.meta.url)),
        ],
      },
      port: 5173,
      proxy: {
        "/api": { target: proxyTarget, changeOrigin: true },
      },
    },
    build: {
      outDir: "dist",
      // Plotly is fetched only when result charts are opened.
      chunkSizeWarningLimit: 5000,
    },
  };
});
