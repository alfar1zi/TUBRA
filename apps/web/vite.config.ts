import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    // host:true so the dev server is reachable from other devices on the LAN
    // and from the Tailscale interface. HTTPS/certs are terminated by
    // `tailscale serve` (see scripts/serve_tailscale.sh), not by Vite, so the
    // origin the phone sees is a trusted *.ts.net cert and mic permission works.
    host: true,
    port: 5173,
    allowedHosts: [".ts.net"],
    proxy: {
      "/api": {
        // Local Flask by default. Point VITE_PROXY_TARGET at a deployed API to
        // exercise the full screen from the dev server when the model is not
        // installed locally (it only lives in the container image).
        target: process.env.VITE_PROXY_TARGET ?? "http://localhost:8080",
        changeOrigin: true,
        secure: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});