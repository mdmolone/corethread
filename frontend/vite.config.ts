import { defineConfig, type ProxyOptions } from "vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"
import path from "node:path"
import type { ClientRequest, IncomingMessage } from "node:http"

type ProxyServer = Parameters<NonNullable<ProxyOptions["configure"]>>[0]

const SSE_HEADER_HOOK = (proxy: ProxyServer) => {
  proxy.on("proxyReq", (proxyReq: ClientRequest, req: IncomingMessage) => {
    const accept = req.headers["accept"] ?? ""
    if (accept.includes("text/event-stream")) {
      proxyReq.setHeader("X-Accel-Buffering", "no")
    }
  })
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/v1": {
        target: "http://localhost:8000",
        changeOrigin: false,
        configure: SSE_HEADER_HOOK,
      },
      "/health": {
        target: "http://localhost:8000",
        changeOrigin: false,
        configure: SSE_HEADER_HOOK,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    target: "es2022",
  },
})
