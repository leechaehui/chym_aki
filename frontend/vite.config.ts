import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

// 상대경로(base:'./')로 빌드 → 산출물(dist)을 어느 경로/오프라인에서 열어도 동작.
// @/* 별칭으로 src 절대경로 임포트 → 깊은 상대경로(../../..) 제거(유지보수성).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
  server: {
    host: true,
    port: 5174,
    open: true,
    // [성능] 프록시 타겟은 반드시 127.0.0.1 (localhost 금지).
    // Windows에서 localhost는 IPv6 ::1을 먼저 시도 → 서버는 IPv4만 듣기에 매 요청 ~220ms 지연.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8010",
        changeOrigin: true,
        ws: true,
        headers: { Origin: "http://127.0.0.1:5174" },
      },
      "/uploads": {
        target: "http://127.0.0.1:8010",
        changeOrigin: true,
      },
      // [중요] WSI/PACS 는 별도 프로세스인 8001(wsi_main) 에서 제공한다. 8010(메인 백엔드) 아님.
      //        /wsi-api/* → 8001 로 프록시(접두어 제거). 머지 시 8010/api/wsi 로 바꾸지 말 것.
      "/wsi-api": {
        target: "http://127.0.0.1:8001",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/wsi-api/, ""),
      },
    },
  },
  base: "./",
});
