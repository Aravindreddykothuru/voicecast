// Browser end-to-end tests against a RUNNING backend (API + workers, see
// sur-backend/scripts/run-local.ps1). Playwright starts the Vite dev server
// itself, so `npm run test:e2e` is also the cold-start check that the UI
// comes up on this machine and can reach the API.
//
//   E2E_VIDEO=path\to\clip.mp4 npm run test:e2e            full dub via the UI
//   E2E_STALL=1 npm run test:e2e -- stall.spec.ts          with workers stopped
import { defineConfig, devices } from "@playwright/test";

const PORT = Number(process.env.E2E_PORT ?? 8443);

export default defineConfig({
  testDir: "./e2e",
  timeout: 90 * 60 * 1000, // a CPU dub takes minutes, voice cloning much longer
  expect: { timeout: 15_000 },
  retries: 0,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: `http://localhost:${PORT}`,
    ...devices["Desktop Chrome"],
    trace: "retain-on-failure",
  },
  webServer: {
    command: `npx vite --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: true,
    timeout: 120_000,
    env: { VITE_API_BASE_URL: process.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000" },
  },
});
