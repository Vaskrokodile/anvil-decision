// Playwright browsers live on E: — C: is space-constrained.
process.env.PLAYWRIGHT_BROWSERS_PATH ??= 'E:/pw-browsers';

import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  use: { baseURL: 'http://localhost:5199' },
  webServer: {
    command: 'npm run dev -- --port 5199 --strictPort',
    url: 'http://localhost:5199',
    reuseExistingServer: true,
    timeout: 30_000,
  },
});
