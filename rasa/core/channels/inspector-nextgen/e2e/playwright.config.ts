import { type PlaywrightTestConfig, devices } from "@playwright/test";
import { CI, BASE_URL } from "./config";

const config: PlaywrightTestConfig = {
  testDir: "./tests",
  /* Maximum time one test can run for. */
  timeout: 4 * 60 * 1000,
  expect: {
    /**
     * Maximum time expect() should wait for the condition to be met.
     * For example in `await expect(locator).toHaveText();`
     */
    timeout: 10000,
  },
  fullyParallel: !!CI,
  forbidOnly: !!CI,
  workers: "100%",
  reporter: [["html"], ["list"]],
  use: {
    actionTimeout: 60 * 1000,
    navigationTimeout: 60 * 1000,
    trace: CI ? "retain-on-failure" : "on",
    screenshot: "only-on-failure",
    baseURL: BASE_URL,
  },

  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        locale: "en-US",
        deviceScaleFactor: undefined,
        viewport: { width: 1240, height: 768 },
        launchOptions: {
          args: ["--start-maximized"],
        },
      },
    },
  ],

  outputDir: "test-results/",
};

export default config;
