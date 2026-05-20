/**
 * Barrel for the merged inspector Playwright harness only.
 *
 * Import `test` and `expect` from here in `*.test.ts` only.
 * In `flows/**`, import `test` from `@playwright/test` for `test.step(...)`.
 * Multi-step journeys live under `@e2e/flows`; UI interactions live under
 * `@e2e/ui-actions`.
 */

export { test } from "./custom-fixtures";
export { expect } from "@playwright/test";
