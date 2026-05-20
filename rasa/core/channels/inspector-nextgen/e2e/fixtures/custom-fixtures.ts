import { mergeTests } from "@playwright/test";

import { inspectorFixture } from "./inspector-fixture";

export const test = mergeTests(inspectorFixture);
