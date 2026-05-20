# E2E Testing for Rasa Inspector

End-to-end tests for the Rasa development inspector UI using Playwright. Tests target the inspector served by the Rasa server at `http://localhost:5005/webhooks/inspector/inspect.html`.

For test authoring conventions, read [CLAUDE.md](CLAUDE.md). For worked examples, see [docs/authoring-tests.md](docs/authoring-tests.md).

## Default project

Tests use the **finance** project at `rasa/cli/project_templates/finance`. The `yarn e2e` script starts `rasa inspect` from that directory, runs the tests, then stops the server.

**One-time setup:** If the model has never been trained for the finance project, train it once before running the tests:

```bash
cd rasa/cli/project_templates/finance
rasa train
```

## Prerequisites

- Node.js (v18 or higher)
- Yarn package manager
- Rasa on `PATH` via `source <repo-root>/.venv/bin/activate` (see **Environment** below)
- Finance project model trained (see above)
- Voice API keys and voice configuration only if you want to run `tests/voice.test.ts` successfully. Without them the app shows `Voice isn't set up yet`.

## Environment

**CI** uses [`.github/actions/inspector-e2e-tests/action.yml`](../../../../../.github/actions/inspector-e2e-tests/action.yml): `source .venv/bin/activate` at the repo root, train finance, start `rasa inspect --port 5005`, then Playwright in Docker.

**Local / agents:** From repo root run `make install`, then `source .venv/bin/activate` before `yarn e2e` in this directory (`rasa` must be on `PATH`). If the inspector is already on port 5005, use `yarn e2e:only`.

## Setup

1. **Install dependencies:**

   ```bash
   yarn
   ```

2. **Install Playwright browsers:**

   ```bash
   npx playwright install
   ```

3. **Optional:** Set `BASE_URL` to override the server URL (e.g. if the Rasa server runs on a different host or port).

4. **Optional (agents / interactive debug):** Install [Playwright CLI](.claude/skills/playwright-cli/references/cli-installation.md) globally (`npm install -g @playwright/cli`). Do not add `@playwright/cli` to this package. See [`.claude/skills/playwright-cli/SKILL.md`](.claude/skills/playwright-cli/SKILL.md).

## Running tests

```bash
# Run all tests (starts Rasa inspector from finance project, then runs tests)
yarn e2e

# Run tests only (use when Rasa inspector is already running on port 5005)
yarn e2e:only

# Run tests with a live trace viewer
yarn e2e:watch

# Run tests in debug mode (opens browser)
yarn e2e:debug

# Repeat a test 10× to reproduce a flake (server already running)
yarn e2e:repeat tests/home.test.ts

# Run a specific test file (with server already running)
yarn e2e:only tests/home.test.ts

# Run tests in headed mode (visible browser)
yarn e2e --headed
```

## Reports

```bash
yarn report
```

## Layout

```
e2e/
├── actions/ui/    # UI triads (getLocators, actions, assertions)
├── fixtures/      # Playwright test harness and setup options
├── flows/         # Multi-step user journeys
├── tests/         # Test specifications (*.test.ts)
├── config.ts
├── playwright.config.ts
└── package.json
```

See [CLAUDE.md](CLAUDE.md) for layers, imports, flows, shared UI surfaces, and authoring rules.
