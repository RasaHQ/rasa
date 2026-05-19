# E2E Testing for Rasa Inspector

This directory contains end-to-end tests for the Rasa development inspector UI using Playwright. The tests target the inspector served by the Rasa server at `http://localhost:5005/webhooks/inspector/inspect.html`.

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
- Rasa installed and on your `PATH` (e.g. via the repo’s virtualenv)
- Finance project model trained (see above)

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

## Running Tests

### Basic Commands

```bash
# Run all tests (starts Rasa inspector from finance project, then runs tests)
yarn e2e

# Run tests only (use when Rasa inspector is already running on port 5005)
yarn e2e:only

# Run tests with a live trace viewer
yarn e2e:watch

# Run tests in debug mode (opens browser)
yarn e2e:debug

# Run a specific test file (with server already running)
yarn e2e:only tests/inspect-page.test.ts

# Run tests in headed mode (visible browser)
yarn e2e --headed
```

### Test Reports

```bash
# View HTML report
yarn report
```

## Test Structure

```
e2e/
├── actions/           # Page object actions and locators
│   ├── inspector.actions.ts
│   └── index.ts
├── flows/             # Test flows and user journeys
│   ├── inspector.flow.ts
│   └── index.ts
├── tests/             # Test specifications
│   └── inspect-page.test.ts
├── config.ts          # Test configuration (BASE_URL, CI)
├── playwright.config.ts
└── package.json
```

### Key Concepts

- **Actions**: Reusable page interactions and assertions (e.g. navigate to inspect page, assert page loaded).
- **Flows**: Complete user journeys combining multiple actions.
- **Tests**: Test specifications that use flows and actions.
