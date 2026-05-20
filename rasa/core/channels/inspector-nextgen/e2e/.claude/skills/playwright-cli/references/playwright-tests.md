# Running & debugging tests in inspector e2e

**How-to-invoke** reference. Authoring rules: [CLAUDE.md](../../../../CLAUDE.md). Live CLI exploration: [SKILL.md](../SKILL.md). CLI setup: [cli-installation.md](cli-installation.md).

Work from `rasa/core/channels/inspector-nextgen/e2e/` unless noted. The Rasa inspector must be reachable at `BASE_URL` (default `http://localhost:5005`).

## Prefer the yarn scripts

Defined in [`package.json`](../../../../package.json):

| Goal | Command |
| --- | --- |
| Full run (start finance + tests) | `yarn e2e` |
| Tests only (server already on 5005) | `yarn e2e:only` |
| One file | `yarn e2e:only tests/home.test.ts` |
| Trace UI | `yarn e2e:watch` |
| Playwright inspector (headed) | `yarn e2e:debug` |
| Repeat 10× (flake repro) | `yarn e2e:repeat` or `yarn e2e:repeat tests/home.test.ts` |
| Open last HTML report | `yarn report` |

From repo root with venv active, same commands after `cd rasa/core/channels/inspector-nextgen/e2e`.

Override server URL:

```bash
BASE_URL=http://localhost:5005 yarn e2e:only
```

## Suppressing the auto-opened HTML report

```bash
PLAYWRIGHT_HTML_OPEN=never yarn e2e:only tests/home.test.ts
```

Use this in debug loops or with `--debug=cli`.

## `--debug=cli`: attach to the real test session

`--debug=cli` pauses the test and prints a session name. Attach with the CLI in another terminal to inspect the **same** page state as the test — including `inspectorPage` / `inspectorOptions` setup.

```bash
# Terminal 1 — Rasa on 5005, then:
cd rasa/core/channels/inspector-nextgen/e2e
PLAYWRIGHT_HTML_OPEN=never yarn e2e:only tests/home.test.ts --debug=cli

# Watch for "Debugging Instructions" and a session name like tw-abc123.

# Terminal 2
playwright-cli attach tw-abc123
playwright-cli snapshot
playwright-cli console
playwright-cli network
```

Edit triads/flows based on findings, then resume or re-run. Do not paste CLI click sequences into `*.test.ts`.

Prefer attach over opening a fresh `playwright-cli open` session when the failure depends on fixture setup (inspect mode, query params, finance demo state).

## Reading a failure without the CLI

Cheapest first move:

```bash
yarn report
```

[`playwright.config.ts`](../../../../playwright.config.ts) records traces (on locally, retain-on-failure in CI). Artifacts live under `test-results/`.

```bash
npx playwright show-trace test-results/.../trace.zip
```

## What not to do

- Don't run `npx playwright test` outside this directory — you lose `playwright.config.ts` and path aliases.
- Don't add `--debug=cli` to committed scripts or CI.
- Don't use codegen output as committed tests — use `generate-locator` for individual locators only.
- Don't skip `source .venv/bin/activate` + finance train when starting Rasa via `yarn e2e` — see [README.md](../../../../README.md#agent--ci-environment).
