# Installing Playwright CLI (global)

**Single source of truth** for the Anthropic Playwright CLI in this suite: global install, workspace constraints, upgrades.

The `playwright-cli` binary comes from **`@playwright/cli`**. That is separate from `npx playwright …`, which uses **`@playwright/test`** from this directory's `package.json`.

## Required setup

Install **globally** so it never enters this workspace's dependency graph:

```bash
npm install -g @playwright/cli
```

Verify:

```bash
playwright-cli --help
```

## Do not add `@playwright/cli` to `package.json`

Declaring **`@playwright/cli`** as a dependency or devDependency here can pull overlapping Playwright packages and cause version conflicts with **`@playwright/test`**. Interactive debugging stays **outside** the install tree via global CLI only.

## Upgrading

```bash
npm update -g @playwright/cli
```

## Related

| Topic | Doc |
| ----- | --- |
| Exploration workflows (`playwright-cli` commands) | [SKILL.md](../SKILL.md) |
| Yarn scripts, `--debug=cli`, attaching to a paused test | [playwright-tests.md](playwright-tests.md) |
| E2E authoring conventions | [CLAUDE.md](../../../../CLAUDE.md) |
