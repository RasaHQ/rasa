# Rasa Inspector (New)

A rapid prototyping tool for enterprise developer personas working in Rasa Pro, built with React + Typescript + Vite.

Important: This inspector app supersedes the old inspector UI. Please use this when developing or extending the Inspector frontend. Changes here affect what is served with `rasa inspect`.

---

## Prerequisites

- **Node.js**: Version **24** (pinned in `.nvmrc`; same major as the [hello](https://github.com/RasaHQ/hello) repo).
  - [How to install Node.js](https://nodejs.org/en/learn/getting-started/how-to-install-nodejs)
  - Recommended: [nvm](https://github.com/nvm-sh/nvm). From this directory, run **`nvm use`** so your shell uses the Node version in `.nvmrc` (install it first with `nvm install` if needed).
- **Yarn 4** via **Corepack** (bundled with Node). Run `corepack enable` once on your machine; the exact Yarn version comes from `packageManager` in `package.json`.
- a `rasa` executable
- a running action server (optional, for advanced use)

### Font Awesome authentication

Private `@fortawesome/pro-*` packages need a Font Awesome npm token ([1Password](https://start.1password.com/open/i?a=W24ZYFDBGZGZHLMYCC2N4UXVNU&v=xc7iwzjpenftpm2g3mhh4iwdoi&i=nm7y66c6v5fmlj4kqrf3623vei&h=team-rasa-1password-com.1password.com)).

**Local development:** copy `.env.example` to **`.env`** in this directory (gitignored) and set `FONTAWESOME_NPM_AUTH_TOKEN`. Yarn loads `.env` via `injectEnvironmentFiles` in `.yarnrc.yml` — do not commit `.env`.

**CI:** the same variable is set from GitHub Actions secrets (`FONTAWESOME_NPM_AUTH_TOKEN`), not from a file.

---

## Installation

In the `rasa/core/channels/inspector-nextgen` directory, ensure **Node 24** is active (see `.nvmrc`). With [nvm](https://github.com/nvm-sh/nvm), run **`nvm use`** here whenever you open a new shell (`nvm install` first if that version is not installed yet). Then:

```
nvm use
corepack enable
yarn install
```

If you do not use nvm, use another tool or install Node 24 manually, then run `corepack enable` and `yarn install`.

`corepack enable` is only needed once per machine.

Repository pre-commit hooks run `yarn lint`, `yarn test`, and `yarn build` here; use **`nvm use`** (if applicable), then `yarn install`, so those commands run on Node 24 with Yarn 4.

## Development Workflow

Start the Vite dev server to enable hot reloading for the Inspector:

```
yarn dev
```

In another terminal, start the Rasa Inspector server, specifying the port used by Vite:

```
RASA_INSPECTOR_DEV_PORT=<port-of-the-served-vite-app> rasa inspect --nextgen
```

Visit `http://localhost:<port-of-the-served-vite-app>/` for the Vite app, or use the Rasa Inspector at the link printed by the CLI. The Inspector frontend will live-reload on changes.

---

## Building and Running for Production

To build the frontend for production (output to `dist/`):

```
yarn build
```

To serve the production Inspector, run:

```
rasa inspect --nextgen
```

This will serve the built frontend (`dist/index.html`).

---

## Versioning and release

The Inspector frontend is published as an NPM package to **GCP Artifact Registry**. Versioning and publishing are handled by [`.github/workflows/publish-rasa-inspector-npm.yml`](../../../.github/workflows/publish-rasa-inspector-npm.yml).

### When the workflow runs

- **Push to `main` or `3.*` branches** — publishes a preview build.
- **Push of a version tag** — publishes a release or prerelease (tag pattern: `X.Y.Z` with optional suffix, e.g. `3.16.0`, `3.16.1-a6`, `3.16.1.dev20260209`).
- **Pull requests** — publishes a PR preview when the PR has label `hello-rasa` or `studio`.

### How versions and dist-tags are set

| Trigger / tag form                            | NPM version                         | NPM dist-tag                   |
| --------------------------------------------- | ----------------------------------- | ------------------------------ |
| Git tag `3.16.0` (exact release)              | `3.16.0`                            | `latest`                       |
| Git tag `3.16.1-a6`, `3.16.1-alpha.6`         | as-is                               | `alpha`                        |
| Git tag `3.16.1-b1`, `3.16.1-beta.1`          | as-is                               | `beta`                         |
| Git tag `3.16.1-rc1`                          | as-is                               | `rc`                           |
| Git tag `3.16.1-dev.0`, `3.16.1-dev.20260209` | as-is                               | `dev`                          |
| Git tag `3.16.1a6` (Python-style)             | normalized to `3.16.1-a6`           | `alpha`                        |
| Git tag `3.16.1.dev20260209` (PEP 440 dev)    | normalized to `3.16.1-dev.20260209` | `dev`                          |
| Push to `main`                                | `0.0.0-main-<sha>`                  | `main`                         |
| Push to `3.16.x`                              | `0.0.0-3.16.x-<sha>`                | `branch-3.16.x`                |
| PR (labeled)                                  | `0.0.0-pr-<number>-<sha>`           | `pr-<number>` (e.g. `pr-4604`) |

Python-style tags (PEP 440) are normalized to semver before publish (e.g. `3.16.1a6` → `3.16.1-a6`, `3.16.1.dev20260209` → `3.16.1-dev.20260209`).

### Installing a specific version from Artifact Registry

To install `@rasahq/rasa-inspector` from GCP Artifact Registry in your own project, configure npm to use the registry for the `@rasahq` scope and authenticate.

#### One-time: set up the registry and GCP token

1. **Authenticate with Google Cloud** (if not already done):

   ```bash
   gcloud auth login
   ```

2. **Set the registry for the `@rasahq` scope** in your project's `.npmrc` (registry URL only, no credentials):

   ```
   @rasahq:registry=https://europe-west3-npm.pkg.dev/rasa-releases/rasa-inspector/
   ```

3. **Export a GCP access token** as an environment variable before installing:

   ```bash
   export NPM_CONFIG_//europe-west3-npm.pkg.dev/rasa-releases/rasa-inspector/:_authToken=$(gcloud auth print-access-token)
   ```

   Add this to your shell profile (e.g. `~/.zshrc` or `~/.bashrc`), or re-run it when the token expires (tokens are valid for ~1 hour).

   See [Artifact Registry Node.js authentication](https://docs.cloud.google.com/artifact-registry/docs/nodejs/authentication) for CI/service account options.

#### Install the package

Install by dist-tag or by exact version:

```bash
npm install @rasahq/rasa-inspector@latest
# or
npm install @rasahq/rasa-inspector@dev
npm install @rasahq/rasa-inspector@3.16.1-a6
```
