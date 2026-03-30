# Rasa Inspector (New)

A rapid prototyping tool for enterprise developer personas working in Rasa Pro, built with React + Typescript + Vite.

Important: This inspector app supersedes the old inspector UI. Please use this when developing or extending the Inspector frontend. Changes here affect what is served with `rasa inspect`.

---

## Prerequisites

- **Node.js**: Version **20.19.2**
  - [How to install Node.js](https://nodejs.org/en/learn/getting-started/how-to-install-nodejs)
  - Recommended: Use [nvm](https://github.com/nvm-sh/nvm) for version management.
- **yarn** version 1.x ([installation guide](https://classic.yarnpkg.com/lang/en/docs/install/))
- a `rasa` executable
- a running action server (optional, for advanced use)

### .npmrc requirement

You must create a `.npmrc` file (in the root or current directory). This is required for accessing private Font Awesome packages, without which dependencies will not install. **Do not commit this file!**

Example `.npmrc` (replace `TOKEN` with your actual font awesome token, available from [1Password](https://start.1password.com/open/i?a=W24ZYFDBGZGZHLMYCC2N4UXVNU&v=xc7iwzjpenftpm2g3mhh4iwdoi&i=nm7y66c6v5fmlj4kqrf3623vei&h=team-rasa-1password-com.1password.com)):

```
@fortawesome:registry=https://npm.fontawesome.com/
//npm.fontawesome.com/:_authToken=TOKEN
```

---

## Installation

In the `/rasa/core/channels/inspector-nextgen` directory:

```
yarn install
```

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

#### One-time: set up `.npmrc` for Artifact Registry

1. **Authenticate with Google Cloud** (if not already done):

   ```bash
   gcloud auth login
   ```

2. **Append the Artifact Registry npm config to your `.npmrc`** (run from your project root; this adds registry and auth settings for `@rasahq`):

   ```bash
   gcloud artifacts print-settings npm \
     --project=rasa-releases \
     --repository=rasa-inspector \
     --location=europe-west3 \
     --scope=@rasahq \
     >> .npmrc
   ```

   See [Artifact Registry Node.js authentication](https://docs.cloud.google.com/artifact-registry/docs/nodejs/authentication) for more options (e.g. CI with a service account).

3. **Refresh the registry token** so npm can access the registry:

   ```bash
   npx google-artifactregistry-auth --repo-config=./.npmrc --credential-config=./.npmrc
   ```

   Re-run this command if your credentials expire.

**Note:** Do not commit `.npmrc` if it contains credentials. Add `.npmrc` to `.gitignore` in projects where you use it.

#### Install the package

Install by dist-tag or by exact version:

```bash
npm install @rasahq/rasa-inspector@latest
# or
npm install @rasahq/rasa-inspector@dev
npm install @rasahq/rasa-inspector@3.16.1-a6
```
