# Rasa Inspector

A rapid prototyping tool for enterprise developer personas working in Rasa Pro built with React + Typescript + Vite.

## How to run

### Prerequisites

- **Node.js**: Version **greater than 18.x.x and less than 20.x.x**.
    - [How to install Node.js](https://nodejs.org/en/learn/getting-started/how-to-install-nodejs)
    - Recommended: Use [nvm](https://github.com/nvm-sh/nvm) for version management.
- `yarn` version 1.x - Check how to install `yarn` [here](https://classic.yarnpkg.com/lang/en/docs/install/).
- a running version of `rasa`
- a running action server (optional)

### Installation

`cd rasa/core/channels/inspector` and then `yarn install`

### Running the stack

- install a local version of the rasa-pro package
- run `yarn build:watch` in the `/inspector` folder (this will watch over the React project and re-build it when
  necessary)
- run `rasa inspect`
- head to [http://localhost:5005/webhooks/socketio/inspect.html](http://localhost:5005/webhooks/socketio/inspect.html) -
  The page needs to be manually refreshed at every build
- engage with the assistant

### Building the project

The project can be built using the `yarn build` command. It will compile the code in the `/dist` folder.
It is important to know that _without_ it, the page that will be served can:

- return a 500 if the project has never been compiled before
- show an outdated version if the project has been compiled before the changes we want to introduce

## Updating the Chat Widget

If updates or modifications need to be made to the chat widget, you’ll need to work within
the [rasa-x](https://github.com/RasaHQ/rasa-x) repository.

This is because the code within the [rasa-private](https://github.com/RasaHQ/rasa-private) repo for the chat widget is
minified in
the [rasa-chat.js](https://github.com/RasaHQ/rasa-private/blob/main/rasa/core/channels/inspector/assets/rasa-chat.js)
file, and direct edits cannot be effectively made there.

### Step-by-Step Guide

1. **Make updates in [rasa-x](https://github.com/RasaHQ/rasa-x) repository:**

    - Make the necessary code changes.
    - Update the **npm** package version. This is done in
      the [src/rasa-chat/package.json](https://github.com/RasaHQ/rasa-x/blob/main/src/rasa-chat/package.json) file.
    - Merge the changes to the lastest version branch or to the `main` branch.

2. **Trigger the release GitHub action:**

    - Navigate to
      the [publish-rasa-chat](https://github.com/RasaHQ/rasa-x/blob/main/.github/workflows/publish-rasa-chat.yml) GitHub
      action within the [rasa-x](https://github.com/RasaHQ/rasa-x) repo and initiate the workflow.

3. **Monitor Deployment to npm:**

    - After the GitHub action successfully runs, the updated package will be deployed to npm. Confirm the deployment by
      visiting the package page on npm at [@rasahq/rasa-chat](https://www.npmjs.com/package/@rasahq/rasa-chat/).

4. **Update the [rasa-private](https://github.com/RasaHQ/rasa-private) Repository:**
    - Download or copy the minified JavaScript from the `widget.js` file in the newly published npm package.
    - Replace the content in
      the [rasa-chat.js](https://github.com/RasaHQ/rasa-private/blob/main/rasa/core/channels/inspector/assets/rasa-chat.js)
      file with this new code to incorporate the updates.
