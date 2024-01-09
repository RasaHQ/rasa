# Rasa Inspector

A rapid prototyping tool for enterprise developer personas working in Rasa Pro built with React + Typescript + Vite.

## How to run

### Prerequisites

- Node JS v.16.x.x - Check how to install `node` [here](https://nodejs.org/en/learn/getting-started/how-to-install-nodejs). Recommended using [`nvm`](https://github.com/nvm-sh/nvm).
- `yarn` version 1.x - Check how to install `yarn` [here](https://classic.yarnpkg.com/lang/en/docs/install/).
- a running version of `rasa-plus`
- a running action server (optional)

### Installation

`cd rasa_plus/channels/inspector` and then `yarn install`

### Running the stack

- install a local version of the rasa-plus package
- run `yarn build:watch` in the `/inspector` folder (this will watch over the React project and re-build it when necessary)
- run `rasa inspect`
- head to [http://localhost:5005/webhooks/inspector/inspect.html](http://localhost:5005/webhooks/inspector/inspect.html) - The page needs to be manually refreshed at every build
- engage with the assistant

### Building the project

The project can be built using the `yarn build` command. It will compile the code in the `/dist` folder.
It is important to know that _without_ it, the page that will be served can:
- return a 500 if the project has never been compiled before
- show an outdated version if the project has been compiled before the changes we want to introduce