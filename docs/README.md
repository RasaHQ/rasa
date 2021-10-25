# Docs

The docs are built using [Docusaurus 2](https://v2.docusaurus.io/).

## Useful commands

### Installation

```
$ yarn
```

### Local Development

```
$ yarn start
```

This command starts a local development server and open up a browser window. Most changes are reflected live without having to restart the server.

### Build

```
$ yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

Deployment is handled by Netlify: it is setup for listening to changes on the `documentation` branch.


## Manual steps after a new version

When a new docs version has been released, we'll need to do the following manual steps:
- Remove all the callouts from previous versions, with the exception of experimental features. You can find
  those using `:::info` or `:::caution` in all the docs files.
- Update the wording of the top banner, configured in `docusaurus.config.js` in `announcementBar`: update the Rasa versions
  that are mentioned and link to the now previous major version documentation.
- Update Netlify redirects in `netflify.toml`, under `# Redirects for latest version permalinks`, by adjusting the
  version number to the now new major version.
