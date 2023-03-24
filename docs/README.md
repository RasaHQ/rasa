# Docs

The docs are built using [Docusaurus 2](https://v2.docusaurus.io/).
To run Docusaurus, install `Node.js 12.x`.

## Useful commands

### Installation
Firstly, install python dependencies for Rasa:

```
$ make install
```

Then, install doc dependencies:

```
$ make install-docs
```

### Local Development
In order to build the docs, run:

```
$ make docs
```

Then, start doc server in watch mode:

```
$ make livedocs
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
- Update Netlify redirects in `netlify.toml`, under `# Redirects for latest version permalinks`, by adjusting the
  version number to the now new major version.


## Handling deadlinks after removal of deprecated features

When removing deprecated features, it will happen that some links become dead because they now link to
parts of the docs that no longer exist. This usually happens in the CHANGELOG or migration links,
and thankfully we do have CI checks that alert for dead links.

The trick here is to make these links point to _previous_ versions of the docs. For instance, if the feature
you removed was documented at `./policies#mapping-policy` and the current latest version for the docs is `2.x`
(this also means that the next version is `3.x`), then you can update the link to `https://rasa.com/docs/rasa/2.x/policies#mapping-policy`.
