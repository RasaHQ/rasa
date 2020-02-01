# Rasa OSS

[![Join the chat on Rasa Community Forum](https://img.shields.io/badge/forum-join%20discussions-brightgreen.svg)](https://forum.rasa.com/?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![PyPI version](https://badge.fury.io/py/rasa.svg)](https://badge.fury.io/py/rasa)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/rasa.svg)](https://pypi.python.org/pypi/rasa)
[![Build Status](https://travis-ci.com/RasaHQ/rasa.svg?branch=master)](https://travis-ci.com/RasaHQ/rasa)
[![Coverage Status](https://coveralls.io/repos/github/RasaHQ/rasa/badge.svg?branch=master)](https://coveralls.io/github/RasaHQ/rasa?branch=master)
[![Documentation Status](https://img.shields.io/badge/docs-stable-brightgreen.svg)](https://rasa.com/docs)
[![FOSSA Status](https://app.fossa.com/api/projects/custom%2B8141%2Fgit%40github.com%3ARasaHQ%2Frasa.git.svg?type=shield)](https://app.fossa.com/projects/custom%2B8141%2Fgit%40github.com%3ARasaHQ%2Frasa.git?ref=badge_shield)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/orgs/RasaHQ/projects/23)

<img align="right" height="244" src="https://www.rasa.com/assets/img/sara/sara-open-source-lg.png">

Rasa is an open source machine learning framework to automate text-and voice-based conversations. With Rasa, you can build contexual assistants on:
- Facebook Messenger
- Slack
- Microsoft Bot Framework
- Rocket.Chat
- Mattermost
- Telegram
- Twilio
- Your own custom conversational channels

or voice assistants as:
- Alexa Skills
- Google Home Actions

Rasa helps you build contextual assistants capable of having layered conversations with 
lots of back-and-forth. In order for a human to have a meaningful exchange with a contextual 
assistant, the assistant needs to be able to use context to build on things that were previously 
discussed – Rasa enables you to build assistants that can do this in a scalable way.

There's a lot more background information in this
[blog post](https://medium.com/rasa-blog/a-new-approach-to-conversational-software-2e64a5d05f2a).

---
- **What does Rasa do? 🤔**
  [Check out our Website](https://rasa.com/)

- **I'm new to Rasa 😄**
  [Get Started with Rasa](https://rasa.com/docs/getting-started/)

- **I'd like to read the detailed docs 🤓**
  [Read The Docs](https://rasa.com/docs/)

- **I'm ready to install Rasa 🚀**
  [Installation](https://rasa.com/docs/rasa/user-guide/installation/)

- **I want to learn how to use Rasa 🚀**
  [Tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/)

- **I have a question ❓**
  [Rasa Community Forum](https://forum.rasa.com/)

- **I would like to contribute 🤗**
  [How to Contribute](#how-to-contribute)

---  
## Where to get help

There is extensive documentation in the [Rasa Docs](https://rasa.com/docs/rasa).
Make sure to select the correct version so you are looking at
the docs for the version you installed.

Please use [Rasa Community Forum](https://forum.rasa.com) for quick answers to
questions.

### README Contents:
- [How to contribute](#how-to-contribute)
- [Development Internals](#development-internals)
- [License](#license)

### How to contribute
We are very happy to receive and merge your contributions. You can
find more information about how to contribute to Rasa (in lots of
different ways!) [here](http://rasa.com/community/contribute).

To contribute via pull request, follow these steps:

1. Create an issue describing the feature you want to work on (or
   have a look at the [contributor board](https://github.com/orgs/RasaHQ/projects/23))
2. Write your code, tests and documentation, and format them with ``black``
3. Create a pull request describing your changes

Your pull request will be reviewed by a maintainer, who will get
back to you about any necessary changes or questions. You will
also be asked to sign a
[Contributor License Agreement](https://cla-assistant.io/RasaHQ/rasa).


## Development Internals
### Running and changing the documentation
To build & edit the docs, first install all necessary dependencies:

```
pip3 install -r requirements-dev.txt
pip3 install -r requirements-docs.txt
```

After the installation has finished, you can run and view the documentation
locally using:
```
make livedocs
```

Visit the local version of the docs at http://localhost:8000 in your browser.
You can now change the docs locally and the web page will automatically reload
and apply your changes.

### Running the Tests
In order to run the tests, make sure that you have the development requirements installed:
```bash
export PIP_USE_PEP517=false
pip3 install -r requirements-dev.txt
pip3 install -e .
make prepare-tests-ubuntu # Only on Ubuntu and Debian based systems
make prepare-tests-macos  # Only on macOS
```

Then, run the tests:
```bash
make test
```

They can also be run at multiple jobs to save some time:
```bash
JOBS=[n] make test
```

Where `[n]` is the number of jobs desired. If omitted, `[n]` will be automatically chosen by pytest.

### Steps to release a new version
Releasing a new version is quite simple, as the packages are build and distributed by travis.

*Terminology*:
* patch release (third version part increases): 1.1.2 -> 1.1.3
* minor release (second version part increases): 1.1.3 -> 1.2.0
* major release (first version part increases): 1.2.0 -> 2.0.0

*Release steps*:
1. Make sure all dependencies are up to date (**especially Rasa SDK**)
2. Switch to the branch you want to cut the release from (`master` in case of a major / minor, the current feature branch for patch releases) 
3. Run `make release`
4. Create a PR against master or the release branch (e.g. `1.2.x`)
5. Once your PR is merged, tag a new release (this SHOULD always happen on master or release branches), e.g. using
    ```bash
    git tag 1.2.0 -m "next release"
    git push origin 1.2.0
    ```
    travis will build this tag and push a package to [pypi](https://pypi.python.org/pypi/rasa)
6. **If this is a minor release**, a new release branch should be created pointing to the same commit as the tag to allow for future patch releases, e.g.
    ```bash
    git checkout -b 1.2.x
    git push origin 1.2.x
    ```

### Code Style

To ensure a standardized code style we use the formatter [black](https://github.com/ambv/black).
To ensure our type annotations are correct we use the type checker [pytype](https://github.com/google/pytype). 
If your code is not formatted properly or doesn't type check, travis will fail to build.

#### Formatting

If you want to automatically format your code on every commit, you can use [pre-commit](https://pre-commit.com/).
Just install it via `pip install pre-commit` and execute `pre-commit install` in the root folder.
This will add a hook to the repository, which reformats files on every commit.

If you want to set it up manually, install black via `pip install -r requirements-dev.txt`.
To reformat files execute
```
make formatter
```

#### Type Checking

If you want to check types on the codebase, install `pytype` using `pip install -r requirements-dev.txt`.
To check the types execute
```
make types
```

### Deploying documentation updates

We use `sphinx-versioning` to build docs for tagged versions and for the master branch.
The static site that gets built is pushed to the `docs` branch of this repo, which doesn't contain
any code, only the site.

We host the site on netlify. On master branch builds (see `.travis.yml`), we push the built docs to the `docs`
branch. Netlify automatically re-deploys the docs pages whenever there is a change to that branch.


## License
Licensed under the Apache License, Version 2.0.
Copyright 2020 Rasa Technologies GmbH. [Copy of the license](LICENSE.txt).

A list of the Licenses of the dependencies of the project can be found at
the bottom of the
[Libraries Summary](https://libraries.io/github/RasaHQ/rasa).
