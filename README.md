# Rasa (Rasa Core + Rasa NLU)

[![Join the chat on Rasa Community Forum](https://img.shields.io/badge/forum-join%20discussions-brightgreen.svg)](https://forum.rasa.com/?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![PyPI version](https://badge.fury.io/py/rasa-nlu.svg)](https://badge.fury.io/py/rasa-nlu)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/rasa_nlu.svg)](https://pypi.python.org/pypi/rasa_nlu)
[![Build Status](https://travis-ci.com/RasaHQ/rasa.svg?branch=master)](https://travis-ci.com/RasaHQ/rasa)
[![Coverage Status](https://coveralls.io/repos/github/RasaHQ/rasa/badge.svg?branch=master)](https://coveralls.io/github/RasaHQ/rasa?branch=master)
[![Documentation Status](https://img.shields.io/badge/docs-stable-brightgreen.svg)](https://rasa.com/docs)
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FRasaHQ%2Frasa_nlu.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2FRasaHQ%2Frasa_nlu?ref=badge_shield)

<img align="right" height="244" src="https://www.rasa.com/assets/img/sara/sara-open-source-lg.png">

**Note** This repository now contains the code for both Rasa NLU **AND** Rasa Core.
Nothing has changed yet in terms of usage, but we are in the process of simplifying
everything ahead of the next major release. 

Rasa is a framework for building conversational software, which includes
chatbots on:
- Facebook Messenger
- Slack
- Microsoft Bot Framework
- Rocket.Chat
- Mattermost
- Telegram
- Twilio

But you can also build assistants using:
- Alexa Skills
- Google Home Actions

Rasa's primary purpose is to help you build contextual, layered
conversations with lots of back-and-forth. To have a real conversation,
you need to have some memory and build on things that were said earlier.
Rasa lets you do that in a scalable way.

There's a lot more background information in this
[blog post](https://medium.com/rasa-blog/a-new-approach-to-conversational-software-2e64a5d05f2a).

---
- **What do Rasa Core & NLU do? 🤔**
  [Read About Rasa](https://rasa.com/products/rasa-stack/)

- **I'd like to read the detailed docs 🤓**
  [Read The Docs](https://rasa.com/docs)

- **I'm ready to install Rasa! 🚀**
  [Installation](https://rasa.com/docs/core/installation/)

- **I have a question ❓**
  [Rasa Community Forum](https://forum.rasa.com)

- **I would like to contribute 🤗**
  [How to contribute](#how-to-contribute)
  
---  
## Where to get help

There is extensive documentation at [Rasa Docs](https://rasa.com/docs). 
Make sure to select the correct version so you are looking at 
the docs for the version you installed.

Please use [Rasa Community Forum](https://forum.rasa.com) for quick answers to
questions.

### README Contents:
- [How to contribute](#how-to-contribute)
- [Development Internals](#development-internals)
- [License](#license)

### How to contribute
We are very happy to receive and merge your contributions. There is
some more information about the style of the code and docs in the
[documentation](https://rasa.com/docs/contributing/).

In general the process is rather simple:
1. create an issue describing the feature you want to work on (or
   have a look at issues with the label
   [help wanted](https://github.com/RasaHQ/rasa/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22))
2. write your code, tests and documentation
3. create a pull request describing your changes

Your pull request will be reviewed by a maintainer, who might get
back to you about any necessary changes or questions. You will
also be asked to sign a
[Contributor License Agreement](https://cla-assistant.io/RasaHQ/rasa).


## Development Internals
### Running and changing the documentation
To build & edit the docs, first install all necessary dependencies:

```
brew install sphinx
pip3 install -r requirements-dev.txt
```

After the installation has finished, you can run and view the documentation
locally using:
```
make livedocs-nlu
make livedocs-core
```

Visit the local version of the docs at http://localhost:8000 in your browser.
You can now change the docs locally and the web page will automatically reload
and apply your changes.

### Running the Tests
In order to run the tests make sure that you have the development requirements installed.
```
make test
```

### Steps to release a new version
Releasing a new version is quite simple, as the packages are build and distributed by travis. The following things need to be done to release a new version
1. update [rasa/version.py](https://github.com/RasaHQ/rasa/blob/master/rasa/version.py) to reflect the correct version number
2. edit the [CHANGELOG.rst](https://github.com/RasaHQ/rasa/blob/master/CHANGELOG.rst), create a new section for the release (eg by moving the items from the collected master section) and create a new master logging section
3. edit the [migration guide](https://github.com/RasaHQ/rasa/blob/master/docs/migrations.rst) to provide assistance for users updating to the new version 
4. commit all the above changes and tag a new release, e.g. using 
    ```
    git tag -f 0.7.0 -m "Some helpful line describing the release"
    git push origin 0.7.0
    ```
    travis will build this tag and push a package to [pypi](https://pypi.python.org/pypi/rasa)
5. only if it is a **major release**, a new branch should be created pointing to the same commit as the tag to allow for future minor patches, e.g.
    ```
    git checkout -b 0.7.x
    git push origin 0.7.x
    ```

### Code Style

To ensure a standardized code style we use the formatter [black](https://github.com/ambv/black).
If your code is not formatted properly, travis will fail to build.

If you want to automatically format your code on every commit, you can use [pre-commit](https://pre-commit.com/).
Just install it via `pip install pre-commit` and execute `pre-commit install` in the root folder.
This will add a hook to the repository, which reformats files on every commit.

If you want to set it up manually, install black via `pip install black`.
To reformat files execute
```
black .
```

## License
Licensed under the Apache License, Version 2.0.
Copyright 2019 Rasa Technologies GmbH. [Copy of the license](LICENSE.txt).

A list of the Licenses of the dependencies of the project can be found at
the bottom of the
[Libraries Summary](https://libraries.io/github/RasaHQ/rasa).
