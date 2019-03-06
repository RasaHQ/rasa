# Rasa Core 

[![Join the chat on Rasa Community Forum](https://img.shields.io/badge/forum-join%20discussions-brightgreen.svg)](https://forum.rasa.com/?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![PyPI version](https://img.shields.io/pypi/v/rasa_core.svg)](https://pypi.python.org/pypi/rasa-core)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/rasa_core.svg)](https://pypi.python.org/pypi/rasa_core)
[![Build Status](https://travis-ci.com/RasaHQ/rasa_core.svg?branch=master)](https://travis-ci.com/RasaHQ/rasa_core)
[![Coverage Status](https://coveralls.io/repos/github/RasaHQ/rasa_core/badge.svg?branch=master)](https://coveralls.io/github/RasaHQ/rasa_core?branch=master)
[![Documentation Status](https://img.shields.io/badge/docs-stable-brightgreen.svg)](https://rasa.com/docs/core)

<img align="right" height="244" src="https://www.rasa.com/assets/img/sara/sara-open-source-lg.png">

Rasa Core is a framework for building conversational software, which includes
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

Rasa Core's primary purpose is to help you build contextual, layered
conversations with lots of back-and-forth. To have a real conversation,
you need to have some memory and build on things that were said earlier.
Rasa Core lets you do that in a scalable way.

There's a lot more background information in this
[blog post](https://medium.com/rasa-blog/a-new-approach-to-conversational-software-2e64a5d05f2a).

---
- **What do Rasa Core & NLU do? 🤔**
  [Read About the Rasa Stack](https://rasa.com/products/rasa-stack/)

- **I'd like to read the detailed docs 🤓**
  [Read The Docs](https://rasa.com/docs/core)

- **I'm ready to install Rasa Core! 🚀**
  [Installation](https://rasa.com/docs/core/installation/)

- **I'm ready to start building! 🤖**
  [Rasa Stack starter-pack](https://github.com/RasaHQ/starter-pack-rasa-stack)

- **I have a question ❓**
  [Rasa Community Forum](https://forum.rasa.com)

- **I would like to contribute 🤗**
  [How to contribute](#how-to-contribute)
  
---  
## Where to get help

There is extensive documentation:

- [master](https://rasa.com/docs/core/master/)&nbsp;
  (if you install from **github**) or
- [stable](https://rasa.com/docs/core)&nbsp;&nbsp;
  (if you install from **pypi**)


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
   [help wanted](https://github.com/RasaHQ/rasa_core/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22))
2. write your code, tests and documentation
3. create a pull request describing your changes

You pull request will be reviewed by a maintainer, who might get
back to you about any necessary changes or questions. You will
also be asked to sign a
[Contributor License Agreement](https://cla-assistant.io/RasaHQ/rasa_core).


## Development Internals
### Running and changing the documentation
To build & edit the docs, first install all necessary dependencies:

```
brew install sphinx
pip install -r dev-requirements.txt
```

After the installation has finished, you can run and view the documentation
locally using:
```
make livedocs
```

Visit the local version of the docs at http://localhost:8000 in your browser.
You can now change the docs locally and the web page will automatically reload
and apply your changes.

## License
Licensed under the Apache License, Version 2.0.
Copyright 2019 Rasa Technologies GmbH. [Copy of the license](LICENSE.txt).

A list of the Licenses of the dependencies of the project can be found at
the bottom of the
[Libraries Summary](https://libraries.io/github/RasaHQ/rasa_core).
