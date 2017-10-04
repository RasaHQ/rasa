# Rasa Core
[![Join the chat at https://gitter.im/RasaHQ/rasa_core](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/RasaHQ/rasa_core?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.com/RasaHQ/rasa_core.svg?token=EvwqtzR3SyKxFaNKbxau&branch=master)](https://travis-ci.com/RasaHQ/rasa_core)

### [Documentation](https://core.rasa.ai)

## Introduction

Rasa Core is a framework for building conversational software, which includes:
- chatbots on Messenger
- Slack bots
- Alexa Skills
- Google Home Actions

etc. 

Rasa Core's primary purpose is to help you build contextual, layered conversations with lots of back-and-forth.
To have a real conversation, you need to have some memory and build on things that were said earlier.
Rasa Core lets you do that in a scalable way. 

There's a lot more background information in this [blog post](https://medium.com/rasa-blog/a-new-approach-to-conversational-software-2e64a5d05f2a)

## Where to get help

There is extensive documentation:

- [master](https://rasahq.github.io/rasa_core/master/)&nbsp; (if you install from **github**) or 
- [stable](https://rasahq.github.io/rasa_core/)&nbsp;&nbsp; (if you install from **pypi**)


Please use [gitter](https://gitter.im/RasaHQ/rasa_core) for quick answers to 
questions.



#### README Contents:
- [Setup](#setup) 
- [How to contribute](#how-to-contribute)
- [Development Internals](#development-internals)
- [License](#license)

## Setup
There isn't a released pypi package yet. Hence, you need to clone and install 
the package from the github repository. For a more detailed description, please 
visit the [**Installation page**](https://rasahq.github.io/rasa_core/installation.html) 
of the docs.

To install, run:
```bash
git clone https://github.com/RasaHQ/rasa_core.git
cd rasa_core
pip install -r requirements.txt
pip install -e .
```


## How to contribute
We are very happy to receive and merge your contributions. There is some more 
information about the style of the code and docs in the documentation.

In general the process is rather simple:
1. create an issue describing the feature you want to work on (or have a look 
at issues labeled [good first issue](https://github.com/RasaHQ/rasa_core/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22))
2. write your code, tests, and documentation
3. create a pull request describing your changes

You pull request will be reviewed by a maintainer, who might get back to you 
about any necessary changes or questions.

## Development Internals
### Running and changing the documentation
To build & edit the docs, first install all necessary dependencies:

```
brew install sphinx
pip install -r dev-requirements.txt
```

After the installation has finished, you can run and view the documentation 
locally using
```
make livedocs
```

Visit the local version of the docs at http://localhost:8000 in your browser. 
You can now change the docs locally and the web page will automatically reload
and apply your changes.

## License
Licensed under the Apache License, Version 2.0. Copyright 2017 
Rasa Technologies GmbH. [Copy of the license](LICENSE.txt).

