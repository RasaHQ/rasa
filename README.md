# Rasa Core
[![Join the chat at https://gitter.im/RasaHQ/rasa_core](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/RasaHQ/rasa_core?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.com/RasaHQ/rasa_dm.svg?token=EvwqtzR3SyKxFaNKbxau&branch=master)](https://travis-ci.com/RasaHQ/rasa_dm)
[![Documentation Status](https://readthedocs.com/projects/lastmile-rasa-dm/badge/?version=latest)](https://lastmile-rasa-dm.readthedocs-hosted.com/en/latest/?badge=master)


## EARLY ACCESS
Please use [gitter](https://gitter.im/RasaHQ/rasa_core) for quick answers to 
questions. Please also fill out the [survey](https://alan328.typeform.com/to/KUf7Zw). 
This feedback will help us make the framework **much** better.

The idea behind Rasa Core is that conversational software is not like normal 
software. Rather than writing a bunch of if-else statements, you provide training 
conversations which are used to create a probabilistic model of what should happen 
next. Why would you want to do this?

- debugging is easier
- greater flexibility
- can improve automatically over time

[Rasa](https://rasa.ai/) is designed to be composable and hackable, with loosely 
coupled pieces interacting through simple APIs. This means that you can use it if 
you don't know anything about machine learning, but if you do it's easy to experiment.

#### Extended documentation:
- [master](https://lastmile-rasa-dm.readthedocs-hosted.com/en/latest/)&nbsp; (if you install from **github**) or 
- [stable](https://lastmile-rasa-dm.readthedocs-hosted.com/en/stable/)&nbsp;&nbsp; (if you install from **pypi**)

If you are new to Rasa and want to create a bot, you should start with 
the [**installation**](https://lastmile-rasa-dm.readthedocs-hosted.com/en/latest/intro.html) 
and head to the [**basic tutorial**](https://lastmile-rasa-dm.readthedocs-hosted.com/en/latest/tutorial.html).


#### README Contents:
- [Setup](#setup) 
- [How to contribute](#how-to-contribute)
- [Development Internals](#development-internals)
- [License](#license)

## Setup
There isn't a released pypi package yet. Hence, you need to clone and install 
the package from the github repository. For a more detailed description, please 
visit the [**Installation page**](https://lastmile-rasa-dm.readthedocs-hosted.com/en/latest/intro.html) 
of the docs.

To install, run:
```bash
git clone https://github.com/RasaHQ/rasa_dm.git
cd rasa_dm
pip install -r requirements.txt
pip install -e .
```

This will install the application and necessary requirements. We use rasa NLU 
for intent classification & entity extraction, but you are free to use other 
NLU services like wit.ai, api.ai, or LUIS.ai.

## How to contribute
We are very happy to receive and merge your contributions. There is some more 
information about the style of the code and docs in the documentation.

In general the process is rather simple:
1. create an issue describing the feature you want to work on (or have a look 
at issues with the label [help wanted](https://github.com/RasaHQ/rasa_dm/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22))
2. write your code, tests, and documentation
3. create a pull request describing your changes

You pull request will be reviewed by a maintainer, who might get back to you 
about any necessary changes or questions.

## Development Internals
### Running and changing the documentation
To build & edit the docs, first install all necessary dependencies:

```
brew install sphinx
pip install sphinx_rtd_theme
pip install sphinx-autobuild
```

After the installation has finished, you can run and view the documentation 
locally using
```
cd docs
sphinx-apidoc -o . ../rasa_core
make livehtml
```

Visit the local version of the docs at http://localhost:8000 in your browser.

## License
Licensed under the Apache License, Version 2.0. Copyright 2017 
Rasa Technologies GmbH. [Copy of the license](LICENSE.txt).

