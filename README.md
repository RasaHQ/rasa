# rasa NLU
[![Join the chat at https://gitter.im/golastmile/rasa_nlu](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/golastmile/rasa_nlu?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/golastmile/rasa_nlu.svg?branch=master)](https://travis-ci.org/golastmile/rasa_nlu)
[![Coverage Status](https://coveralls.io/repos/github/golastmile/rasa_nlu/badge.svg?branch=master)](https://coveralls.io/github/golastmile/rasa_nlu?branch=master)
[![PyPI version](https://badge.fury.io/py/rasa_nlu.svg)](https://badge.fury.io/py/rasa_nlu)
[![Documentation Status](https://readthedocs.org/projects/rasa-nlu/badge/)](https://rasa-nlu.readthedocs.io/en/stable/)

rasa NLU (Natural Language Understanding) is a tool for intent classification and entity extraction. You can think of rasa NLU as a set of high level APIs for building your own language parser using existing NLP and ML libraries. Find out more on the [homepage of the project](https://rasa.ai/), where you can also sign up for the mailing list.

**Extended documentation:**
- [latest](https://rasa-nlu.readthedocs.io/en/latest/)&nbsp; (if you install from **github**) or 
- [stable](https://rasa-nlu.readthedocs.io/en/stable/) (if you install from **pypi**)

If you are new to rasa NLU and want to create a bot, you should start with the [**tutorial**](http://rasa-nlu.readthedocs.io/en/stable/tutorial.html).

**Contents:**
- [Setup](#setup) 
- [FAQ](#faq)
- [How to contribute](#how-to-contribute)
- [License](#license)


## Setup
### A. Install Locally
From pypi:
```
pip install rasa_nlu
```
From github:
```
git clone git@github.com:golastmile/rasa_nlu.git
cd rasa_nlu
python setup.py install
```

To test the installation use (this will run a very stupid default model. you need to [train your own model](http://rasa-nlu.readthedocs.io/en/stable/tutorial.html) to do something useful!):
```
python -m rasa_nlu.server &
curl 'http://localhost:5000/parse?q=hello'
```

### B. Install with Docker
Before you start, ensure you have the latest version of docker engine on your machine. You can check if you have docker installed by typing ```docker -v``` in your terminal.

#### 1. Build the image:
```
docker build -t rasa_nlu .
``` 

#### 2. Start the web server:
```
docker run -p 5000:5000 rasa_nlu start
```

Caveat for Docker for Windows users: please share your C: in docker settings, and add ```-v C:\path\to\rasa_nlu:/app``` to your docker run commands for download and training to work correctly.

#### 3. Test it!
```
curl 'http://localhost:5000/parse?q=hello'
```

### C. (Experimental) Deploying to Docker Cloud
[![Deploy to Docker Cloud](https://files.cloud.docker.com/images/deploy-to-dockercloud.svg)](https://cloud.docker.com/stack/deploy/)

## FAQ

### Who is it for?
The intended audience is mainly __people developing bots__, starting from scratch or looking to find a a drop-in replacement for [wit](https://wit.ai), [LUIS](https://luis.ai), or [api.ai](https://api.ai). The setup process is designed to be as simple as possible. rasa NLU is written in Python, but you can use it from any language through a [HTTP API](http://rasa-nlu.readthedocs.io/en/stable/http.html). If your project is written in Python you can [simply import the relevant classes](http://rasa-nlu.readthedocs.io/en/stable/python.html). If you're currently using wit/LUIS/api.ai, you just:

1. Download your app data from wit, LUIS, or api.ai and feed it into rasa NLU
2. Run rasa NLU on your machine and switch the URL of your wit/LUIS api calls to `localhost:5000/parse`.

### Why should I use rasa NLU?
* You don't have to hand over your data to FB/MSFT/GOOG
* You don't have to make a `https` call to parse every message.
* You can tune models to work well on your particular use case.

These points are laid out in more detail in a [blog post](https://medium.com/lastmile-conversations/do-it-yourself-nlp-for-bot-developers-2e2da2817f3d). rasa is a set of tools for building more advanced bots, developed by [LASTMILE](https://golastmile.com). rasa NLU is the natural language understanding module, and the first component to be open sourced. 

### What languages does it support?
Short answer: English, German, and Spanish currently. 
Longer answer: If you want to add a new language, the key things you need are a tokenizer and a set of word vectors. More information can be found in the [language documentation](https://rasa-nlu.readthedocs.io/en/stable/languages.html).

## How to contribute
We are very happy to receive and merge your contributions. There is some more information about the style of the code and docs in the [documentation](http://rasa-nlu.readthedocs.io/en/stable/contribute.html).

In general the process is rather simple:
1. create an issue describing the feature you want to work on (or have a look at issues with the label [help wanted](https://github.com/golastmile/rasa_nlu/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22))
2. write your code, tests and documentation
3. create a pull request describing your changes

You pull request will be reviewed by a maintainer, who might get back to you about any necessary changes or questions.

## License
Copyright 2016 LastMile Technologies Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this project except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
