# Rasa NLU
[![Join the chat at https://gitter.im/RasaHQ/rasa_nlu](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/RasaHQ/rasa_nlu?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/RasaHQ/rasa_nlu.svg?branch=master)](https://travis-ci.org/RasaHQ/rasa_nlu)
[![Coverage Status](https://coveralls.io/repos/github/RasaHQ/rasa_nlu/badge.svg?branch=master)](https://coveralls.io/github/RasaHQ/rasa_nlu?branch=master)
[![PyPI version](https://badge.fury.io/py/rasa_nlu.svg)](https://badge.fury.io/py/rasa_nlu)
[![Documentation Status](https://readthedocs.org/projects/rasa-nlu/badge/)](https://rasa-nlu.readthedocs.io/en/stable/)

Rasa NLU (Natural Language Understanding) is a tool for intent classification and entity extraction. You can think of Rasa NLU as a set of high level APIs for building your own language parser using existing NLP and ML libraries. Find out more on the [homepage of the project](https://rasa.ai/), where you can also sign up for the mailing list.

**Extended documentation:**
- [latest](https://rasa-nlu.readthedocs.io/en/latest/)&nbsp; (if you install from **github**) or 
- [stable](https://rasa-nlu.readthedocs.io/en/stable/) (if you install from **pypi**)

If you are new to Rasa NLU and want to create a bot, you should start with the [**tutorial**](http://rasa-nlu.readthedocs.io/en/stable/tutorial.html).

**Contents:**
- [Setup](#setup) 
- [FAQ](#faq)
- [How to contribute](#how-to-contribute)
- [Development Internals](#development-internals)
- [License](#license)


## Setup
### A. Install Locally
From pypi:
```
pip install rasa_nlu
```
From github:
```
git clone git@github.com:RasaHQ/rasa_nlu.git
cd rasa_nlu
pip install -r requirements.txt
pip install -e .
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
The intended audience is mainly __people developing bots__, starting from scratch or looking to find a a drop-in replacement for [wit](https://wit.ai), [LUIS](https://luis.ai), or [api.ai](https://api.ai). The setup process is designed to be as simple as possible. Rasa NLU is written in Python, but you can use it from any language through a [HTTP API](http://rasa-nlu.readthedocs.io/en/stable/http.html). If your project is written in Python you can [simply import the relevant classes](http://rasa-nlu.readthedocs.io/en/stable/python.html). If you're currently using wit/LUIS/api.ai, you just:

1. Download your app data from wit, LUIS, or api.ai and feed it into Rasa NLU
2. Run Rasa NLU on your machine and switch the URL of your wit/LUIS api calls to `localhost:5000/parse`.

### Why should I use Rasa NLU?
* You don't have to hand over your data to FB/MSFT/GOOG
* You don't have to make a `https` call to parse every message.
* You can tune models to work well on your particular use case.

These points are laid out in more detail in a [blog post](https://medium.com/lastmile-conversations/do-it-yourself-nlp-for-bot-developers-2e2da2817f3d). Rasa is a set of tools for building more advanced bots, developed by the company [Rasa](https://rasa.ai). Rasa NLU is the natural language understanding module, and the first component to be open sourced. 

### What languages does it support?
Short answer: English, German, and Spanish currently. 
Longer answer: If you want to add a new language, the key things you need are a tokenizer and a set of word vectors. More information can be found in the [language documentation](https://rasa-nlu.readthedocs.io/en/stable/languages.html).

## How to contribute
We are very happy to receive and merge your contributions. There is some more information about the style of the code and docs in the [documentation](http://rasa-nlu.readthedocs.io/en/stable/contribute.html).

In general the process is rather simple:
1. create an issue describing the feature you want to work on (or have a look at issues with the label [help wanted](https://github.com/RasaHQ/rasa_nlu/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22))
2. write your code, tests and documentation
3. create a pull request describing your changes

You pull request will be reviewed by a maintainer, who might get back to you about any necessary changes or questions.

## Development Internals

### Steps to release a new version
Releasing a new version is quite simple, as the packages are build and distributed by travis. The following things need to be done to release a new version
1. update [rasa_nlu/version.py](https://github.com/RasaHQ/rasa_nlu/blob/master/rasa_nlu/version.py) to reflect the correct version number
2. edit the [CHANGELOG.rst](https://github.com/RasaHQ/rasa_nlu/blob/master/CHANGELOG.rst), create a new section for the release (eg by moving the items from the collected master section) and create a new master logging section
3. edit the [migration guide](https://github.com/RasaHQ/rasa_nlu/blob/master/docs/migrations.rst) to provide assistance for users updating to the new version 
4. commit all the above changes and tag a new release, e.g. using 
    ```
    git tag -f 0.7.0 -m "Some helpful line describing the release"
    git push origin master --tags
    ```
    travis will build this tag and push a package to [pypi](https://pypi.python.org/pypi/rasa_nlu)
5. only if it is a **major release**, a new branch should be created pointing to the same commit as the tag to allow for future minor patches, e.g.
    ```
    git checkout -b 0.7.x
    git push origin 0.7.x
    ```
## License
Licensed under the Apache License, Version 2.0. Copyright 2017 Lastmile Technologies GmbH. [Copy of the license](LICENSE.txt).

As a reference, the following contains a listing of the licenses of the different dependencies as of this writing. 
Licenses of minimal dependencies:

| required package | License            	|
|------------------|------------------------|
| gevent     	   | MIT                	|
| flask      	   | BSD 3-clause       	|
| boto3      	   | Apache License 2.0 	|
| typing     	   | PSF                	|
| future     	   | MIT                	|
| six        	   | MIT                	|
| jsonschema 	   | MIT                	|
| matplotlib       | PSF                    |

Licenses of optional dependencies (only required for certain components of Rasa NLU. Hence, they are optional):

| optional package     | License            	    |
|----------------------|----------------------------|
| MITIE     	       | Boost Software License 1.0 |
| spacy      	       | MIT       	                |
| scikit-learn         | BSD 3-clause             	|
| scipy                | BSD 3-clause             	|
| numpy                | BSD 3-clause             	|
| duckling     	       | Apache License 2.0         |
| sklearn-crfsuite     | MIT                     	|
| cloudpickle          | BSD 3-clause             	|
| google-cloud-storage | Apache License 2.0    	    |
