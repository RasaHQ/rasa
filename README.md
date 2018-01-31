# Rasa NLU
[![Join the chat at https://gitter.im/RasaHQ/rasa_nlu](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/RasaHQ/rasa_nlu?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/RasaHQ/rasa_nlu.svg?branch=master)](https://travis-ci.org/RasaHQ/rasa_nlu)
[![Coverage Status](https://coveralls.io/repos/github/RasaHQ/rasa_nlu/badge.svg?branch=master)](https://coveralls.io/github/RasaHQ/rasa_nlu?branch=master)
[![PyPI version](https://badge.fury.io/py/rasa_nlu.svg)](https://badge.fury.io/py/rasa_nlu)
[![Documentation Status](https://img.shields.io/badge/docs-stable-brightgreen.svg)](https://rasahq.github.io/rasa_nlu/)
[![roadmap badge](https://img.shields.io/badge/visit%20the-roadmap-blue.svg)](https://github.com/RasaHQ/rasa_nlu/projects/2)

Rasa NLU (Natural Language Understanding) is a tool for understanding what is being said in short pieces of text.
For example, taking a short message like:

> *"I'm looking for a Mexican restaurant in the center of town"*

And returning structured data like:

```
  intent: search_restaurant
  entities: 
    - cuisine : Mexican
    - location : center
```

Rasa NLU is primarily used to build chatbots and voice apps, where this is called intent classification and entity extraction.
To use Rasa, *you have to provide some training data*.
That is, a set of messages which you've already labelled with their intents and entities.
Rasa then uses machine learning to pick up patterns and generalise to unseen sentences. 

You can think of Rasa NLU as a set of high level APIs for building your own language parser using existing NLP and ML libraries.

If you are new to Rasa NLU and want to create a bot, you should start with the [**tutorial**](https://rasahq.github.io/rasa_nlu/tutorial.html).

- **What does Rasa NLU do? 🤔** [Read About the Rasa Stack](http://rasa.com/products/rasa-stack/)

- **I'd like to read the detailed docs 🤓** [Read The Docs](https://nlu.rasa.ai)

- **I'm ready to install Rasa NLU! 🚀** [Installation](https://nlu.rasa.ai/installation.html)

- **I have a question ❓** [Gitter channel](https://gitter.im/RasaHQ/rasa_nlu)

- **I would like to contribute 🤗** [How to contribute](#how-to-contribute)

# Quick Install

Fur the full installation instructions, please head over to the documenation: [Installation](https://nlu.rasa.ai/installation.html)

**Via Docker Image**
From docker hub:
```
docker run -p 5000:5000 rasa/rasa_nlu:latest-full
```
(for more docker installation options see [Advanced Docker Installation](#advanced-docker))

**Via Python Library**
From pypi:
```
pip install rasa_nlu
python -m rasa_nlu.server &
```
(for more python installation options see [Advanced Python Installation](#advanced-python))

### Basic test
The below command can be executed for either method used above.
```
curl 'http://localhost:5000/parse?q=hello'
```

# Example use

### Get the Server Status
```
curl 'http://localhost:5000/status'
```

### Check the Server Version
```
curl 'http://localhost:5000/version'
```

### Training New Models
[Examples](https://github.com/RasaHQ/rasa_nlu/tree/master/data/examples/rasa) and [Documentation](https://rasahq.github.io/rasa_nlu/dataformat.html) of the training data format are provided. But as a quick start execute the below command to train a new model

#### Json format
```
curl 'https://raw.githubusercontent.com/RasaHQ/rasa_nlu/master/data/examples/rasa/demo-rasa.json' | \
curl --request POST --header 'content-type: application/json' -d@- --url 'localhost:5000/train?project=test_model&pipeline=keyword'
```

This will train a simple keyword based models (not usable for anything but this demo). For better
pipelines consult the documentation.

#### Markdown format
```
wget 'https://raw.githubusercontent.com/RasaHQ/rasa_nlu/master/data/examples/rasa/demo-rasa.md'
curl --request POST --data-binary @demo-rasa.md --url 'localhost:5000/train?project=test_model&pipeline=keyword'
```

The above command does the following:
1. It Fetches some of the example data in the repo
2. It `POSTS` that data to the `/train` endpoint and names the model `/name=test_model`

### Parsing New Requests
Make sure the above command has finished before executing the below. You can check with the `/status` command above.
```
curl 'http://localhost:5000/parse?q=hello&model=test_model'
```

# FAQ

### Who is it for?
The intended audience is mainly __people developing bots__, starting from scratch or looking to find a a drop-in replacement for [wit](https://wit.ai), [LUIS](https://luis.ai), or [Dialogflow](https://dialogflow.com). The setup process is designed to be as simple as possible. Rasa NLU is written in Python, but you can use it from any language through a [HTTP API](https://rasahq.github.io/rasa_nlu/http.html). If your project is written in Python you can [simply import the relevant classes](https://rasahq.github.io/rasa_nlu/python.html). If you're currently using wit/LUIS/Dialogflow, you just:

1. Download your app data from wit, LUIS, or Dialogflow and feed it into Rasa NLU
2. Run Rasa NLU on your machine and switch the URL of your wit/LUIS api calls to `localhost:5000/parse`.

### Why should I use Rasa NLU?
* You don't have to hand over your data to FB/MSFT/GOOG
* You don't have to make a `https` call to parse every message.
* You can tune models to work well on your particular use case.

These points are laid out in more detail in a [blog post](https://medium.com/lastmile-conversations/do-it-yourself-nlp-for-bot-developers-2e2da2817f3d). Rasa is a set of tools for building more advanced bots, developed by the company [Rasa](https://rasa.ai). Rasa NLU is the natural language understanding module, and the first component to be open sourced. 

### What languages does it support?
Short answer: English, German, and Spanish currently. 
Longer answer: If you want to add a new language, the key things you need are a tokenizer and a set of word vectors. More information can be found in the [language documentation](https://rasahq.github.io/rasa_nlu/languages.html).

### How to contribute
We are very happy to receive and merge your contributions. There is some more information about the style of the code and docs in the [documentation](https://rasahq.github.io/rasa_nlu/contribute.html).

In general the process is rather simple:
1. create an issue describing the feature you want to work on (or have a look at issues with the label [help wanted](https://github.com/RasaHQ/rasa_nlu/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22))
2. write your code, tests and documentation
3. create a pull request describing your changes

You pull request will be reviewed by a maintainer, who might get back to you about any necessary changes or questions. You will also be asked to sign the [Contributor License Agreement](https://cla-assistant.io/RasaHQ/rasa_nlu)

# Advanced installation
### Advanced Python
From github:
```
git clone git@github.com:RasaHQ/rasa_nlu.git
cd rasa_nlu
pip install -r requirements.txt
pip install -e .
```

For local development make sure you install the development requirements:
```
pip install -r alt_requirements/requirements_dev.txt
pip install -e .
```

To test the installation use (this will run a very stupid default model. you need to [train your own model](https://rasahq.github.io/rasa_nlu/tutorial.html) to do something useful!):

### Advanced Docker
Before you start, ensure you have the latest version of docker engine on your machine. You can check if you have docker installed by typing ```docker -v``` in your terminal.

To see all available builds go to the [Rasa docker hub](https://hub.docker.com/r/rasa/rasa_nlu/), but to get up and going the quickest just run:
```
docker run -p 5000:5000 rasa/rasa_nlu:latest-full
```

There are also three volumes, which you may want to map: `/app/projects`, `/app/logs`, and `/app/data`. It is also possible to override the config file used by the server by mapping a new config file to the volume `/app/config.json`. For complete docker usage instructions go to the official [docker hub readme](https://hub.docker.com/r/rasa/rasa_nlu/).

To test run the below command after the container has started. For more info on using the HTTP API see [here](https://rasahq.github.io/rasa_nlu/http.html#endpoints)
```
curl 'http://localhost:5000/parse?q=hello'
```

### Docker Cloud
Warning! setting up Docker Cloud is quite involved - this method isn't recommended unless you've already configured Docker Cloud Nodes (or swarms)

[![Deploy to Docker Cloud](https://files.cloud.docker.com/images/deploy-to-dockercloud.svg)](https://cloud.docker.com/stack/deploy/?repo=https://github.com/RasaHQ/rasa_nlu/tree/master/docker)

### Install Pretrained Models for Spacy & Mitie
In order to use the Spacy or Mitie backends make sure you have one of their pretrained models installed.
```
python -m spacy download en
```

To download the Mitie model run and place it in a location that you can 
reference in your configuration during model training:
```
wget https://github.com/mit-nlp/MITIE/releases/download/v0.4/MITIE-models-v0.2.tar.bz2
tar jxf MITIE-models-v0.2.tar.bz2
```

If you want to run the tests, you need to copy the model into the Rasa folder:

```
cp MITIE-models/english/total_word_feature_extractor.dat RASA_NLU_ROOT/data/
``` 

Where `RASA_NLU_ROOT` points to your Rasa installation directory.

# Development Internals

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

### Running the Tests
In order to run the tests make sure that you have the development requirements installed.
```
make test
```

## License
Licensed under the Apache License, Version 2.0. Copyright 2018 Rasa Technologies GmbH. [Copy of the license](LICENSE.txt).

As a reference, the following contains a listing of the licenses of the different dependencies as of this writing. 
Licenses of minimal dependencies:

| required package | License              |
|------------------|----------------------|
| gevent           | MIT                  |
| klein            | MIT                  |
| boto3            | Apache License 2.0   |
| typing           | PSF                  |
| future           | MIT                  |
| six              | MIT                  |
| jsonschema       | MIT                  |
| matplotlib       | PSF                  |
| requests         | Apache Licence 2.0   |

Licenses of optional dependencies (only required for certain components of Rasa NLU. Hence, they are optional): 

| optional package     | License                    |
|----------------------|----------------------------|
| MITIE                | Boost Software License 1.0 |
| spacy                | MIT                        |
| scikit-learn         | BSD 3-clause               |
| scipy                | BSD 3-clause               |
| numpy                | BSD 3-clause               |
| duckling             | Apache License 2.0         |
| sklearn-crfsuite     | MIT                        |
| cloudpickle          | BSD 3-clause               |
| google-cloud-storage | Apache License 2.0         |
