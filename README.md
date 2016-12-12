# rasa NLU
[![Build Status](https://travis-ci.org/golastmile/rasa_nlu.svg?branch=master)](https://travis-ci.org/golastmile/rasa_nlu)
[![Coverage Status](https://coveralls.io/repos/github/golastmile/rasa_nlu/badge.svg?branch=master)](https://coveralls.io/github/golastmile/rasa_nlu?branch=master)
[![Documentation Status](https://readthedocs.org/projects/rasa-nlu/badge/?version=latest)](http://rasa-nlu.readthedocs.io/en/latest/?badge=latest)

documentation is [here](http://rasa-nlu.readthedocs.io/)
homepage is here [here](https://rasa.ai/)

### Deploying to Docker Cloud
[![Deploy to Docker Cloud](https://files.cloud.docker.com/images/deploy-to-dockercloud.svg)](https://cloud.docker.com/stack/deploy/)
Click the button to deploy a rasa NLU server to Docker Cloud.


## Motivation

rasa NLU is a tool for intent classification and entity extraction. 
You can think of rasa NLU as a set of high level APIs for building your own language parser using existing NLP and ML libraries.
The intended audience is mainly people developing bots. 
It can be used as a drop-in replacement for [wit](https://wit.ai) or [LUIS](https://luis.ai), but works as a local service rather than a web API. 

The setup process is designed to be as simple as possible. If you're currently using wit or LUIS, you just:
1. download your app data from wit or LUIS and feed it into rasa NLU
2. run rasa NLU on your machine and switch the URL of your wit/LUIS api calls to `localhost:5000/parse`.

Reasons you might use this over one of the aforementioned services: 
- you don't have to hand over your data to FB/MSFT/GOOG
- you don't have to make a `https` call every time.
- you can tune models to work well on your particular use case.

These points are laid out in more detail in a [blog post](https://medium.com/lastmile-conversations/do-it-yourself-nlp-for-bot-developers-2e2da2817f3d).

rasa NLU is written in Python, but it you can use it from any language through a HTTP API. 
If your project *is* written in Python you can simply import the relevant classes.

rasa is a set of tools for building more advanced bots, developed by [LASTMILE](https://golastmile.com). This is the natural language understanding module, and the first component to be open sourced. 

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
