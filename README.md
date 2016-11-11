# rasa_nlu
[ ![Codeship Status for amn41/rasa_nlu](https://app.codeship.com/projects/8ab4d1e0-8a27-0134-70ee-5a7c9acf56e8/status?branch=master)](https://app.codeship.com/projects/184386)

**preface: if you're reading this now, you're an alpha tester of this code - your feedback is super valuable! thanks for trying it out**

## Motivation

rasa_nlu is a tool for intent classification and entity extraction. 
You can think of rasa_nlu as a set of high level APIs for building your own language parser using existing NLP and ML libraries.
The intended audience is mainly people developing bots. 
It can be used as a drop-in replacement for [wit](https://wit.ai) or [LUIS](https://luis.ai), but works as a local service rather than a web API. 

The setup process is designed to be as simple as possible. If you're currently using wit or LUIS, you just:
1. download your app data from wit or LUIS and feed it into rasa_nlu
2. run rasa_nlu on your machine and switch the URL of your wit/LUIS api calls to `localhost:5000/parse`.

Reasons you might use this over one of the aforementioned services: 
- you don't have to hand over your data to FB/MSFT/GOOG
- you don't have to make a `https` call every time.
- you can tune models to work well on your particular use case.

These points are laid out in more detail in a [blog post](https://medium.com/lastmile-conversations/do-it-yourself-nlp-for-bot-developers-2e2da2817f3d).

rasa_nlu is written in Python, but it you can use it from any language through a HTTP API. 
If your project *is* written in Python you can simply import the relevant classes.

rasa is a set of tools for building more advanced bots, developed by [LASTMILE](https://golastmile.com). This is the natural language understanding module, and the first component to be open sourced. 
 
## Getting Started
```bash
python setup.py install
python -m rasa_nlu.server -e wit &

curl 'http://localhost:5000/parse?q=hello'
# returns e.g. '{"intent":"greet","entities":[]}'
```

There you go! you just parsed some text. Important command line options for `rasa_nlu.server` are as follows:
- emulate: which service to emulate, can be 'wit' or 'luis', or just leave blank for default mode. This only affects the format of the json response.
- server_model_dir: dir where your trained models are saved. If you leave this blank rasa_nlu will just use a naive keyword matcher.

run `python -m rasa_nlu.server -h` to see more details.


## Configuring a backend
rasa_nlu itself doesn't have any external requirements, but to do something useful with it you need to install & configure a backend. 

#### Option 1 : MITIE
The [MITIE](https://github.com/mit-nlp/MITIE) backend is all-inclusive, in the sense that it provides both the NLP and the ML parts.

`pip install git+https://github.com/mit-nlp/MITIE.git`
and then download the [MITIE models](https://github.com/mit-nlp/MITIE/releases/download/v0.4/MITIE-models-v0.2.tar.bz2). The file you need is `total_word_feature_extractor.dat`

#### Option 2 : spaCy + scikit-learn
You can also run using these two in combination. 
[spaCy](https://spacy.io/) is an excellent library for NLP tasks.
[scikit-learn](http://scikit-learn.org/) is a popular ML library.

```bash
pip install -U spacy
python -m spacy.en.download all
pip install -U scikit-learn
```

OR if you prefer (especially if you don't already have `numpy/scipy` installed), you can install scikit-learn by:

1. installing [anaconda](https://www.continuum.io/downloads)
2. `conda install scikit-learn`


<!---
- [NLTK](www.nltk.org/)
-->

## Creating your own language parser

As of now, rasa_nlu doesn't provide a tool to help you create & annotate training data. 
If you don't have an existing wit or LUIS app, you can try this example using the `data/demo-restaurants.json` file, or create your own json file in the same format. 

### Cloning an existing wit or LUIS app:

Download your data from wit or LUIS. When you export your model from wit you will get a zipped directory. The file you need is `expressions.json`.
If you're exporting from LUIS you get a single json file, and that's the one you need. Create a config file (json format) like this one:

```json
{
  "path" : "/path/to/models/",
  "data" : "expressions.json",
  "backend" : "mitie",
  "backends" : {
    "mitie": {
      "fe_file":"/path/to/total_word_feature_extractor.dat"
    }
  }
}
```

and then pass this file to the training script

```bash
python -m rasa_nlu.train -c config.json
```

you can also override any of the params in config.json with command line arguments. Run `python -m rasa_nlu.train -h` for details.

### Running the server with your newly trained models

After training you will have a new dir containing your models, e.g. `/path/to/models/model_XXXXXX`. 
Just pass this path to the `rasa_nlu.server` script:

```bash
python -m rasa_nlu.server -e wit -d '/path/to/models/model_XXXXXX'
```

<!---
### Using Rasa from python
Pretty simple really, just open your python interpreter and type:
```python
from rasa.backends import MITIEInterpreter

interpreter = MITIEInterpreter('data/intent_classifier.dat','data/ner.dat','data/total_word_feature_extractor.dat')
interpreter.parse("hello world")  # -> {'intent':'greet','entities':[]}
```
-->

### Improving your models
When the rasa_nlu server is running, it keeps track of all the predictions it's made and saves these to a log file. By default this is called `rasa_nlu_log.json`
You can fix any incorrect predictions and add them to your training set to improve your parser.

## Roadmap 
- better test coverage
- entity normalisation: as is, the named entity extractor will happily extract `cheap` & `inexpensive` as entities of the `expense` class, but will not tell you that these are realisations of the same underlying concept. You can easily handle that with a list of aliases in your code, but we want to offer a more elegant & generalisable solution. [Word Forms](https://github.com/gutfeeling/word_forms) looks promising.
- parsing structured data, e.g. dates. We might use [parsedatetime](https://pypi.python.org/pypi/parsedatetime/) or [parserator](https://github.com/datamade/parserator) or wit.ai's very own [duckling](https://duckling.wit.ai/). 
- python 3 support
- support for more (human) languages

## Troubleshooting
- not tested with python 3, so probably won't work


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
