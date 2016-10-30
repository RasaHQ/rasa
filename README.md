# parsa
[ ![Codeship Status for amn41/parsa](https://app.codeship.com/projects/b06f6000-7444-0134-8053-76df66f7aa2d/status?branch=master)](https://app.codeship.com/projects/179147)

Parsa is a tool for intent classification and entity extraction. 
The intended audience is mainly people developing bots. 
It can be used as a drop-in replacement [wit](https://wit.ai), [LUIS](https://luis.ai), or [api.ai](https://api.ai), but as a local service rather than a web API. 
You can think of parsa as a set of high level APIs for building your own language parser using existing NLP and ML libraries.

Reasons you might use this over one of the aforementioned services: 
- you don't have to hand over your data to FB/MSFT/GOOG
- you don't have to make a `https` call every time.
- you can tune models to work well on your particular use case.

These points are laid out in more detail in a [blog post](https://medium.com/lastmile-conversations/do-it-yourself-nlp-for-bot-developers-2e2da2817f3d).

Parsa is written in Python, but it you can use it from any language through a HTTP API. 
If your project *is* written in Python you can simply import the relevant classes.
Training your models always happens in python, whereas you can use them in two different ways: (1) by instantiating the relevant `Interpreter` subclass in your python project, or (2) by running a simple http API locally (if you're not using python). The file `src/main.py` contains an example of the latter.
 
## Getting Started
```bash
python setup.py install
python -m parsa.server --mode=wit &
curl 'http://localhost:5000/parse?q=hello'
# returns e.g. '{"intent":"greet","entities":[]}'
```

There you go! you just parsed some text. The command line options for the `parsa.server` are as follows:
- mode: which service to emulate, can be 'wit' or 'luis', or just leave blank for default (clutter-free) mode.
- backend: which backend to use. default is to use a built in, extremely naive keyword matcher. Valid options are 'mitie', 'sklearn', 'spacy-keras'


## Configuring a backend
Parsa itself doesn't have any external requirements, but in order to make it useful you need to install & configure a backend. 

There are several supported NLP backends:

- [MITIE](https://github.com/mit-nlp/MITIE)
- [spaCy](https://github.com/spacy-io/spaCy)
- [NLTK](www.nltk.org/)

NB that if you use spaCy or NLTK you will also need to use a separate machine learning library like scikit-learn or keras.

#### MITIE
Using MITIE is the simplest way to get started. Just install with
`pip install git+https://github.com/mit-nlp/MITIE.git`
and then download the [MITIE models](https://github.com/mit-nlp/MITIE/releases/download/v0.4/MITIE-models-v0.2.tar.bz2)

#### SpaCy,  NLTK
Install one of the above & then also a ML lib, e.g. scikit-learn or keras. 


## Creating your own language parser
### Cloning an existing wit or LUIS app:

Download your data from wit or LUIS. When you export your model from wit you will get a zipped directory. The file you need is `expressions.json`.
If you're exporting from LUIS you get a single json file, and that's the one you need. Just pass your data file to the train script:

```bash
python -m parsa.train --data=expressions.json --backend=mitie --path=/save/models/here
```
Once you’ve trained your model, you will have a few new files in the dir you specified. You can then run a parsa server which runs your new model: 
```bash
python -mparsa.server --mode=wit --backend=mitie --path=/path/to/models
```


### Using Parsa from python
Pretty simple really, just open your python interpreter and type:
```python
from parsa import MITIEInterpreter
interpreter = MITIEInterpreter('data/intent_classifier.dat','data/ner.dat','data/total_word_feature_extractor.dat')
interpreter.parse("hello world")  # -> {'intent':'greet','entities':[]}
```


## Roadmap (message me if you'd like to submit a PR)
- entity normalisation: as is, the named entity extractor will happily extract `cheap` & `inexpensive` as entities of the `expense` class, but will not tell you that these are realisations of the same underlying concept. You can easily handle that with a list of aliases in your code, but we want to offer a more elegant & generalisable solution.
- parsing structured data, e.g. dates. We might use [parsedatetime](https://pypi.python.org/pypi/parsedatetime/) or possibly wit.ai's very own [duckling](https://duckling.wit.ai/). 
- support for more languages

